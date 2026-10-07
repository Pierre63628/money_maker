"""Numba simulation loop for the bar-based portfolio backtester.

Per bar ``t`` (all symbols), in this order:
  1. financing (if any) is charged on the gross position, then funding settles on the position held
     *before* trading, at the open price;
  2. taker orders fill at the open (+ spread/slippage/impact, + fee), subject to a
     liquidity cap; unfilled remainder stays pending for the next bar. Maker mode: a limit
     order at the previous close (+/- offset) fills at its limit only if the bar trades
     THROUGH it by ``trade_through_bps`` (queue-position haircut); if the open is already
     beyond the limit it is marketable and fills as taker; risk-reducing orders fall back
     to taker after ``maker_fallback_bars`` unfilled bars;
  3. intrabar liquidation check against the adverse extreme (low for longs,
     high for shorts) of every position simultaneously (conservative);
  4. mark-to-market at the close; forced exit where the symbol's data ends or a gap
     longer than ``max_gap_bars`` starts (delisting / ticker reuse).

Accounting: equity = cash + sum(qty * price). Cash may be negative (leverage).
Trade PnL = sum(-dq * fill) - fees - funding (+ mark value while open), so the
sum of trade PnLs equals the equity change exactly.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# trade state columns (per symbol)
S_ACTIVE, S_ENTRY_T, S_DIR, S_CASH, S_FEES, S_SLIP, S_FUND, S_ENTRY_EQ = 0, 1, 2, 3, 4, 5, 6, 7
S_MAX_NOTIONAL, S_IN_QTY, S_IN_VAL, S_OUT_QTY, S_OUT_VAL, S_MAX_QTY = 8, 9, 10, 11, 12, 13
N_STATE = 14

# trade record columns
TRADE_COLUMNS = (
    "symbol_idx", "entry_bar", "exit_bar", "direction", "entry_price", "exit_price",
    "max_notional", "pnl", "fees", "spread_slippage", "funding", "entry_equity",
    "is_open", "exit_reason", "max_qty",
)
N_TRADE_COLS = len(TRADE_COLUMNS)
REASON_SIGNAL, REASON_LIQUIDATION, REASON_DELISTING, REASON_END_OF_DATA = 0, 1, 2, 3


@njit(cache=True)
def _grow(trades):
    bigger = np.empty((trades.shape[0] * 2 + 16, trades.shape[1]))
    bigger[: trades.shape[0]] = trades
    return bigger


@njit(cache=True)
def _open_trade(st, i, t, direction, equity):
    for k in range(N_STATE):
        st[i, k] = 0.0
    st[i, S_ACTIVE] = 1.0
    st[i, S_ENTRY_T] = t
    st[i, S_DIR] = direction
    st[i, S_ENTRY_EQ] = equity


@njit(cache=True)
def _add_fill(st, i, dq, fill, fee, slip):
    st[i, S_CASH] -= dq * fill
    st[i, S_FEES] += fee
    st[i, S_SLIP] += slip
    if dq * st[i, S_DIR] > 0:
        st[i, S_IN_QTY] += abs(dq)
        st[i, S_IN_VAL] += abs(dq) * fill
    else:
        st[i, S_OUT_QTY] += abs(dq)
        st[i, S_OUT_VAL] += abs(dq) * fill


@njit(cache=True)
def _close_trade(st, i, t, trades, n, reason, mark_value, is_open):
    if n >= trades.shape[0]:
        trades = _grow(trades)
    trades[n, 0] = i
    trades[n, 1] = st[i, S_ENTRY_T]
    trades[n, 2] = t
    trades[n, 3] = st[i, S_DIR]
    trades[n, 4] = st[i, S_IN_VAL] / st[i, S_IN_QTY] if st[i, S_IN_QTY] > 0 else np.nan
    trades[n, 5] = st[i, S_OUT_VAL] / st[i, S_OUT_QTY] if st[i, S_OUT_QTY] > 0 else np.nan
    trades[n, 6] = st[i, S_MAX_NOTIONAL]
    trades[n, 7] = st[i, S_CASH] + mark_value - st[i, S_FEES] - st[i, S_FUND]
    trades[n, 8] = st[i, S_FEES]
    trades[n, 9] = st[i, S_SLIP]
    trades[n, 10] = st[i, S_FUND]
    trades[n, 11] = st[i, S_ENTRY_EQ]
    trades[n, 12] = 1.0 if is_open else 0.0
    trades[n, 13] = reason
    trades[n, 14] = st[i, S_MAX_QTY]
    st[i, S_ACTIVE] = 0.0
    return trades, n + 1


@njit(cache=True)
def _fill(st, qty, i, t, dq, fill, fee, slip, equity, trades, n, reason, px):
    """Apply one fill to position ``i`` and the trade ledger (handles opens, reductions, flips)."""
    old = qty[i]
    new = old + dq
    if old == 0.0:
        _open_trade(st, i, t, 1.0 if dq > 0 else -1.0, equity)
        _add_fill(st, i, dq, fill, fee, slip)
    elif old * new < 0.0:
        frac = abs(old) / abs(dq)
        _add_fill(st, i, -old, fill, fee * frac, slip * frac)
        trades, n = _close_trade(st, i, t, trades, n, reason, 0.0, False)
        _open_trade(st, i, t, 1.0 if new > 0 else -1.0, equity)
        _add_fill(st, i, new, fill, fee * (1.0 - frac), slip * (1.0 - frac))
    else:
        _add_fill(st, i, dq, fill, fee, slip)
        if new == 0.0:
            trades, n = _close_trade(st, i, t, trades, n, reason, 0.0, False)
    qty[i] = new
    if new != 0.0:
        notional = abs(new) * px
        if notional > st[i, S_MAX_NOTIONAL]:
            st[i, S_MAX_NOTIONAL] = notional
        if abs(new) > st[i, S_MAX_QTY]:
            st[i, S_MAX_QTY] = abs(new)
    return trades, n


@njit(cache=True)
def simulate(open_, high, low, close, target, change, funding, adv, sigma, half_spread_bps,
             force_exit, initial_capital, fee_rate, slippage_bps, impact_coef, max_participation,
             maintenance_margin, liquidation_fee, every_bar, rebalance_threshold, max_gross_leverage,
             leverage_tolerance, min_trade_notional, delist_penalty_bps, sigma_fallback, apply_funding,
             maker, maker_fee_rate, limit_offset_bps, trade_through_bps, maker_fallback_bars,
             financing_rate_per_bar, financing_any):
    T, N = close.shape
    cash = initial_capital
    qty = np.zeros(N)
    last_px = np.full(N, np.nan)
    pending = np.zeros(N, dtype=np.bool_)
    wait = np.zeros(N, dtype=np.int64)
    n_maker = 0
    n_taker = 0
    n_waits = 0
    st = np.zeros((N, N_STATE))
    trades = np.empty((64, N_TRADE_COLS))
    n_trades = 0

    equity = np.full(T, np.nan)
    gross = np.zeros(T)
    net = np.zeros(T)
    fees = np.zeros(T)
    spread_slip = np.zeros(T)
    funding_paid = np.zeros(T)
    traded_notional = np.zeros(T)
    n_partial = 0
    liquidated_at = -1
    ruined_at = -1
    px_open = np.empty(N)

    for t in range(T):
        # ---- prices available at the open
        for i in range(N):
            o = open_[t, i]
            px_open[i] = o if not math.isnan(o) else last_px[i]

        # ---- 0. financing charged on the gross position (CFD-style: both legs pay)
        if financing_any:
            for i in range(N):
                if qty[i] != 0.0 and not math.isnan(px_open[i]):
                    charge = abs(qty[i]) * px_open[i] * financing_rate_per_bar[i]
                    cash -= charge
                    funding_paid[t] += charge
                    st[i, S_FUND] += charge

        # ---- 1. funding on the pre-trade position
        if apply_funding:
            for i in range(N):
                rate = funding[t, i]
                if qty[i] != 0.0 and not math.isnan(rate) and not math.isnan(px_open[i]):
                    pay = qty[i] * px_open[i] * rate
                    cash -= pay
                    funding_paid[t] += pay
                    st[i, S_FUND] += pay

        eq_open = cash
        gross_open = 0.0
        for i in range(N):
            if qty[i] != 0.0:
                eq_open += qty[i] * px_open[i]
                gross_open += abs(qty[i] * px_open[i])

        if eq_open <= 0.0:
            ruined_at = t
            for k in range(t, T):
                equity[k] = 0.0
            break

        # ---- 2. decide which symbols need orders
        for i in range(N):
            if change[t, i]:
                pending[i] = True
                wait[i] = 0
            elif every_bar and not math.isnan(px_open[i]):
                w_cur = qty[i] * px_open[i] / eq_open
                if abs(w_cur - target[t, i]) > rebalance_threshold:
                    pending[i] = True
        if gross_open / eq_open > max_gross_leverage * (1.0 + leverage_tolerance):
            for i in range(N):
                if qty[i] != 0.0 or target[t, i] != 0.0:
                    pending[i] = True

        # ---- 3. fills at the open
        for i in range(N):
            if not pending[i]:
                continue
            o = open_[t, i]
            a = adv[t, i]
            if math.isnan(o) or math.isnan(a) or a <= 0.0:
                continue  # no tradable price / unknown liquidity: keep pending
            desired = target[t, i] * eq_open / o
            dq = desired - qty[i]
            notional = abs(dq) * o
            closing = desired == 0.0 and qty[i] != 0.0
            if notional < min_trade_notional and not closing:
                pending[i] = False
                continue
            if notional == 0.0:
                pending[i] = False
                continue
            cap = max_participation * a
            partial = notional > cap
            if partial:
                dq = math.copysign(cap / o, dq)
                notional = cap
            if maker:
                # resting limit order placed at the decision (previous close), re-quoted every bar
                ref = last_px[i]
                reducing = abs(desired) < abs(qty[i]) and desired * qty[i] >= 0.0
                fallback = reducing and maker_fallback_bars > 0 and wait[i] >= maker_fallback_bars
                if not math.isnan(ref) and not fallback:
                    if dq > 0:
                        limit = ref * (1.0 - limit_offset_bps / 1e4)
                        marketable = o <= limit
                        touched = low[t, i] <= limit * (1.0 - trade_through_bps / 1e4)
                    else:
                        limit = ref * (1.0 + limit_offset_bps / 1e4)
                        marketable = o >= limit
                        touched = high[t, i] >= limit * (1.0 + trade_through_bps / 1e4)
                    if not marketable:
                        if not touched:
                            wait[i] += 1
                            n_waits += 1
                            continue  # not filled this bar: keep the order working
                        fee = abs(dq) * limit * maker_fee_rate[i]
                        maker_notional = abs(dq) * limit
                        cash -= dq * limit + fee
                        fees[t] += fee
                        traded_notional[t] += maker_notional
                        trades, n_trades = _fill(st, qty, i, t, dq, limit, fee, 0.0, eq_open, trades, n_trades,
                                                 REASON_SIGNAL, limit)
                        n_maker += 1
                        wait[i] = 0
                        if partial:
                            n_partial += 1
                        else:
                            pending[i] = False
                        continue
            if partial:
                n_partial += 1
            else:
                pending[i] = False
            n_taker += 1
            wait[i] = 0
            sig = sigma[t, i]
            if math.isnan(sig):
                sig = sigma_fallback
            cost_bps = half_spread_bps[t, i] + slippage_bps + impact_coef * sig * 1e4 * math.sqrt(notional / a)
            fill = o * (1.0 + math.copysign(cost_bps, dq) / 1e4)
            fee = abs(dq) * fill * fee_rate[i]
            slip = notional * cost_bps / 1e4
            cash -= dq * fill + fee
            fees[t] += fee
            spread_slip[t] += slip
            traded_notional[t] += notional
            trades, n_trades = _fill(st, qty, i, t, dq, fill, fee, slip, eq_open, trades, n_trades,
                                     REASON_SIGNAL, o)

        # ---- 4. intrabar liquidation (all adverse extremes at once: conservative)
        worst = cash
        gross_worst = 0.0
        for i in range(N):
            if qty[i] != 0.0:
                p = low[t, i] if qty[i] > 0 else high[t, i]
                if math.isnan(p):
                    p = px_open[i]
                worst += qty[i] * p
                gross_worst += abs(qty[i]) * p
        if gross_worst > 0.0 and worst <= maintenance_margin * gross_worst:
            for i in range(N):
                if qty[i] != 0.0:
                    p = low[t, i] if qty[i] > 0 else high[t, i]
                    if math.isnan(p):
                        p = px_open[i]
                    fee = abs(qty[i]) * p * liquidation_fee
                    cash += qty[i] * p - fee
                    fees[t] += fee
                    trades, n_trades = _fill(st, qty, i, t, -qty[i], p, fee, 0.0, eq_open, trades, n_trades,
                                             REASON_LIQUIDATION, p)
            liquidated_at = t
            for k in range(t, T):
                equity[k] = max(cash, 0.0)
            break

        # ---- 5. mark to market at the close, delisting exits
        eq_close = cash
        g = 0.0
        nt = 0.0
        for i in range(N):
            c = close[t, i]
            if not math.isnan(c):
                last_px[i] = c
            if qty[i] != 0.0:
                v = qty[i] * last_px[i]
                eq_close += v
                g += abs(v)
                nt += v
        for i in range(N):
            if qty[i] != 0.0 and force_exit[t, i]:
                c = last_px[i]
                fill = c * (1.0 - math.copysign(delist_penalty_bps, qty[i]) / 1e4)
                fee = abs(qty[i]) * fill * fee_rate[i]
                slip = abs(qty[i]) * c * delist_penalty_bps / 1e4
                v = qty[i] * c
                cash += qty[i] * fill - fee
                fees[t] += fee
                spread_slip[t] += slip
                traded_notional[t] += abs(v)
                eq_close -= v - (qty[i] * fill - fee)
                g -= abs(v)
                nt -= v
                trades, n_trades = _fill(st, qty, i, t, -qty[i], fill, fee, slip, eq_close, trades, n_trades,
                                         REASON_DELISTING, c)
                pending[i] = False
        equity[t] = eq_close
        if eq_close > 0.0:
            gross[t] = g / eq_close
            net[t] = nt / eq_close
        else:
            ruined_at = t
            for k in range(t, T):
                equity[k] = 0.0
            break

    # ---- open trades at the end: mark-to-market records
    for i in range(N):
        if st[i, S_ACTIVE] > 0.0 and qty[i] != 0.0:
            trades, n_trades = _close_trade(st, i, T - 1, trades, n_trades, REASON_END_OF_DATA,
                                            qty[i] * last_px[i], True)

    return (equity, gross, net, fees, spread_slip, funding_paid, traded_notional,
            trades[:n_trades].copy(), n_partial, liquidated_at, ruined_at, qty, cash, n_maker, n_taker, n_waits)
