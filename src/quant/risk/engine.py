"""Pre-trade risk engine.

Sits between Strategy and Portfolio/Broker. For every proposed target it computes
expected return, estimated risk, position size, maximum loss, portfolio impact,
correlation impact and liquidity risk, then approves, shrinks or blocks the order.

Principles:
- risk-reducing orders are always allowed (a limit must never trap a position);
- risk-increasing orders are clipped to the tightest limit, or blocked outright;
- breaching the drawdown limit trips a kill switch: everything is flattened and new
  risk stays blocked until a human resets it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class Action(str, Enum):
    APPROVE = "approve"
    REDUCE = "reduce"        # approved with a smaller target
    BLOCK = "block"
    FLATTEN = "flatten"      # kill switch forced exit


@dataclass(frozen=True)
class RiskLimits:
    max_gross_leverage: float = 1.0
    max_net_leverage: float = 1.0
    max_position_weight: float = 0.25
    max_daily_loss: float = 0.05             # fraction of start-of-day equity; blocks new risk for the day
    max_drawdown: float = 0.25               # from equity peak; trips the kill switch
    max_participation: float = 0.05          # order notional / average bar quote volume
    min_daily_quote_volume: float = 5e6
    max_cluster_weight: float = 0.5          # |net weight| among symbols correlated above the threshold (hedges net out)
    correlation_threshold: float = 0.8
    max_position_loss_at_gap: float = 0.03   # |w| * stress move must stay below this fraction of equity
    gap_stress_sigmas: float = 6.0           # stress move = sigmas * daily vol (fat-tail gap proxy)


@dataclass
class RiskState:
    equity_peak: float
    day_start_equity: float
    day: pd.Timestamp
    kill_switch: bool = False
    kill_reason: str = ""

    def update(self, equity: float, now: pd.Timestamp) -> None:
        today = now.floor("1D")
        if today != self.day:
            self.day = today
            self.day_start_equity = equity
        self.equity_peak = max(self.equity_peak, equity)

    def reset_kill_switch(self, operator: str) -> None:
        """Manual reset only (never automatic)."""
        self.kill_switch = False
        self.kill_reason = f"reset by {operator}"


@dataclass
class MarketSnapshot:
    prices: pd.Series                 # last price per symbol
    daily_vol: pd.Series              # daily return volatility per symbol (lagged estimate)
    avg_bar_quote_volume: pd.Series   # lagged average bar notional
    daily_quote_volume: pd.Series     # lagged daily notional
    correlation: pd.DataFrame         # lagged return correlation matrix
    expected_return: pd.Series | None = None  # per-symbol daily expected return from the strategy (optional)

    @classmethod
    def from_hourly_bars(cls, raw: dict[str, pd.DataFrame], symbols: list[str], prices: pd.Series) -> "MarketSnapshot":
        """Risk inputs from closed 1h bars: 30-day daily volatility, last-24h volume, 60-day correlation."""
        usable = [s for s in symbols if s in raw and len(raw[s])]
        if not usable:  # nothing to trade this step (e.g. a carry strategy waiting for funding to be rich)
            empty = pd.Series(dtype=float)
            return cls(prices=prices, daily_vol=empty, avg_bar_quote_volume=empty, daily_quote_volume=empty,
                       correlation=pd.DataFrame(dtype=float))
        closes = pd.DataFrame({s: raw[s]["close"] for s in usable})
        daily = closes.resample("1D").last()
        daily_ret = daily.pct_change(fill_method=None).iloc[-61:]
        qv = pd.DataFrame({s: raw[s]["quote_volume"] for s in usable})
        return cls(prices=prices, daily_vol=daily_ret.iloc[-30:].std(), avg_bar_quote_volume=qv.iloc[-24:].mean(),
                   daily_quote_volume=qv.iloc[-24:].sum(), correlation=daily_ret.corr(min_periods=20))


@dataclass
class RiskDecision:
    symbol: str
    action: Action
    requested_weight: float
    approved_weight: float
    reasons: list[str] = field(default_factory=list)
    expected_return: float = float("nan")      # daily, on approved position (fraction of equity)
    estimated_risk: float = float("nan")       # daily 1-sigma P&L on approved position (fraction of equity)
    position_notional: float = 0.0
    max_loss: float = float("nan")             # stress-gap loss on approved position (fraction of equity)
    portfolio_gross_after: float = float("nan")
    correlation_impact: float = float("nan")   # cluster |weight| sum after the order
    liquidity_risk: float = float("nan")       # order notional / average bar notional


class RiskEngine:
    def __init__(self, limits: RiskLimits, state: RiskState):
        self.limits = limits
        self.state = state

    def evaluate(self, equity: float, current: pd.Series, targets: pd.Series, market: MarketSnapshot,
                 now: pd.Timestamp) -> dict[str, RiskDecision]:
        """``current``/``targets``: weights (fraction of equity) per symbol. Returns one decision per symbol."""
        lim = self.limits
        self.state.update(equity, now)
        symbols = sorted(set(current.index) | set(targets.index))
        cur = current.reindex(symbols).fillna(0.0)
        tgt = targets.reindex(symbols).fillna(0.0)
        decisions: dict[str, RiskDecision] = {}

        drawdown = equity / self.state.equity_peak - 1.0
        if drawdown <= -lim.max_drawdown and not self.state.kill_switch:
            self.state.kill_switch = True
            self.state.kill_reason = f"drawdown {drawdown:.1%} breached limit {-lim.max_drawdown:.0%}"
        if self.state.kill_switch:
            for s in symbols:
                decisions[s] = RiskDecision(s, Action.FLATTEN if cur[s] != 0 else Action.BLOCK, float(tgt[s]), 0.0,
                                            [f"kill switch: {self.state.kill_reason}"])
            return decisions

        daily_loss = equity / self.state.day_start_equity - 1.0
        new_risk_blocked = daily_loss <= -lim.max_daily_loss
        approved = tgt.copy()
        reasons: dict[str, list[str]] = {s: [] for s in symbols}

        for s in symbols:
            increasing = abs(tgt[s]) > abs(cur[s]) + 1e-12 or np.sign(tgt[s]) * np.sign(cur[s]) < 0
            if not increasing:
                continue
            if new_risk_blocked:
                approved[s] = cur[s] if np.sign(tgt[s]) == np.sign(cur[s]) else 0.0
                reasons[s].append(f"daily loss {daily_loss:.1%} beyond {-lim.max_daily_loss:.0%}: new risk blocked")
                continue
            w = tgt[s]
            if abs(w) > lim.max_position_weight:
                w = math.copysign(lim.max_position_weight, w)
                reasons[s].append(f"position weight capped at {lim.max_position_weight:.0%}")
            dqv = market.daily_quote_volume.get(s, np.nan)
            if not np.isfinite(dqv) or dqv < lim.min_daily_quote_volume:
                w = cur[s] if abs(cur[s]) < abs(w) and np.sign(cur[s]) == np.sign(w) else 0.0
                reasons[s].append(f"daily volume {dqv:,.0f} below minimum {lim.min_daily_quote_volume:,.0f}")
            vol = market.daily_vol.get(s, np.nan)
            if np.isfinite(vol) and vol > 0 and w != 0:
                cap = lim.max_position_loss_at_gap / (lim.gap_stress_sigmas * vol)
                if abs(w) > cap:
                    w = math.copysign(cap, w)
                    reasons[s].append(f"gap-risk cap {cap:.1%} ({lim.gap_stress_sigmas:.0f}σ daily move)")
            elif w != 0 and not np.isfinite(vol):
                w = 0.0
                reasons[s].append("unknown volatility: cannot size risk")
            bar_notional = market.avg_bar_quote_volume.get(s, np.nan)
            order_notional = abs(w - cur[s]) * equity
            if np.isfinite(bar_notional) and bar_notional > 0 and order_notional > lim.max_participation * bar_notional:
                allowed = lim.max_participation * bar_notional / equity
                w = cur[s] + math.copysign(allowed, w - cur[s])
                reasons[s].append(f"order clipped to {lim.max_participation:.0%} of average bar volume")
            approved[s] = w

        self._apply_cluster_limit(approved, cur, market, reasons)
        self._apply_leverage_limits(approved, cur, reasons)

        gross_after = float(approved.abs().sum())
        clusters = self._clusters(symbols, market.correlation)
        for s in symbols:
            vol = market.daily_vol.get(s, np.nan)
            w = float(approved[s])
            er = market.expected_return.get(s, np.nan) if market.expected_return is not None else np.nan
            bar_notional = market.avg_bar_quote_volume.get(s, np.nan)
            order_notional = abs(w - cur[s]) * equity
            if abs(w - tgt[s]) < 1e-12:
                action = Action.APPROVE
            elif abs(w - cur[s]) < 1e-12 and abs(tgt[s] - cur[s]) > 1e-12:
                action = Action.BLOCK
            else:
                action = Action.REDUCE
            decisions[s] = RiskDecision(
                symbol=s, action=action, requested_weight=float(tgt[s]), approved_weight=w, reasons=reasons[s],
                expected_return=w * er if np.isfinite(er) else float("nan"),
                estimated_risk=abs(w) * vol if np.isfinite(vol) else float("nan"),
                position_notional=abs(w) * equity,
                max_loss=abs(w) * lim.gap_stress_sigmas * vol if np.isfinite(vol) else float("nan"),
                portfolio_gross_after=gross_after,
                correlation_impact=float(abs(sum(approved[o] for o in clusters.get(s, [s])))),
                liquidity_risk=order_notional / bar_notional if np.isfinite(bar_notional) and bar_notional > 0 else float("inf"),
            )
        return decisions

    # ----------------------------------------------------------------- helpers
    def _clusters(self, symbols: list[str], corr: pd.DataFrame) -> dict[str, list[str]]:
        out = {}
        for s in symbols:
            if s in corr.index:
                row = corr.loc[s].reindex(symbols)
                out[s] = [o for o in symbols if o == s or (np.isfinite(row.get(o, np.nan))
                                                          and row[o] >= self.limits.correlation_threshold)]
            else:
                out[s] = [s]
        return out

    def _apply_cluster_limit(self, approved: pd.Series, cur: pd.Series, market: MarketSnapshot,
                             reasons: dict[str, list[str]]) -> None:
        clusters = self._clusters(list(approved.index), market.correlation)
        for s, members in clusters.items():
            net = sum(approved[m] for m in members)
            if abs(net) <= self.limits.max_cluster_weight + 1e-12:
                continue
            # shrink only the orders that push the cluster's net exposure further in its direction
            increasing = [m for m in members if abs(approved[m]) > abs(cur[m]) + 1e-12
                          and math.copysign(1.0, approved[m]) == math.copysign(1.0, net)]
            if not increasing:
                continue
            excess = abs(net) - self.limits.max_cluster_weight
            growth = sum(abs(approved[m]) - abs(cur[m]) for m in increasing)
            shrink = min(1.0, excess / growth) if growth > 0 else 0.0
            for m in increasing:
                new_abs = abs(approved[m]) - shrink * (abs(approved[m]) - abs(cur[m]))
                approved[m] = math.copysign(new_abs, approved[m])
                reasons[m].append(f"correlated cluster net exposure capped at {self.limits.max_cluster_weight:.0%}")

    def _apply_leverage_limits(self, approved: pd.Series, cur: pd.Series, reasons: dict[str, list[str]]) -> None:
        lim = self.limits
        for kind, limit, measure in (("gross", lim.max_gross_leverage, lambda w: w.abs().sum()),
                                     ("net", lim.max_net_leverage, lambda w: abs(w.sum()))):
            value = measure(approved)
            if value <= limit + 1e-12:
                continue
            increasing = [s for s in approved.index if abs(approved[s]) > abs(cur[s]) + 1e-12]
            if not increasing:
                continue
            base = approved.copy()
            base[increasing] = cur[increasing].where(np.sign(cur[increasing]) == np.sign(approved[increasing]), 0.0)
            lo, hi = 0.0, 1.0  # fraction of the risk increase to keep
            for _ in range(40):
                mid = (lo + hi) / 2
                trial = base + mid * (approved - base)
                if measure(trial) <= limit + 1e-12:
                    lo = mid
                else:
                    hi = mid
            approved[:] = (base + lo * (approved - base)).to_numpy()
            for s in increasing:
                reasons[s].append(f"{kind} leverage capped at {limit:.2f}x")
