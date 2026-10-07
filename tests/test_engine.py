from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtesting.costs import CostModel
from quant.backtesting.engine import BacktestConfig, run_backtest
from tests.conftest import frame_from_arrays, gbm_frame, market

ZERO = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0)


def ramp(n=8, **kw):
    open_ = 100.0 + np.arange(n)
    return frame_from_arrays(open_, close=open_ + 0.5, **kw)


def weights_for(data, values, symbol="AAA"):
    return pd.DataFrame({symbol: values}, index=data.index)


def test_zero_weights_keep_capital():
    data = market({"AAA": ramp()})
    res = run_backtest(weights_for(data, [0.0] * 8), data, ZERO)
    assert np.allclose(res.equity, 10_000.0)
    assert res.trades.empty


def test_same_bar_execution_is_rejected():
    with pytest.raises(ValueError, match="look-ahead"):
        BacktestConfig(execution_delay=0)


def test_decision_at_close_t_fills_at_open_t_plus_1():
    data = market({"AAA": ramp()})
    res = run_backtest(weights_for(data, [0, 0, 1, 1, 1, 1, 1, 1]), data, ZERO)
    qty = 10_000.0 / 103.0  # bar 2 decision -> bar 3 open = 103
    assert res.equity.iloc[2] == pytest.approx(10_000.0)
    assert res.equity.iloc[3] == pytest.approx(10_000.0 + qty * (103.5 - 103.0))
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 + qty * (107.5 - 103.0))
    trade = res.trades.iloc[0]
    assert trade["entry_time"] == data.index[3]
    assert trade["entry_price"] == pytest.approx(103.0)
    assert bool(trade["is_open"])


def test_execution_delay_two_bars():
    data = market({"AAA": ramp()})
    cfg = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0, execution_delay=2)
    res = run_backtest(weights_for(data, [0, 0, 1, 1, 1, 1, 1, 1]), data, cfg)
    assert res.trades.iloc[0]["entry_price"] == pytest.approx(104.0)


def test_future_prices_do_not_change_past_fills():
    base = ramp()
    shocked = base.copy()
    shocked.loc[shocked.index[5:], ["open", "high", "low", "close"]] *= 3.0
    w = [0, 1, 1, 1, 1, 1, 1, 1]
    r1 = run_backtest(weights_for(market({"AAA": base}), w), market({"AAA": base}), ZERO)
    r2 = run_backtest(weights_for(market({"AAA": shocked}), w), market({"AAA": shocked}), ZERO)
    assert np.allclose(r1.equity.iloc[:5], r2.equity.iloc[:5])


def test_round_trip_costs_are_exact():
    data = market({"AAA": ramp()})
    costs = CostModel(fee_bps=10.0, half_spread_bps=5.0, slippage_bps=0.0, impact_coef=0.0, funding=False)
    cfg = BacktestConfig(costs=costs, min_trade_notional=0.0)
    res = run_backtest(weights_for(data, [0, 0, 1, 1, 0, 0, 0, 0]), data, cfg)
    qty = 10_000.0 / 103.0
    buy = 103.0 * (1 + 5e-4)
    sell = 105.0 * (1 - 5e-4)
    expected = 10_000.0 - qty * buy - qty * buy * 1e-3 + qty * sell - qty * sell * 1e-3
    assert res.equity.iloc[-1] == pytest.approx(expected)
    trade = res.trades.iloc[0]
    assert trade["pnl"] == pytest.approx(expected - 10_000.0)
    assert trade["fees"] == pytest.approx(qty * (buy + sell) * 1e-3)
    assert trade["exit_reason"] == "signal"
    assert trade["bars_held"] == 3


def test_short_position_loses_when_price_rises():
    data = market({"AAA": ramp()})
    cfg = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0, max_gross_leverage=2.0)
    res = run_backtest(weights_for(data, [-1.0] * 8), data, cfg)
    qty = 10_000.0 / 101.0
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 - qty * (107.5 - 101.0))


def test_leverage_drift_above_cap_triggers_deleveraging():
    data = market({"AAA": ramp()})
    res = run_backtest(weights_for(data, [-1.0] * 8), data, ZERO)  # losing short: leverage creeps up
    assert (res.bars["traded_notional"] > 0).sum() >= 2
    assert res.bars["gross_exposure"].max() <= 1.0 * 1.10 + 0.02


def test_allow_short_false_rejects_negative_weights():
    data = market({"AAA": ramp()})
    with pytest.raises(ValueError, match="allow_short"):
        run_backtest(weights_for(data, [-1.0] * 8), data, BacktestConfig(allow_short=False))


def test_funding_paid_by_longs_received_by_shorts():
    frame = ramp()
    funding = pd.DataFrame({"funding_rate": [0.001]}, index=[frame.index[4]])
    for sign in (1.0, -1.0):
        data = market({"AAA": frame}, funding={"AAA": funding})
        cfg = BacktestConfig(costs=CostModel(fee_bps=0, half_spread_bps=0, slippage_bps=0, impact_coef=0, funding=True),
                             min_trade_notional=0.0, max_gross_leverage=2.0)
        res = run_backtest(weights_for(data, [sign] * 8), data, cfg)
        qty = sign * 10_000.0 / 101.0
        paid = qty * 104.0 * 0.001
        assert res.bars["funding"].sum() == pytest.approx(paid)
        assert res.equity.iloc[-1] == pytest.approx(10_000.0 + qty * (107.5 - 101.0) - paid)


def test_leverage_cap_scales_weights():
    data = market({"AAA": ramp()})
    cfg = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0, max_gross_leverage=2.0)
    res = run_backtest(weights_for(data, [3.0] * 8), data, cfg)
    qty = 2.0 * 10_000.0 / 101.0
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 + qty * (107.5 - 101.0))


def test_liquidation_on_adverse_wick():
    open_ = np.full(8, 100.0)
    low = np.full(8, 99.0)
    low[4] = 75.0  # -25% wick with 5x leverage wipes the account
    frame = frame_from_arrays(open_, close=open_, high=open_ + 1, low=low)
    data = market({"AAA": frame})
    cfg = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0, max_gross_leverage=5.0)
    res = run_backtest(weights_for(data, [5.0] * 8), data, cfg)
    assert res.liquidated_at == data.index[4]
    assert (res.equity.iloc[4:] == 0.0).all()
    assert res.trades.iloc[-1]["exit_reason"] == "liquidation"


def test_liquidity_cap_splits_large_orders():
    frame = ramp(n=12, quote_volume=20_000.0)  # 5% participation -> 1,000 per bar
    data = market({"AAA": frame})
    res = run_backtest(weights_for(data, [1.0] * 12), data, ZERO)
    assert res.n_partial_fills > 0
    filled = res.bars["traded_notional"]
    assert (filled <= 1_000.0 + 1e-6).all()
    assert filled.sum() > 5_000.0


def test_missing_bar_defers_execution():
    frame = ramp()
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = np.nan
    data = market({"AAA": frame})
    res = run_backtest(weights_for(data, [0, 1, 1, 1, 1, 1, 1, 1]), data, ZERO)
    assert res.trades.iloc[0]["entry_time"] == data.index[3]


def test_delisting_forces_exit():
    a = ramp(n=10)
    b = ramp(n=6)
    data = market({"AAA": a, "BBB": b})
    w = pd.DataFrame({"AAA": 0.0, "BBB": 0.5}, index=data.index)
    res = run_backtest(w, data, ZERO)
    trade = res.trades[res.trades["symbol"] == "BBB"].iloc[0]
    assert trade["exit_reason"] == "delisting"
    assert trade["exit_time"] == data.index[5]
    assert res.final_positions["BBB"] == 0.0


def test_flip_creates_two_trades():
    data = market({"AAA": ramp()})
    res = run_backtest(weights_for(data, [1, 1, 1, -1, -1, -1, -1, -1]), data, ZERO)
    assert list(res.trades["direction"]) == [1.0, -1.0]
    assert res.trades.iloc[0]["exit_time"] == res.trades.iloc[1]["entry_time"]


def test_trade_pnl_sums_to_equity_change_with_all_frictions():
    frames = {"AAA": gbm_frame(2500, 11), "BBB": gbm_frame(2500, 12, sigma=0.012)}
    idx = frames["AAA"].index
    rng = np.random.default_rng(3)
    funding = {s: pd.DataFrame({"funding_rate": rng.normal(1e-4, 3e-4, len(idx[::8]))}, index=idx[::8]) for s in frames}
    data = market(frames, funding=funding)
    raw = rng.choice([-1.0, 0.0, 0.5, 1.0], size=(len(idx), 2))
    hold = np.repeat(raw[::24], 24, axis=0)[: len(idx)]
    w = pd.DataFrame(hold, index=idx, columns=["AAA", "BBB"]) * 0.8
    res = run_backtest(w, data, BacktestConfig(max_gross_leverage=1.6))
    assert res.liquidated_at is None and res.ruined_at is None
    assert res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - 10_000.0, rel=1e-9, abs=1e-6)
    total_costs = res.bars[["fees", "spread_slippage"]].sum().sum()
    assert total_costs > 0


def test_buy_and_hold_does_not_rebalance():
    data = market({"AAA": gbm_frame(500, 5)})
    res = run_backtest(weights_for(data, [1.0] * 500), data, BacktestConfig())
    assert len(res.trades) == 1
    assert (res.bars["traded_notional"] > 0).sum() == 1


def test_long_data_gap_forces_exit_before_gap():
    frame = ramp(n=120)
    frame.loc[frame.index[20:100], ["open", "high", "low", "close"]] = np.nan  # 80-bar gap > 72
    data = market({"AAA": frame})
    res = run_backtest(weights_for(data, [1.0] * 120), data, ZERO)
    first = res.trades.iloc[0]
    assert first["exit_reason"] == "delisting"
    assert first["exit_time"] == data.index[19]


def test_short_data_gap_keeps_position():
    frame = ramp(n=60)
    frame.loc[frame.index[20:30], ["open", "high", "low", "close"]] = np.nan
    data = market({"AAA": frame})
    res = run_backtest(weights_for(data, [1.0] * 60), data, ZERO)
    assert len(res.trades) == 1 and bool(res.trades.iloc[0]["is_open"])


def test_volume_spread_model():
    costs = CostModel(spread_model="volume", spread_coef_bps=1.0, min_half_spread_bps=0.5, max_half_spread_bps=25.0,
                      per_symbol_half_spread_bps={"C": 3.0})
    dqv = np.array([[1e9, 20e6, 1e9], [np.nan, 1e3, 1e12]])
    m = costs.half_spread_matrix(["A", "B", "C"], dqv)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(np.sqrt(50))
    assert m[1, 0] == 25.0 and m[1, 1] == 25.0
    assert (m[:, 2] == 3.0).all()
    assert costs.scaled(2).half_spread_matrix(["A", "B", "C"], dqv)[0, 0] == pytest.approx(2.0)


def test_illiquid_symbol_pays_wider_spread():
    liquid = market({"AAA": ramp(quote_volume=1e9 / 24)})
    illiquid = market({"AAA": ramp(quote_volume=1e6 / 24)})
    cfg = BacktestConfig(costs=CostModel(fee_bps=0, slippage_bps=0, impact_coef=0, spread_model="volume", funding=False),
                         min_trade_notional=0.0, max_participation=1.0)
    w = [0, 1, 1, 1, 1, 1, 1, 1]
    s_liquid = run_backtest(weights_for(liquid, w), liquid, cfg).bars["spread_slippage"].sum()
    s_illiquid = run_backtest(weights_for(illiquid, w), illiquid, cfg).bars["spread_slippage"].sum()
    assert s_illiquid > 20 * s_liquid


# ------------------------------------------------------------------ maker execution
from quant.backtesting.engine import ExecutionModel  # noqa: E402

MAKER = BacktestConfig(costs=CostModel(fee_bps=5.0, half_spread_bps=1.0, slippage_bps=1.0, impact_coef=0.0,
                                       funding=False),
                       min_trade_notional=0.0, max_gross_leverage=2.0,
                       execution=ExecutionModel(mode="maker", maker_fee_bps=2.0, limit_offset_bps=1.0,
                                                trade_through_bps=2.0, maker_fallback_bars=3))


def flat_frame(n=10, lows=None, highs=None, opens=None):
    open_ = np.full(n, 100.0) if opens is None else np.asarray(opens, dtype=float)
    close = np.full(n, 100.0)
    low = np.minimum(open_, close) - 0.001 if lows is None else np.asarray(lows, dtype=float)
    high = np.maximum(open_, close) + 0.001 if highs is None else np.asarray(highs, dtype=float)
    return frame_from_arrays(open_, close, high, low)


def test_maker_buy_fills_only_when_price_trades_through_limit():
    lows = np.full(10, 99.999)
    lows[1] = 99.975   # touches the 99.99 limit but not by 2 bps: no fill
    lows[2] = 99.90    # trades through: fill at the limit
    data = market({"AAA": flat_frame(lows=lows)})
    res = run_backtest(weights_for(data, [1.0] * 10), data, MAKER)
    trade = res.trades.iloc[0]
    assert trade["entry_time"] == data.index[2]
    assert trade["entry_price"] == pytest.approx(99.99)
    qty = 10_000.0 / 100.0
    assert trade["fees"] == pytest.approx(qty * 99.99 * 2e-4)
    assert res.n_maker_fills == 1 and res.n_taker_fills == 0 and res.n_unfilled_order_bars == 1
    assert res.bars["spread_slippage"].sum() == 0.0


def test_marketable_limit_at_gap_fills_as_taker():
    opens = np.full(10, 100.0)
    opens[1] = 99.0  # gapped below the limit: a buy limit at 99.99 executes immediately at the market
    data = market({"AAA": flat_frame(opens=opens)})
    res = run_backtest(weights_for(data, [1.0] * 10), data, MAKER)
    assert res.n_taker_fills == 1 and res.trades.iloc[0]["entry_price"] == pytest.approx(99.0 * (1 + 2e-4))


def test_unfilled_entry_is_cancelled_when_signal_disappears():
    data = market({"AAA": flat_frame()})
    res = run_backtest(weights_for(data, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), data, MAKER)
    assert res.trades.empty and res.n_unfilled_order_bars == 3
    assert (res.bars["traded_notional"] == 0).all()


def test_unfilled_exit_falls_back_to_taker():
    lows = np.full(12, 99.999)
    lows[1] = 99.0  # entry fills on bar 1
    data = market({"AAA": flat_frame(n=12, lows=lows)})
    res = run_backtest(weights_for(data, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), data, MAKER)
    trade = res.trades.iloc[0]
    # exit decided at bar 3 close -> working from bar 4; never filled (highs never trade through) -> taker at bar 7
    assert trade["exit_time"] == data.index[7]
    assert trade["exit_price"] == pytest.approx(100.0 * (1 - 2e-4))
    assert res.n_maker_fills == 1 and res.n_taker_fills == 1


def test_maker_mode_accounting_invariant():
    frames = {"AAA": gbm_frame(2000, 31), "BBB": gbm_frame(2000, 32, sigma=0.012)}
    idx = frames["AAA"].index
    rng = np.random.default_rng(8)
    hold = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=(len(idx) // 12 + 1, 2)), 12, axis=0)[: len(idx)] * 0.6
    data = market(frames)
    res = run_backtest(pd.DataFrame(hold, index=idx, columns=["AAA", "BBB"]), data,
                       BacktestConfig(max_gross_leverage=1.2, execution=ExecutionModel(mode="maker")))
    assert res.n_maker_fills > 0 and res.n_taker_fills > 0
    assert res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - 10_000.0, rel=1e-9, abs=1e-6)
    stressed = res.config.stressed(2.0)
    assert stressed.execution.trade_through_bps == 4.0 and stressed.execution.maker_fee_bps == 4.0


def test_per_symbol_fees():
    data = market({"AAA": ramp(), "AAA@spot": ramp()})
    costs = CostModel(fee_bps=5.0, half_spread_bps=0.0, slippage_bps=0.0, impact_coef=0.0, funding=False,
                      per_symbol_fee_bps={"AAA@spot": 10.0})
    w = pd.DataFrame({"AAA": -0.5, "AAA@spot": 0.5}, index=data.index)
    res = run_backtest(w, data, BacktestConfig(costs=costs, min_trade_notional=0.0, max_gross_leverage=1.0))
    fees = res.trades.set_index("symbol")["fees"]
    assert fees["AAA@spot"] == pytest.approx(2 * fees["AAA"])
    assert res.equity.iloc[-1] == pytest.approx(10_000.0 - fees.sum())  # perfectly hedged legs


def test_cfd_financing_charges_both_legs():
    data = market({"AAA": ramp(), "BBB": ramp()})
    costs = CostModel(fee_bps=0, half_spread_bps=0, slippage_bps=0, impact_coef=0, funding=False,
                      financing_bps_per_day=24.0)  # 1 bp per hourly bar
    cfg = BacktestConfig(costs=costs, min_trade_notional=0.0, max_gross_leverage=1.0)
    w = pd.DataFrame({"AAA": 0.5, "BBB": -0.5}, index=data.index)
    res = run_backtest(w, data, cfg)
    charged = res.bars["funding"]
    assert (charged.iloc[2:] > 0).all()            # both legs pay, every bar
    gross_notional = res.bars["gross_exposure"] * res.bars["equity"]
    assert charged.iloc[5] == pytest.approx(gross_notional.iloc[5] * 1e-4, rel=0.02)
    assert res.equity.iloc[-1] < 10_000.0          # a hedged book still bleeds financing
