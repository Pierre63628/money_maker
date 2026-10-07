from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.risk.engine import Action, MarketSnapshot, RiskEngine, RiskLimits, RiskState

NOW = pd.Timestamp("2026-01-05 12:00", tz="UTC")


def snapshot(symbols, vol=0.02, dqv=1e9, bar=5e7, corr=None, er=None):
    idx = list(symbols)
    c = corr if corr is not None else pd.DataFrame(np.eye(len(idx)), index=idx, columns=idx)
    return MarketSnapshot(prices=pd.Series(100.0, index=idx), daily_vol=pd.Series(vol, index=idx),
                          avg_bar_quote_volume=pd.Series(bar, index=idx), daily_quote_volume=pd.Series(dqv, index=idx),
                          correlation=c, expected_return=er)


def engine(**limits):
    state = RiskState(equity_peak=100_000.0, day_start_equity=100_000.0, day=NOW.floor("1D"))
    return RiskEngine(RiskLimits(**limits), state)


def test_approves_within_limits_and_reports_trade_metrics():
    eng = engine()
    er = pd.Series({"A": 0.001})
    d = eng.evaluate(100_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.1}), snapshot(["A"], er=er), NOW)["A"]
    assert d.action is Action.APPROVE and d.approved_weight == 0.1
    assert d.estimated_risk == pytest.approx(0.1 * 0.02)
    assert d.max_loss == pytest.approx(0.1 * 6 * 0.02)
    assert d.expected_return == pytest.approx(0.0001)
    assert d.position_notional == pytest.approx(10_000)
    assert d.liquidity_risk == pytest.approx(10_000 / 5e7)


def test_position_and_gap_caps():
    eng = engine(max_position_weight=0.25)
    d = eng.evaluate(100_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.6}), snapshot(["A"], vol=0.01), NOW)["A"]
    assert d.action is Action.REDUCE and d.approved_weight == pytest.approx(0.25)
    d = eng.evaluate(100_000, pd.Series({"A": 0.0}), pd.Series({"A": -0.25}), snapshot(["A"], vol=0.05), NOW)["A"]
    assert d.approved_weight == pytest.approx(-0.03 / (6 * 0.05))


def test_illiquid_symbol_blocked_but_exit_allowed():
    eng = engine()
    snap = snapshot(["A"], dqv=1e6)
    assert eng.evaluate(100_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.1}), snap, NOW)["A"].action is Action.BLOCK
    exit_ = eng.evaluate(100_000, pd.Series({"A": 0.1}), pd.Series({"A": 0.0}), snap, NOW)["A"]
    assert exit_.action is Action.APPROVE and exit_.approved_weight == 0.0


def test_participation_clip():
    eng = engine()
    d = eng.evaluate(1_000_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.2}), snapshot(["A"], bar=1e6), NOW)["A"]
    assert d.approved_weight == pytest.approx(0.05 * 1e6 / 1_000_000)


def test_correlated_cluster_and_gross_leverage():
    syms = ["A", "B", "C", "D", "E"]
    corr = pd.DataFrame(np.eye(5), index=syms, columns=syms)
    corr.loc[["A", "B", "C"], ["A", "B", "C"]] = 0.9
    eng = engine(max_cluster_weight=0.5, max_gross_leverage=1.0)
    targets = pd.Series(0.25, index=syms)
    out = eng.evaluate(100_000, pd.Series(0.0, index=syms), targets, snapshot(syms, vol=0.01, corr=corr), NOW)
    cluster = sum(abs(out[s].approved_weight) for s in ["A", "B", "C"])
    assert cluster <= 0.5 + 1e-9
    assert sum(abs(d.approved_weight) for d in out.values()) <= 1.0 + 1e-9
    assert out["A"].correlation_impact <= 0.5 + 1e-9


def test_daily_loss_blocks_new_risk_only():
    eng = engine(max_daily_loss=0.05)
    snap = snapshot(["A", "B"], vol=0.01)
    out = eng.evaluate(94_000, pd.Series({"A": 0.2, "B": 0.0}), pd.Series({"A": 0.1, "B": 0.2}), snap, NOW)
    assert out["A"].approved_weight == pytest.approx(0.1) and out["A"].action is Action.APPROVE
    assert out["B"].approved_weight == 0.0 and out["B"].action is Action.BLOCK
    later = NOW + pd.Timedelta(days=1)  # new day: limit resets from the new start-of-day equity
    assert eng.evaluate(94_000, pd.Series({"B": 0.0}), pd.Series({"B": 0.2}), snap, later)["B"].approved_weight > 0


def test_kill_switch_flattens_and_requires_manual_reset():
    eng = engine(max_drawdown=0.25)
    snap = snapshot(["A"], vol=0.01)
    out = eng.evaluate(74_000, pd.Series({"A": 0.2}), pd.Series({"A": 0.2}), snap, NOW)["A"]
    assert out.action is Action.FLATTEN and out.approved_weight == 0.0
    recovered = eng.evaluate(99_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.2}), snap, NOW)["A"]
    assert recovered.action is Action.BLOCK
    eng.state.reset_kill_switch("operator")
    eng.state.equity_peak = 99_000
    assert eng.evaluate(99_000, pd.Series({"A": 0.0}), pd.Series({"A": 0.2}), snap, NOW)["A"].approved_weight > 0


def test_hedged_pair_is_not_treated_as_concentrated_risk():
    syms = ["AAA", "AAA@spot"]
    corr = pd.DataFrame(0.99, index=syms, columns=syms)
    eng = engine(max_cluster_weight=0.2, max_position_weight=0.5, max_gross_leverage=2.0)
    out = eng.evaluate(100_000, pd.Series(0.0, index=syms), pd.Series({"AAA": -0.3, "AAA@spot": 0.3}),
                       snapshot(syms, vol=0.01, corr=corr), NOW)
    assert out["AAA"].approved_weight == pytest.approx(-0.3) and out["AAA@spot"].approved_weight == pytest.approx(0.3)
    assert out["AAA"].correlation_impact == pytest.approx(0.0)
