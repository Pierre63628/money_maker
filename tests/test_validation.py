from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quant.backtesting.costs import CostModel
from quant.backtesting.engine import BacktestConfig, run_backtest
from quant.data.market_data import MarketData
from quant.evaluation.metrics import period_returns
from quant.evaluation.registry import ExperimentRegistry
from quant.evaluation.stats import (bootstrap_performance, deflated_sharpe_ratio, expected_max_sharpe, pbo_cscv,
                                    probabilistic_sharpe_ratio)
from quant.evaluation.validation import (WalkForwardSpec, calendar_returns, parameter_sensitivity,
                                         performance_by_regime, regime_labels, sensitivity_summary, slice_result,
                                         walk_forward)
from quant.strategies.base import Strategy
from quant.strategies.baselines import SmaCross
from tests.conftest import frame_from_arrays, gbm_frame, market


# ------------------------------------------------------------------ statistics
def test_psr_distinguishes_skill_from_noise():
    rng = np.random.default_rng(0)
    assert probabilistic_sharpe_ratio(rng.normal(0.002, 0.01, 1000)) > 0.99
    assert 0.02 < probabilistic_sharpe_ratio(rng.normal(0.0, 0.01, 1000)) < 0.98


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(1)
    n_trials, n_obs = 50, 500
    maxima = []
    for _ in range(200):
        r = rng.normal(0, 0.01, (n_obs, n_trials))
        maxima.append((r.mean(0) / r.std(0, ddof=1)).max())
    theory = expected_max_sharpe(n_trials, 1.0 / n_obs)
    assert theory == pytest.approx(np.mean(maxima), rel=0.15)
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01)


def test_deflated_sharpe_penalises_selection_from_many_trials():
    rng = np.random.default_rng(2)
    trials = rng.normal(0, 0.01, (1000, 200))
    sharpes = trials.mean(0) / trials.std(0, ddof=1)
    best = trials[:, np.argmax(sharpes)]
    assert probabilistic_sharpe_ratio(best) > 0.9          # looks significant in isolation...
    assert deflated_sharpe_ratio(best, 200, sharpes)["dsr"] < 0.6  # ...but not after deflation
    skilled = rng.normal(0.003, 0.01, 1000)
    assert deflated_sharpe_ratio(skilled, 200, sharpes)["dsr"] > 0.95


def test_bootstrap_probabilities():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=1500, freq="D", tz="UTC")
    good = bootstrap_performance(pd.Series(rng.normal(0.002, 0.02, 1500), index=idx), n_samples=500)
    noise = bootstrap_performance(pd.Series(rng.normal(0.0, 0.02, 1500), index=idx), n_samples=500)
    assert good["prob_sharpe_le_0"] < 0.01
    assert 0.1 < noise["prob_sharpe_le_0"] < 0.9
    lo, mid, hi = good["sharpe_q05_q50_q95"]
    assert lo < mid < hi


def test_pbo_noise_vs_persistent_skill():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=1600, freq="D", tz="UTC")
    noise = pd.DataFrame(rng.normal(0, 0.01, (1600, 30)), index=idx)
    assert 0.3 < pbo_cscv(noise, n_blocks=10)["pbo"] < 0.7
    skilled = noise.copy()
    skilled[0] = skilled[0] + 0.003
    assert pbo_cscv(skilled, n_blocks=10)["pbo"] < 0.05


# --------------------------------------------------------------- walk forward
@dataclass(frozen=True)
class Constant(Strategy):
    weight: float = 1.0

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        return pd.DataFrame(self.weight, index=data.index, columns=data.symbols)


def regime_market(days_up=300, days_down=300, drift=0.004) -> MarketData:
    n = (days_up + days_down) * 24
    rng = np.random.default_rng(5)
    step = np.where(np.arange(n) < days_up * 24, drift / 24, -drift / 24) + 0.002 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(step))
    open_ = np.concatenate([[100.0], close[:-1]])
    return market({"AAA": frame_from_arrays(open_, close, np.maximum(open_, close) * 1.001,
                                            np.minimum(open_, close) * 0.999, start="2022-01-01")})


CFG = BacktestConfig(costs=CostModel.zero(), min_trade_notional=0.0, max_gross_leverage=2.0)


def test_walk_forward_follows_regimes_and_pays_switching():
    data = regime_market()
    wf = walk_forward([Constant(1.0), Constant(-1.0)], data, CFG, WalkForwardSpec(train_days=60, test_days=30))
    folds = wf.folds.set_index("test_start")
    up = folds[folds.index < pd.Timestamp("2022-10-28", tz="UTC")]
    down = folds[folds.index > pd.Timestamp("2023-01-15", tz="UTC")]
    assert (up["chosen"] == "Constant(weight=1.0)").all()
    assert (down["chosen"] == "Constant(weight=-1.0)").all()
    assert wf.oos_start == pd.Timestamp("2022-03-02", tz="UTC")
    assert (wf.oos.bars.loc[wf.oos.bars.index < wf.oos_start, "traded_notional"] == 0).all()
    assert wf.oos_only().metrics()["total_return"] > 0


def test_walk_forward_choices_ignore_future_data():
    data = regime_market()
    spec = WalkForwardSpec(train_days=60, test_days=30)
    base = walk_forward([Constant(1.0), Constant(-1.0)], data, CFG, spec).folds
    cutoff = base["test_start"].iloc[5]
    close = data.close.copy()
    future = close.index >= cutoff
    close.loc[future] = close.loc[future] * np.exp(-np.linspace(0, 3, future.sum()))[:, None]
    shocked = MarketData(interval="1h", open=close.shift(1).fillna(100.0), high=close * 1.001, low=close * 0.999,
                         close=close, volume=data.volume, quote_volume=data.quote_volume)
    moved = walk_forward([Constant(1.0), Constant(-1.0)], shocked, CFG, spec).folds
    assert base["chosen"].iloc[:6].tolist() == moved["chosen"].iloc[:6].tolist()


def test_slice_result_rebases_initial_capital():
    data = regime_market()
    res = run_backtest(Constant(1.0).target_weights(data), data, CFG)
    cut = data.index[24 * 100]
    sub = slice_result(res, start=cut)
    assert sub.config.initial_capital == pytest.approx(res.equity.loc[:cut].iloc[-2])
    assert sub.metrics()["total_return"] == pytest.approx(res.equity.iloc[-1] / sub.config.initial_capital - 1)


# -------------------------------------------------------- sensitivity & regimes
def test_parameter_sensitivity_table():
    data = market({"AAA": gbm_frame(3000, 21)})
    table = parameter_sensitivity(SmaCross, SmaCross(fast=20, slow=100), data, BacktestConfig(),
                                  rel_steps=(-0.25, 0.25))
    assert set(table["param"]) == {"base", "fast", "slow"}
    summary = sensitivity_summary(table)
    assert summary["n_neighbours"] == 4


def test_regime_labels_are_causal():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2020-01-01", periods=900 * 24, freq="h", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(len(idx)))), index=idx)
    full = regime_labels(close)
    part = regime_labels(close.iloc[: 600 * 24])
    pd.testing.assert_frame_equal(full.iloc[: len(part)], part)
    assert set(full["trend"].dropna()) == {"bull", "bear"}
    daily = pd.Series(rng.normal(0, 0.01, 899), index=pd.date_range("2020-01-02", periods=899, freq="D", tz="UTC"))
    table = performance_by_regime(daily, full)
    assert {"bull", "bear", "high_vol", "low_vol"} <= set(table.index)
    assert table.loc[["bull", "bear"], "share_days"].sum() == pytest.approx(1.0)


def test_calendar_returns_compound_to_total():
    data = regime_market()
    res = run_backtest(Constant(1.0).target_weights(data), data, CFG)
    yearly = calendar_returns(res)
    assert (1 + yearly).prod() - 1 == pytest.approx(res.metrics()["total_return"], rel=1e-9)


def test_registry_counts_trials(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    for i in range(3):
        reg.log("exp", f"s{i}", "trend", "internal", {"a": i}, {"interval": "1h"},
                {"sharpe": np.float64(0.5), "start": pd.Timestamp("2024-01-01", tz="UTC"), "nan": float("nan")})
    reg.log("other", "s", "mean_reversion", "internal", {}, {}, {})
    assert reg.n_trials() == 4 and reg.n_trials("exp") == 3 and reg.n_trials(family="trend") == 3
    assert period_returns is not None
