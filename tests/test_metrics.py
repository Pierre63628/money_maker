from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant.backtesting.engine import BacktestConfig
from quant.backtesting.result import BacktestResult
from quant.data.timeframes import interval_to_timedelta
from quant.evaluation.metrics import compute_metrics, max_drawdown, trade_statistics


def result_from_equity(values, interval="1d", trades=None) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=len(values), freq=interval_to_timedelta(interval), tz="UTC").as_unit("ns")
    bars = pd.DataFrame({"equity": values, "gross_exposure": 1.0, "net_exposure": 1.0, "fees": 0.0,
                         "spread_slippage": 0.0, "funding": 0.0, "traded_notional": 0.0}, index=idx)
    if trades is None:
        trades = pd.DataFrame(columns=["pnl", "return_on_equity", "is_open", "exit_reason", "bars_held"])
    return BacktestResult("t", interval, BacktestConfig(initial_capital=100.0), bars, trades,
                          pd.DataFrame(index=idx), 0, None, None, pd.Series(dtype=float))


def test_cagr_and_total_return_over_one_year():
    values = np.linspace(101.0, 150.0, 365)
    m = compute_metrics(result_from_equity(values))
    assert m["years"] == pytest.approx(1.0)
    assert m["total_return"] == pytest.approx(0.5)
    assert m["cagr"] == pytest.approx(0.5)


def test_sharpe_sortino_volatility_match_manual_computation():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, 400)
    values = 100.0 * np.cumprod(1 + rets)
    m = compute_metrics(result_from_equity(values))
    assert m["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * math.sqrt(365), rel=1e-9)
    downside = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert m["sortino"] == pytest.approx(rets.mean() / downside * math.sqrt(365), rel=1e-9)
    assert m["volatility"] == pytest.approx(rets.std(ddof=1) * math.sqrt(365), rel=1e-9)


def test_daily_resampling_of_hourly_equity():
    rng = np.random.default_rng(1)
    daily = rng.normal(0.0, 0.01, 30)
    hourly_values = np.repeat(100.0 * np.cumprod(1 + daily), 24)  # flat within each day
    m = compute_metrics(result_from_equity(hourly_values, interval="1h"))
    assert m["sharpe"] == pytest.approx(daily.mean() / daily.std(ddof=1) * math.sqrt(365), rel=1e-9)


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 104], index=pd.date_range("2024", periods=5, freq="D", tz="UTC"))
    dd, peak, trough = max_drawdown(eq)
    assert dd == pytest.approx(90 / 120 - 1)
    assert peak == eq.index[1] and trough == eq.index[2]


def test_trade_statistics():
    trades = pd.DataFrame({
        "pnl": [10.0, -5.0, 20.0, -5.0, 99.0],
        "return_on_equity": [0.1, -0.05, 0.2, -0.05, 0.99],
        "is_open": [False, False, False, False, True],
        "exit_reason": ["signal"] * 4 + ["end_of_data"],
        "bars_held": [2, 4, 6, 8, 1],
    })
    s = trade_statistics(trades)
    assert s["n_trades"] == 4 and s["n_open_trades"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["profit_factor"] == pytest.approx(3.0)
    assert s["expectancy"] == pytest.approx(5.0)
    assert s["worst_trade_pct_equity"] == pytest.approx(-0.05)
    assert s["avg_bars_held"] == pytest.approx(5.0)
