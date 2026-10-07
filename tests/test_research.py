from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant.backtesting.costs import CostModel
from quant.backtesting.engine import BacktestConfig
from quant.data.market_data import resample_bars
from quant.data.store import ParquetStore
from quant.data.universe import UniverseRule, classify_symbols, expand_to_bars, point_in_time_eligibility
from quant.evaluation.registry import ExperimentRegistry
from quant.evaluation.validation import WalkForwardSpec
from quant.research.experiments import Experiment
from quant.research.runner import run_experiment
from quant.research.universe_data import load_universe
from quant.strategies.cross_sectional import CrossSectionalMomentum
from quant.strategies.trend import TrendFollowing
from tests.conftest import gbm_frame


def test_classify_symbols_excludes_non_crypto():
    info = {"symbols": [
        {"symbol": "BTCUSDT", "baseAsset": "BTC", "underlyingType": "COIN", "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": 0},
        {"symbol": "AAPLUSDT", "baseAsset": "AAPL", "underlyingType": "EQUITY", "contractType": "TRADIFI_PERPETUAL", "status": "TRADING", "onboardDate": 0},
        {"symbol": "USDCUSDT", "baseAsset": "USDC", "underlyingType": "COIN", "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": 0},
        {"symbol": "BTCDOMUSDT", "baseAsset": "BTCDOM", "underlyingType": "INDEX", "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": 0},
    ]}
    df = classify_symbols(["BTCUSDT", "AAPLUSDT", "USDCUSDT", "BTCDOMUSDT", "LUNAUSDT", "BTCUSDT_240628", "ETHBUSD"], info)
    assert sorted(df.index[df["is_crypto"]]) == ["BTCUSDT", "LUNAUSDT"]
    assert df.loc["LUNAUSDT", "status"] == "DELISTED_UNKNOWN"


def test_eligibility_is_lagged_ranked_and_causal():
    days = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    qv = pd.DataFrame({"A": 100e6, "B": 50e6, "C": 10e6}, index=days)
    qv.loc[days[:20], "C"] = np.nan          # C listed on day 20
    qv.loc[days[40]:, "C"] = 500e6            # C becomes the most liquid on day 40
    rule = UniverseRule(min_history_days=5, liquidity_window_days=5, min_median_quote_volume=20e6, top_n=2)
    mask = point_in_time_eligibility(qv, rule)
    assert not mask.loc[days[:25], "C"].any()
    assert not mask.loc[days[40], "C"]         # yesterday's info does not include day 40 yet
    assert mask.loc[days[45], "C"] and mask.loc[days[45], "A"] and not mask.loc[days[45], "B"]
    part = point_in_time_eligibility(qv.iloc[:42], rule)
    pd.testing.assert_frame_equal(mask.iloc[:42], part)
    bars = pd.date_range("2024-01-10", periods=48, freq="h", tz="UTC")
    expanded = expand_to_bars(mask, bars)
    assert (expanded.loc[bars[:24]].to_numpy() == mask.loc[days[9]].to_numpy()).all()


def test_resample_requires_complete_bars():
    df = gbm_frame(48, 1)[["open", "high", "low", "close", "volume", "quote_volume"]]
    out = resample_bars(df.drop(df.index[5]), "1h", "4h")
    assert len(out) == 11  # the 4h bar missing one hour is dropped
    full = resample_bars(df, "1h", "4h")
    assert full["open"].iloc[1] == df["open"].iloc[4] and full["close"].iloc[1] == df["close"].iloc[7]
    assert full["high"].iloc[0] == df["high"].iloc[:4].max()


@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    store = ParquetStore(root)
    rng = np.random.default_rng(9)
    n = 24 * 700
    symbols = []
    for i in range(14):
        sym = f"C{i:02d}USDT"
        f = gbm_frame(n, 300 + i, start="2022-01-01", sigma=0.01, drift=rng.normal(0, 2e-5),
                      quote_volume=(40e6 if i < 12 else 1e5) / 24)
        if i == 13:
            f = f.iloc[: n // 2]  # delisted halfway
        store.write_klines(f, "um", sym, "1h")
        idx = f.index[f.index.hour % 8 == 0]
        store.write_funding(pd.DataFrame({"funding_rate": rng.normal(1e-4, 2e-4, len(idx)),
                                          "funding_interval_hours": 8}, index=idx), "um", sym)
        symbols.append(sym)
    info = {"symbols": [{"symbol": s, "baseAsset": s[:-4], "underlyingType": "COIN", "contractType": "PERPETUAL",
                         "status": "TRADING", "onboardDate": 0} for s in symbols]}
    classify_symbols(symbols, info).to_csv(root / "binance" / "um" / "symbol_classification.csv")
    return store


def test_load_universe_marks_illiquid_symbols_ineligible(synthetic_store):
    data = load_universe(synthetic_store, UniverseRule(top_n=10, min_median_quote_volume=5e6), "4h",
                         "2022-01-01", "2023-12-01")
    assert "C12USDT" not in data.symbols  # never liquid enough
    assert data.universe.sum(axis=1).max() <= 10
    assert data.interval == "4h" and data.funding is not None


def test_runner_end_to_end_and_holdout(synthetic_store, tmp_path):
    exp = Experiment(
        name="synthetic_trend",
        hypothesis="test",
        universe=UniverseRule(top_n=10, min_median_quote_volume=5e6, min_history_days=10),
        interval="4h",
        candidates=lambda: [TrendFollowing(lookback_days=7, band=0.5), TrendFollowing(lookback_days=21, band=0.0),
                            CrossSectionalMomentum(lookback_hours=24 * 5, min_symbols=5)],
        start="2022-01-01",
        holdout_start="2023-09-01",
        holdout_end="2023-11-30",
        walk_forward=WalkForwardSpec(train_days=180, test_days=60),
        config=BacktestConfig(costs=CostModel(spread_model="volume")),
    )
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    out = run_experiment(exp, synthetic_store, registry, tmp_path / "reports", bootstrap_samples=100, log=lambda m: None)
    assert out["verdict"] in {"PROMOTE_TO_PAPER", "REJECT"}
    assert (tmp_path / "reports" / "synthetic_trend" / "report.md").exists()
    assert registry.n_trials("synthetic_trend") == 4  # 3 candidates + walk-forward record
    assert out["folds"]["test_start"].min() == pd.Timestamp("2022-06-30", tz="UTC")
    assert out["oos"]["end"] <= pd.Timestamp("2023-09-01", tz="UTC")
    hold = run_experiment(exp, synthetic_store, registry, tmp_path / "reports", holdout=True, bootstrap_samples=100,
                          log=lambda m: None)
    assert hold["holdout"] is not None
    reg = registry.load()
    assert (reg["stage"] == "holdout").sum() == 1
    # the research-window walk-forward is unchanged by adding holdout data (causality)
    assert hold["oos"]["sharpe"] == pytest.approx(out["oos"]["sharpe"], rel=1e-6)
