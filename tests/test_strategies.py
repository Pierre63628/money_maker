from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.market_data import MarketData
from quant.evaluation.lookahead import check_no_lookahead
from quant.portfolio.sizing import hold_between_rebalances, rebalance_bars
from quant.signals.latch import latch_positions
from quant.strategies.cross_sectional import (BtcLeadLag, CarryTrendFiltered, CrossSectionalMomentum, FundingCarry,
                                             quantile_long_short)
from quant.strategies.events import IdiosyncraticCascadeReversal, LiquidationCascadeReversal
from quant.strategies.positioning import (CascadeReversalOI, CrowdingContrarian, CrowdingContrarianRobust,
                                          OpenInterestTrend)
from quant.strategies.seasonality import HourOfDaySeasonality
from quant.strategies.mean_reversion import FastRsiReversion
from quant.strategies.trend import TrendFollowing
from tests.conftest import gbm_frame


@pytest.fixture(scope="module")
def universe_market_with_metrics(universe_market) -> MarketData:
    rng = np.random.default_rng(7)
    shape = universe_market.close.shape
    oi = pd.DataFrame(1e7 * np.exp(np.cumsum(rng.normal(0, 0.01, shape), axis=0)),
                      index=universe_market.index, columns=universe_market.symbols)
    ls = pd.DataFrame(1.5 + 0.3 * np.cumsum(rng.normal(0, 0.05, shape), axis=0) / 10,
                      index=universe_market.index, columns=universe_market.symbols)
    return universe_market.with_extras(open_interest=oi, global_account_ls=ls)


@pytest.fixture(scope="module")
def universe_market() -> MarketData:
    frames, funding = {}, {}
    rng = np.random.default_rng(42)
    for i in range(12):
        f = gbm_frame(2400, 100 + i, sigma=0.008 + 0.001 * i)
        if i >= 9:
            f = f.iloc[300 * (i - 8):]  # late listings
        sym = "BTCUSDT" if i == 0 else f"S{i:02d}USDT"
        frames[sym] = f
        idx = f.index[f.index.hour % 8 == 0]
        funding[sym] = pd.DataFrame({"funding_rate": rng.normal(1e-4 * (i - 6), 2e-4, len(idx))}, index=idx)
    data = MarketData.from_frames(frames, "1h", funding)
    mask = pd.DataFrame(rng.random(data.close.shape) > 0.1, index=data.index, columns=data.symbols)
    daily_mask = mask.groupby(mask.index.floor("1D")).transform("first")
    return data.with_universe(daily_mask)


STRATEGIES = [
    TrendFollowing(lookback_days=10, band=0.5, rebalance_hours=24),
    TrendFollowing(lookback_days=5, band=0.0, rebalance_hours=8, vol_target=0.5, long_only=True),
    CrossSectionalMomentum(lookback_hours=72, skip_hours=4, rebalance_hours=24, quantile=0.25, min_symbols=5),
    CrossSectionalMomentum(lookback_hours=6, rebalance_hours=4, reverse=True, min_symbols=5),
    FundingCarry(lookback_days=2, rebalance_hours=8, min_symbols=5),
    FastRsiReversion(max_hold_bars=6, trend_filter_days=5),
    CarryTrendFiltered(lookback_days=2, momentum_days=3, rebalance_hours=8, quantile=0.3, min_symbols=5),
    BtcLeadLag(lookback_hours=3, beta_window_days=10, rebalance_hours=2, min_symbols=5),
    HourOfDaySeasonality(lookback_days=20, top_hours=4, long_short=True),
    LiquidationCascadeReversal(sigma_threshold=2.0, volume_multiple=1.2, max_taker_buy_share=0.45, hold_bars=6,
                               vol_window_days=10, long_short=True, max_positions=4),
    IdiosyncraticCascadeReversal(sigma_threshold=1.5, volume_multiple=1.2, hold_bars=24, vol_window_days=10,
                                 max_breadth=0.5, max_positions=4),
]


OI_STRATEGIES = [
    CascadeReversalOI(sigma_threshold=1.5, volume_multiple=1.2, min_oi_drop=0.001, hold_bars=12, vol_window_days=10,
                      max_positions=4),
    OpenInterestTrend(lookback_hours=12, rebalance_hours=4, min_symbols=4, mode="confirm"),
    OpenInterestTrend(lookback_hours=12, rebalance_hours=4, min_symbols=4, mode="fade"),
    CrowdingContrarian(z_window_days=10, rebalance_hours=4, min_symbols=4),
    CrowdingContrarianRobust(z_window_days=10, rebalance_hours=4, min_symbols=4, max_weight=0.15, min_age_days=5,
                             vol_window_days=5),
]


@pytest.mark.parametrize("strategy", OI_STRATEGIES, ids=lambda s: s.name)
def test_positioning_strategies_are_causal_and_bounded(strategy, universe_market_with_metrics):
    data = universe_market_with_metrics
    report = check_no_lookahead(strategy, data, n_cuts=4, min_bars=900)
    assert report.passed, str(report)
    w = strategy.target_weights(data)
    assert np.isfinite(w.to_numpy()).all()
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9
    assert (w != 0).to_numpy().mean() > 0.02


def test_open_interest_modes_differ(universe_market_with_metrics):
    confirm = OpenInterestTrend(lookback_hours=12, mode="confirm").target_weights(universe_market_with_metrics)
    fade = OpenInterestTrend(lookback_hours=12, mode="fade").target_weights(universe_market_with_metrics)
    assert not confirm.equals(fade)
    from dataclasses import replace
    without = replace(universe_market_with_metrics, extras=None)
    with pytest.raises(ValueError, match="not loaded"):
        OpenInterestTrend().target_weights(without)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_strategies_are_causal(strategy, universe_market):
    report = check_no_lookahead(strategy, universe_market, n_cuts=5, min_bars=600)
    assert report.passed, str(report)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
def test_weights_respect_universe_and_leverage(strategy, universe_market):
    w = strategy.target_weights(universe_market)
    assert np.isfinite(w.to_numpy()).all()
    assert (w != 0).to_numpy().mean() > 0.05, "strategy never takes positions"
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9
    decided = w.index[rebalance_bars(w.index, getattr(strategy, "rebalance_hours", 1))]
    outside = (w.loc[decided] != 0) & ~universe_market.tradable.loc[decided]
    assert not outside.to_numpy().any()


def test_quantile_long_short_legs():
    score = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]], columns=list("abcdefghij"))
    tradable = pd.DataFrame(True, index=score.index, columns=score.columns)
    w = quantile_long_short(score, tradable, 0.2, 5, 1.0).iloc[0]
    assert w[["i", "j"]].tolist() == [0.25, 0.25]
    assert w[["a", "b"]].tolist() == [-0.25, -0.25]
    assert w.abs().sum() == pytest.approx(1.0)
    tradable.loc[:, ["c", "d", "e", "f", "g", "h"]] = False
    assert (quantile_long_short(score, tradable, 0.2, 5, 1.0) == 0).all().all()


def test_hysteresis_and_time_stop():
    z = pd.DataFrame({"x": [0.0, 0.6, 0.2, -0.2, -0.6, 0.1, 0.7]})
    pos = latch_positions(z > 0.5, z < -0.5, z < -0.5, z > 0.5)["x"].tolist()
    assert pos == [0, 1, 1, 1, -1, -1, 1]
    entry = pd.DataFrame({"x": [True, False, False, False, False, False]})
    never = pd.DataFrame({"x": [False] * 6})
    assert latch_positions(entry, never, max_hold=3)["x"].tolist() == [1, 1, 1, 0, 0, 0]


def test_hold_between_rebalances():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    w = pd.DataFrame({"x": np.arange(10.0)}, index=idx)
    held = hold_between_rebalances(w, 4)["x"].tolist()
    assert held == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8]


def test_ensemble_and_risk_parity_are_causal_and_bounded(universe_market):
    from quant.strategies.combine import ParameterEnsemble, RiskParitySleeves
    ens = ParameterEnsemble((TrendFollowing(lookback_days=5), TrendFollowing(lookback_days=10, band=0.0)), "t")
    w = ens.target_weights(universe_market)
    a = TrendFollowing(lookback_days=5).target_weights(universe_market)
    b = TrendFollowing(lookback_days=10, band=0.0).target_weights(universe_market)
    pd.testing.assert_frame_equal(w, (a + b) / 2)
    rp = RiskParitySleeves((ens, CrossSectionalMomentum(lookback_hours=72, min_symbols=5)), vol_lookback_days=20,
                           target_vol=0.5, max_gross=1.2)
    assert check_no_lookahead(rp, universe_market, n_cuts=3, min_bars=900).passed
    assert rp.target_weights(universe_market).abs().sum(axis=1).max() <= 1.2 + 1e-9


def test_cash_and_carry_is_delta_neutral_and_causal():
    from quant.strategies.cross_sectional import CashAndCarry
    rng = np.random.default_rng(5)
    frames, funding = {}, {}
    for i in range(4):
        base = gbm_frame(2400, 900 + i, sigma=0.01)
        spot = base.copy()
        spot[["open", "high", "low", "close"]] *= 1 + rng.normal(0, 2e-4, (len(base), 1))
        frames[f"C{i}USDT"] = base
        frames[f"C{i}USDT@spot"] = spot
        idx = base.index[base.index.hour % 8 == 0]
        level = 3e-4 if i < 2 else 2e-5
        funding[f"C{i}USDT"] = pd.DataFrame({"funding_rate": level + rng.normal(0, 5e-5, len(idx))}, index=idx)
    data = MarketData.from_frames(frames, "1h", funding)
    strat = CashAndCarry(entry_apr=0.15, exit_apr=0.05, lookback_days=3, rebalance_hours=8, max_pairs=2,
                         gross_per_pair=0.5)
    w = strat.target_weights(data)
    assert check_no_lookahead(strat, data, n_cuts=4, min_bars=600).passed
    assert (w.sum(axis=1).abs() < 1e-12).all()                  # delta neutral
    assert (w[["C0USDT", "C1USDT"]] < 0).any().all()            # rich funding: short perp
    assert (w[["C2USDT", "C3USDT"]] == 0).all().all()           # poor funding: no trade
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9


def test_robust_crowding_caps_and_scales_positions(universe_market_with_metrics):
    data = universe_market_with_metrics
    strat = CrowdingContrarianRobust(z_window_days=10, rebalance_hours=4, min_symbols=4, quantile=0.3,
                                     max_weight=0.12, vol_window_days=5)
    w = strat.target_weights(data)
    active = w[w != 0].stack()
    assert active.abs().max() <= 0.12 + 1e-9                      # per-symbol cap respected
    gross = w.abs().sum(axis=1)
    assert gross[gross > 0].max() <= 1.0 + 1e-9
    assert abs(w.sum(axis=1)).max() < 1e-9                        # still market neutral
    young = CrowdingContrarianRobust(z_window_days=10, rebalance_hours=4, min_symbols=4, min_age_days=60,
                                     vol_window_days=5).target_weights(data)
    assert (young != 0).to_numpy().sum() < (w != 0).to_numpy().sum()   # the age filter removes names
