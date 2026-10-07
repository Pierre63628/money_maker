from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quant.data.market_data import MarketData
from quant.evaluation.lookahead import check_no_lookahead
from quant.features import core as F
from quant.strategies.base import Strategy
from quant.strategies.baselines import BuyAndHold, DonchianBreakout, RsiReversion, SmaCross, TimeSeriesMomentum


@dataclass(frozen=True)
class PeeksTomorrow(Strategy):
    def target_weights(self, data: MarketData) -> pd.DataFrame:
        return np.sign(data.close.shift(-1) - data.close).fillna(0.0) / 2


@dataclass(frozen=True)
class FullSampleZScore(Strategy):
    def target_weights(self, data: MarketData) -> pd.DataFrame:
        c = data.close
        return ((c - c.mean()) / c.std()).clip(-1, 1).fillna(0.0) / 2


@dataclass(frozen=True)
class CentredWindow(Strategy):
    def target_weights(self, data: MarketData) -> pd.DataFrame:
        smooth = data.close.rolling(21, center=True).mean()
        return np.sign(data.close - smooth).fillna(0.0) / 2


@pytest.mark.parametrize("cheater", [PeeksTomorrow(), FullSampleZScore(), CentredWindow()])
def test_detector_catches_leaky_strategies(cheater, synthetic_market):
    assert not check_no_lookahead(cheater, synthetic_market).passed


@pytest.mark.parametrize("strategy", [
    BuyAndHold(), SmaCross(10, 50), TimeSeriesMomentum(48), DonchianBreakout(40, 15), RsiReversion(14),
    SmaCross(10, 50, long_only=True), DonchianBreakout(40, 15, long_only=True),
])
def test_baselines_are_causal(strategy, synthetic_market):
    report = check_no_lookahead(strategy, synthetic_market, n_cuts=10)
    assert report.passed, str(report)


FEATURES = {
    "sma": lambda d: F.sma(d.close, 20),
    "ema": lambda d: F.ema(d.close, 20),
    "zscore": lambda d: F.zscore(d.close, 50),
    "realized_vol": lambda d: F.realized_vol(d.close, 30),
    "atr": lambda d: F.atr(d.high, d.low, d.close, 14),
    "rsi": lambda d: F.rsi(d.close, 14),
    "donchian_high": lambda d: F.donchian_high(d.high, 20),
    "donchian_low": lambda d: F.donchian_low(d.low, 20),
}


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_features_are_causal(name, synthetic_market):
    fn = FEATURES[name]
    full = fn(synthetic_market)
    for cut in (100, 777, 2001):
        part = fn(synthetic_market.head(cut + 1))
        pd.testing.assert_frame_equal(full.iloc[: cut + 1], part, check_exact=False, rtol=1e-10)


def test_donchian_excludes_current_bar(synthetic_market):
    high = synthetic_market.high
    upper = F.donchian_high(high, 5)
    assert upper.iloc[10, 0] == pytest.approx(high.iloc[5:10, 0].max())


def test_rsi_bounds_and_reference_value():
    close = pd.DataFrame({"x": [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                                45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]})
    r = F.rsi(close, 14)
    assert r["x"].iloc[:14].isna().all()
    assert ((r["x"].dropna() >= 0) & (r["x"].dropna() <= 100)).all()
    assert r["x"].iloc[14] == pytest.approx(70.53, abs=0.5)  # Wilder's textbook example
