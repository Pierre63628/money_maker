"""Time-series trend following with turnover control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day
from quant.portfolio.sizing import annualised_vol, hold_between_rebalances, normalise_by_count, vol_scaled
from quant.signals.latch import latch_positions
from quant.strategies.base import Strategy


@dataclass(frozen=True)
class TrendFollowing(Strategy):
    """Volatility-normalised momentum with hysteresis, scheduled rebalancing and optional vol targeting.

    z = log-return over ``lookback_days`` / (bar vol * sqrt(lookback bars)).
    Go long when z > band, short when z < -band, and hold the position until the opposite
    threshold is crossed (hysteresis). ``band = 0`` reduces to the sign of past returns.
    Decisions are only taken every ``rebalance_hours``.
    """

    lookback_days: int = 30
    band: float = 0.5
    rebalance_hours: int = 24
    vol_target: float = 0.0
    vol_window_days: int = 30
    max_asset_weight: float = 1.0
    long_only: bool = False
    family = "trend"
    provenance = "textbook:moskowitz-ooi-pedersen-2012 (+hysteresis, scheduled rebalance)"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        bpd = bars_per_day(data.interval)
        lookback = max(1, int(round(self.lookback_days * bpd)))
        close = data.close.ffill()
        log_close = np.log(close)
        vol_bars = max(10, int(round(self.vol_window_days * bpd)))
        bar_vol = log_close.diff().rolling(vol_bars, min_periods=vol_bars // 2).std()
        z = log_close.diff(lookback) / (bar_vol * np.sqrt(lookback))
        up, down = z > self.band, z < -self.band
        if self.long_only:
            pos = latch_positions(up, down)
        else:
            pos = latch_positions(up, down, down, up)
        tradable = data.tradable
        pos = pos.where(tradable, 0.0)
        if self.vol_target > 0:
            raw = vol_scaled(pos, annualised_vol(close, vol_bars, data.interval), self.vol_target, self.max_asset_weight)
        else:
            raw = pos
        weights = normalise_by_count(raw, tradable)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)
