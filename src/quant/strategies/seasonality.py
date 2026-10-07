"""Intraday (hour-of-day) seasonality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import interval_ms
from quant.strategies.base import Strategy


@dataclass(frozen=True)
class HourOfDaySeasonality(Strategy):
    """Trade the equal-weight basket of tradable symbols during the hours of day that had the best
    (long) / worst (short) trailing mean return.

    For each hour h, Z[h] = mean basket return over the last ``lookback_days`` observations of hour h
    (known at the close of the latest one). At the close of bar t the position for the next hour
    h' is long if Z[h'] ranks in the top ``top_hours`` and is positive, short if in the bottom and
    negative (``long_short``), else flat.
    """

    lookback_days: int = 90
    top_hours: int = 4
    long_short: bool = False
    family = "seasonality"
    provenance = "literature:crypto-intraday-seasonality"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if interval_ms(data.interval) != 3_600_000:
            raise ValueError("HourOfDaySeasonality needs 1h bars")
        tradable = data.tradable
        close = data.close.ffill()
        r = close.pct_change(fill_method=None).where(tradable.shift(1, fill_value=False))
        basket = r.mean(axis=1)
        hours = data.index.hour
        z = pd.DataFrame(np.nan, index=data.index, columns=range(24))
        min_obs = max(10, self.lookback_days // 2)
        for h in range(24):
            obs = basket[hours == h]
            z.loc[obs.index, h] = obs.rolling(self.lookback_days, min_periods=min_obs).mean().to_numpy()
        z = z.ffill()
        next_hour = (hours + 1) % 24
        ranks = z.rank(axis=1, ascending=False, method="first").to_numpy()
        zv = z.to_numpy()
        rows = np.arange(len(z))
        rank_next = ranks[rows, next_hour]
        z_next = zv[rows, next_hour]
        available = (~np.isnan(zv)).sum(axis=1) == 24
        signal = np.where(available & (rank_next <= self.top_hours) & (z_next > 0), 1.0, 0.0)
        if self.long_short:
            signal = np.where(available & (rank_next > 24 - self.top_hours) & (z_next < 0), -1.0, signal)
        n = tradable.sum(axis=1).replace(0, np.nan).to_numpy()
        per_symbol = np.where(np.isnan(n), 0.0, signal / np.where(np.isnan(n), 1.0, n))
        return tradable.astype(float).mul(per_symbol, axis=0)
