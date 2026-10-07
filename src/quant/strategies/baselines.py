"""Simple, well-known baselines. Anything more complex must beat these after costs.

Signals are computed on the forward-filled close (last known price, causal) so a
single missing bar does not flip positions. Symbols are flat before listing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.features import core as F
from quant.signals.latch import latch_positions
from quant.strategies.base import Strategy, equal_allocation


def _listed(data: MarketData) -> pd.DataFrame:
    return data.close.ffill().notna()


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    family = "benchmark"
    provenance = "textbook:buy-and-hold"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        return equal_allocation(_listed(data).astype(float))


@dataclass(frozen=True)
class SmaCross(Strategy):
    fast: int = 20
    slow: int = 100
    long_only: bool = False
    family = "trend"
    provenance = "textbook:moving-average-crossover"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        close = data.close.ffill()
        fast, slow = F.sma(close, self.fast), F.sma(close, self.slow)
        signal = pd.DataFrame(np.where(fast > slow, 1.0, 0.0 if self.long_only else -1.0),
                              index=close.index, columns=close.columns)
        signal = signal.where(slow.notna(), 0.0)
        return equal_allocation(signal)


@dataclass(frozen=True)
class TimeSeriesMomentum(Strategy):
    lookback: int = 168
    long_only: bool = False
    family = "momentum"
    provenance = "textbook:moskowitz-ooi-pedersen-2012"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.close.ffill()
        past = np.log(close).diff(self.lookback)
        signal = np.sign(past)
        if self.long_only:
            signal = signal.clip(lower=0.0)
        return equal_allocation(signal.fillna(0.0))


@dataclass(frozen=True)
class DonchianBreakout(Strategy):
    """Turtle-style channel breakout, decided on the bar close (no intrabar stop fills)."""

    entry_window: int = 55
    exit_window: int = 20
    long_only: bool = False
    family = "breakout"
    provenance = "textbook:turtle-donchian"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if self.exit_window > self.entry_window:
            raise ValueError("exit_window must be <= entry_window")
        close = data.close.ffill()
        high, low = data.high.ffill(), data.low.ffill()
        upper_in, lower_in = F.donchian_high(high, self.entry_window), F.donchian_low(low, self.entry_window)
        upper_out, lower_out = F.donchian_high(high, self.exit_window), F.donchian_low(low, self.exit_window)
        entry_long, exit_long = close > upper_in, close < lower_out
        if self.long_only:
            pos = latch_positions(entry_long, exit_long)
        else:
            pos = latch_positions(entry_long, exit_long, close < lower_in, close > upper_out)
        return equal_allocation(pos)


@dataclass(frozen=True)
class RsiReversion(Strategy):
    """Buy oversold / sell overbought, exit when RSI crosses back through ``exit_level``."""

    window: int = 14
    lower: float = 30.0
    upper: float = 70.0
    exit_level: float = 50.0
    long_only: bool = False
    family = "mean_reversion"
    provenance = "textbook:rsi-reversion"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        r = F.rsi(data.close.ffill(), self.window)
        entry_long, exit_long = r < self.lower, r > self.exit_level
        if self.long_only:
            pos = latch_positions(entry_long, exit_long)
        else:
            pos = latch_positions(entry_long, exit_long, r > self.upper, r < self.exit_level)
        return equal_allocation(pos)
