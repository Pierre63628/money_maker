"""Short-horizon mean reversion."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day
from quant.features import core as F
from quant.signals.latch import latch_positions
from quant.strategies.base import Strategy


@dataclass(frozen=True)
class FastRsiReversion(Strategy):
    """RSI(2)-style reversion with a time stop and optional trend filter.

    Idea source: Strategy Factory "Vanta v7 RSI(2) mean-rev SUI 1h"
    (resultId 01M2AC3J2MJ966Q8EC18EX9ZZD). Their Pine source is not public; this is a
    re-implementation from the described logic. Changes vs the source backtest:
    decision at bar close, execution at the next open (they filled at the signal bar's
    close), no pyramiding, fixed fraction per position, realistic costs + funding,
    multi-asset instead of a single hand-picked symbol.
    """

    rsi_window: int = 2
    lower: float = 10.0
    upper: float = 90.0
    exit_level: float = 50.0
    max_hold_bars: int = 12
    trend_filter_days: int = 0
    max_positions: int = 10
    long_only: bool = False
    family = "mean_reversion"
    provenance = "strategy_factory:01M2AC3J2MJ966Q8EC18EX9ZZD"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.close.ffill()
        r = F.rsi(close, self.rsi_window)
        entry_long, exit_long = r < self.lower, r > self.exit_level
        entry_short, exit_short = r > self.upper, r < self.exit_level
        if self.trend_filter_days > 0:
            sma = F.sma(close, int(self.trend_filter_days * bars_per_day(data.interval)))
            entry_long &= close > sma
            entry_short &= close < sma
        tradable = data.tradable
        entry_long &= tradable
        entry_short &= tradable
        if self.long_only:
            pos = latch_positions(entry_long, exit_long, max_hold=self.max_hold_bars)
        else:
            pos = latch_positions(entry_long, exit_long, entry_short, exit_short, max_hold=self.max_hold_bars)
        pos = pos.where(tradable, 0.0)
        return pos / self.max_positions
