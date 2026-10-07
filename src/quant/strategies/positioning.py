"""Strategies using open interest and positioning ratios (Binance ``metrics``).

Timing: a metrics value stamped on bar ``t`` is the last 5-minute observation inside that bar, so it
is known at the same moment as the bar close, like any other signal input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day
from quant.signals.latch import latch_positions
from quant.strategies.base import Strategy
from quant.strategies.cross_sectional import _hours_to_bars, quantile_long_short
from quant.portfolio.sizing import hold_between_rebalances


@dataclass(frozen=True)
class CascadeReversalOI(Strategy):
    """Cascade reversal confirmed by an open-interest drop.

    A price crash with *falling* open interest is position unwinding (stops and liquidations); a crash
    with rising open interest is new shorts taking a view. Only the first kind is faded.
    """

    sigma_threshold: float = 4.0
    volume_multiple: float = 3.0
    max_taker_buy_share: float = 0.45
    min_oi_drop: float = 0.01
    hold_bars: int = 24
    vol_window_days: int = 30
    max_positions: int = 10
    family = "event_reversal"
    provenance = "internal:cascade + open-interest unwind confirmation"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if data.taker_buy_quote_volume is None:
            raise ValueError("CascadeReversalOI needs taker buy volume")
        oi = data.extra("open_interest")
        window = max(24, int(self.vol_window_days * bars_per_day(data.interval)))
        close = data.close.ffill()
        ret = np.log(close).diff()
        sigma = ret.rolling(window, min_periods=window // 2).std().shift(1)
        qv = data.quote_volume
        heavy = qv >= self.volume_multiple * qv.rolling(window, min_periods=window // 2).median().shift(1)
        taker_share = data.taker_buy_quote_volume / qv.where(qv > 0)
        oi_change = oi.ffill(limit=3).pct_change(fill_method=None)
        unwinding = oi_change <= -self.min_oi_drop
        tradable = data.tradable
        down = ((ret <= -self.sigma_threshold * sigma) & heavy & (taker_share <= self.max_taker_buy_share)
                & unwinding & tradable)
        never = pd.DataFrame(False, index=data.index, columns=data.symbols)
        pos = latch_positions(down, never, max_hold=self.hold_bars).where(tradable, 0.0)
        active = pos.abs().sum(axis=1)
        return pos.div(np.maximum(active, self.max_positions), axis=0)


@dataclass(frozen=True)
class OpenInterestTrend(Strategy):
    """Cross-sectional momentum split by whether open interest confirms the move.

    ``confirm``: rank momentum among perps whose open interest rose over the same window (new money
    following the move) and trade with the move. ``fade``: rank momentum among perps whose open
    interest fell (the move came from unwinding) and trade against it.
    """

    lookback_hours: int = 24
    rebalance_hours: int = 24
    quantile: float = 0.2
    min_symbols: int = 8
    mode: str = "confirm"
    gross: float = 1.0
    family = "positioning"
    provenance = "literature:open-interest confirmation of price moves"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        oi = data.extra("open_interest")
        look = _hours_to_bars(self.lookback_hours, data.interval)
        close = data.close.ffill()
        momentum = np.log(close).diff(look)
        oi_change = oi.ffill(limit=3).pct_change(look, fill_method=None)
        if self.mode == "confirm":
            eligible = oi_change > 0
            score = momentum
        elif self.mode == "fade":
            eligible = oi_change < 0
            score = -momentum
        else:
            raise ValueError("mode must be 'confirm' or 'fade'")
        tradable = data.tradable & eligible & momentum.notna() & oi_change.notna()
        weights = quantile_long_short(score, tradable, self.quantile, self.min_symbols, self.gross)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


@dataclass(frozen=True)
class CrowdingContrarian(Strategy):
    """Fade crowded retail positioning.

    Binance publishes the long/short *account* ratio of retail traders. When it is unusually high for
    a perp (z-score over ``z_window_days``), longs are crowded: short it, and buy the perps where
    accounts are unusually short.
    """

    z_window_days: int = 30
    rebalance_hours: int = 8
    quantile: float = 0.2
    min_symbols: int = 8
    ratio: str = "global_account_ls"
    gross: float = 1.0
    family = "positioning"
    provenance = "literature:retail positioning as a contrarian signal"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        series = data.extra(self.ratio).ffill(limit=3)
        window = max(24, int(self.z_window_days * bars_per_day(data.interval)))
        mean = series.rolling(window, min_periods=window // 2).mean()
        sd = series.rolling(window, min_periods=window // 2).std()
        z = (series - mean) / sd.where(sd > 0)
        tradable = data.tradable & z.notna()
        weights = quantile_long_short(-z, tradable, self.quantile, self.min_symbols, self.gross)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


@dataclass(frozen=True)
class CrowdingContrarianRobust(Strategy):
    """Crowding contrarian with the concentration controls the holdout showed were missing.

    Registered after the 2026 holdout: the period's profit came from three collapsing tokens, and
    removing them turned +38% into -10.6%. Three a priori controls are added, all of which lower the
    weight of the wildest names:
    - inverse-volatility weights inside each leg (a 200%-vol token gets a smaller ticket);
    - a hard cap per symbol;
    - a minimum listing age (recent listings are the most extreme movers).
    """

    z_window_days: int = 90
    rebalance_hours: int = 24
    quantile: float = 0.2
    min_symbols: int = 8
    ratio: str = "global_account_ls"
    gross: float = 1.0
    vol_window_days: int = 30
    max_weight: float = 0.10
    min_age_days: int = 0
    family = "positioning"
    provenance = "internal:crowding + concentration controls (post-holdout)"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        series = data.extra(self.ratio).ffill(limit=3)
        bpd = bars_per_day(data.interval)
        window = max(24, int(self.z_window_days * bpd))
        mean = series.rolling(window, min_periods=window // 2).mean()
        sd = series.rolling(window, min_periods=window // 2).std()
        z = (series - mean) / sd.where(sd > 0)
        tradable = data.tradable & z.notna()
        if self.min_age_days > 0:
            age = data.close.notna().cumsum()
            tradable &= age >= self.min_age_days * bpd
        legs = quantile_long_short(-z, tradable, self.quantile, self.min_symbols, self.gross)

        vol_bars = max(24, int(self.vol_window_days * bpd))
        vol = np.log(data.close.ffill()).diff().rolling(vol_bars, min_periods=vol_bars // 2).std().shift(1)
        inv_vol = (1.0 / vol.where(vol > 0)).replace([np.inf, -np.inf], np.nan)
        sides = {}
        for sign in (1.0, -1.0):
            leg = (legs * sign > 0)
            scores = inv_vol.where(leg).fillna(0.0)
            total = scores.sum(axis=1).replace(0, np.nan)
            sized = (scores.div(total, axis=0) * (self.gross / 2)).clip(upper=self.max_weight)
            # redistribute what the cap removed, then cap again (one pass is enough at these sizes)
            slack = (self.gross / 2) - sized.sum(axis=1)
            headroom = (self.max_weight - sized).where(leg & (sized > 0)).fillna(0.0)
            head_total = headroom.sum(axis=1).replace(0, np.nan)
            sized = (sized + headroom.div(head_total, axis=0).mul(slack, axis=0).fillna(0.0)).clip(upper=self.max_weight)
            sides[sign] = sized.fillna(0.0)
        # the cap can leave the two legs unequal (different name counts): rebalance to the smaller one
        target = pd.concat([sides[1.0].sum(axis=1), sides[-1.0].sum(axis=1)], axis=1).min(axis=1)
        weights = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        for sign, sized in sides.items():
            leg_total = sized.sum(axis=1).replace(0, np.nan)
            weights += sized.div(leg_total, axis=0).mul(target, axis=0).fillna(0.0) * sign
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)
