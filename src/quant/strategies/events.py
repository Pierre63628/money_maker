"""Event-driven strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day
from quant.signals.latch import latch_positions
from quant.strategies.base import Strategy


@dataclass(frozen=True)
class LiquidationCascadeReversal(Strategy):
    """Provide liquidity after forced selling (and, optionally, forced buying).

    A down-cascade bar: return <= -k * sigma, quote volume >= m * trailing median and aggressive
    sellers dominate (taker-buy share <= ``max_taker_buy_share``). Go long for ``hold_bars``.
    Up-cascades (short squeezes) mirror it when ``long_short``. Sigma and the volume median are
    computed from bars strictly before the event bar.
    """

    sigma_threshold: float = 4.0
    volume_multiple: float = 3.0
    max_taker_buy_share: float = 0.45
    hold_bars: int = 12
    vol_window_days: int = 30
    long_short: bool = False
    max_positions: int = 10
    family = "event_reversal"
    provenance = "literature:liquidity provision after forced liquidations"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if data.taker_buy_quote_volume is None:
            raise ValueError("LiquidationCascadeReversal needs taker buy volume (load with order flow)")
        window = max(24, int(self.vol_window_days * bars_per_day(data.interval)))
        close = data.close.ffill()
        ret = np.log(close).diff()
        sigma = ret.rolling(window, min_periods=window // 2).std().shift(1)
        qv = data.quote_volume
        vol_median = qv.rolling(window, min_periods=window // 2).median().shift(1)
        taker_share = data.taker_buy_quote_volume / qv.where(qv > 0)
        heavy = qv >= self.volume_multiple * vol_median
        tradable = data.tradable
        down = (ret <= -self.sigma_threshold * sigma) & heavy & (taker_share <= self.max_taker_buy_share) & tradable
        never = pd.DataFrame(False, index=data.index, columns=data.symbols)
        if self.long_short:
            up = (ret >= self.sigma_threshold * sigma) & heavy & (taker_share >= 1 - self.max_taker_buy_share) & tradable
            pos = latch_positions(down, never, up, never, max_hold=self.hold_bars)
        else:
            pos = latch_positions(down, never, max_hold=self.hold_bars)
        pos = pos.where(tradable, 0.0)
        active = pos.abs().sum(axis=1)
        return pos.div(np.maximum(active, self.max_positions), axis=0)


@dataclass(frozen=True)
class IdiosyncraticCascadeReversal(Strategy):
    """Cascade reversal restricted to idiosyncratic shocks.

    Registered after cascade_maker_top50_1h: the reversal premium was positive in 4 of 6 years but
    every variant lost in 2022 and 2025, with the worst trades on market-wide crashes (2022-05-11,
    2025-10-10). Hypothesis: forced selling isolated to one perp reverts; market-wide cascades
    carry information and keep going.

    Event on the BTC-beta residual: e = r - beta * r_BTC <= -k * sigma_e (sigma from prior bars),
    volume >= m * trailing median, aggressive sellers dominate, and at most ``max_breadth`` of the
    tradable universe has a raw return <= -``breadth_sigma`` sigma in the same bar.
    """

    sigma_threshold: float = 5.0
    volume_multiple: float = 3.0
    max_taker_buy_share: float = 0.45
    hold_bars: int = 24
    vol_window_days: int = 30
    max_breadth: float = 0.10
    breadth_sigma: float = 3.0
    max_positions: int = 10
    anchor: str = "BTCUSDT"
    family = "event_reversal"
    provenance = "internal:idiosyncratic-cascade-reversal (post batch 5)"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if data.taker_buy_quote_volume is None:
            raise ValueError("IdiosyncraticCascadeReversal needs taker buy volume (load with order flow)")
        zero = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        if self.anchor not in data.symbols:
            return zero
        window = max(24, int(self.vol_window_days * bars_per_day(data.interval)))
        close = data.close.ffill()
        ret = np.log(close).diff()
        btc = ret[self.anchor]
        beta = (ret.rolling(window, min_periods=window // 2).cov(btc)
                .div(btc.rolling(window, min_periods=window // 2).var(), axis=0)).shift(1)
        resid = ret - beta.mul(btc, axis=0)
        sigma_resid = resid.rolling(window, min_periods=window // 2).std().shift(1)
        sigma_raw = ret.rolling(window, min_periods=window // 2).std().shift(1)
        qv = data.quote_volume
        heavy = qv >= self.volume_multiple * qv.rolling(window, min_periods=window // 2).median().shift(1)
        taker_share = data.taker_buy_quote_volume / qv.where(qv > 0)
        tradable = data.tradable
        crashing = (ret <= -self.breadth_sigma * sigma_raw) & tradable
        breadth = crashing.sum(axis=1) / tradable.sum(axis=1).replace(0, np.nan)
        calm_market = breadth <= self.max_breadth
        down = ((resid <= -self.sigma_threshold * sigma_resid) & heavy & (taker_share <= self.max_taker_buy_share)
                & tradable)
        down = down.mul(calm_market, axis=0).astype(bool)
        down[self.anchor] = False
        never = pd.DataFrame(False, index=data.index, columns=data.symbols)
        pos = latch_positions(down, never, max_hold=self.hold_bars).where(tradable, 0.0)
        active = pos.abs().sum(axis=1)
        return pos.div(np.maximum(active, self.max_positions), axis=0)
