"""Cross-sectional (relative value) strategies over a point-in-time universe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import interval_ms
from quant.portfolio.sizing import hold_between_rebalances
from quant.strategies.base import Strategy


def _hours_to_bars(hours: float, interval: str) -> int:
    return max(1, int(round(hours * 3_600_000 / interval_ms(interval))))


def quantile_long_short(score: pd.DataFrame, tradable: pd.DataFrame, quantile: float, min_symbols: int,
                        gross: float, long_short: bool = True) -> pd.DataFrame:
    """Long the top ``quantile`` of ``score``, short the bottom one, equal weight within each leg."""
    s = score.where(tradable)
    count = s.notna().sum(axis=1)
    k = np.maximum(1, np.floor(count * quantile)).to_numpy()[:, None]
    ordinal = s.rank(axis=1, ascending=False, method="first").to_numpy()
    top = ordinal <= k
    bottom = (count.to_numpy()[:, None] - ordinal) < k
    valid = (count >= min_symbols).to_numpy()[:, None] & ~np.isnan(ordinal)
    top &= valid
    bottom &= valid & ~top
    n_top = top.sum(axis=1, keepdims=True)
    n_bottom = bottom.sum(axis=1, keepdims=True)
    long_budget = gross / 2 if long_short else gross
    w = np.where(top, long_budget / np.maximum(n_top, 1), 0.0)
    if long_short:
        w = w - np.where(bottom, (gross / 2) / np.maximum(n_bottom, 1), 0.0)
    return pd.DataFrame(w, index=score.index, columns=score.columns)


@dataclass(frozen=True)
class CrossSectionalMomentum(Strategy):
    """Rank symbols by past return over ``lookback_hours`` (skipping the last ``skip_hours``).

    ``reverse=True`` turns it into short-term reversal (buy losers, sell winners).
    """

    lookback_hours: int = 24 * 14
    skip_hours: int = 0
    rebalance_hours: int = 24
    quantile: float = 0.2
    min_symbols: int = 10
    gross: float = 1.0
    reverse: bool = False
    long_short: bool = True
    family = "cross_sectional"
    provenance = "textbook:jegadeesh-titman-1993 / crypto xs-momentum literature"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        close = data.close.ffill()
        skip = _hours_to_bars(self.skip_hours, data.interval) if self.skip_hours else 0
        look = _hours_to_bars(self.lookback_hours, data.interval)
        log_close = np.log(close)
        score = log_close.shift(skip) - log_close.shift(skip + look)
        if self.reverse:
            score = -score
        # a symbol must have a full lookback of history to be ranked
        history_ok = close.shift(skip + look).notna()
        weights = quantile_long_short(score, data.tradable & history_ok, self.quantile, self.min_symbols,
                                      self.gross, self.long_short)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


@dataclass(frozen=True)
class FundingCarry(Strategy):
    """Short the perps paying the highest recent funding, long the lowest (collect funding both legs).

    Uses only settled funding rates (known at settlement), averaged over ``lookback_days``.
    """

    lookback_days: int = 3
    rebalance_hours: int = 8
    quantile: float = 0.2
    min_symbols: int = 10
    gross: float = 1.0
    family = "carry"
    provenance = "textbook:perpetual-funding-carry"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if data.funding is None:
            raise ValueError("FundingCarry needs funding rates")
        bars = _hours_to_bars(24 * self.lookback_days, data.interval)
        settled = data.funding
        total = settled.fillna(0.0).rolling(bars, min_periods=1).sum()
        count = settled.notna().rolling(bars, min_periods=1).sum()
        avg = (total / count.where(count > 0)).where(count >= max(1, self.lookback_days * 2))
        weights = quantile_long_short(-avg, data.tradable, self.quantile, self.min_symbols, self.gross)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


def _leg_weights(mask: np.ndarray, budget: float) -> np.ndarray:
    n = mask.sum(axis=1, keepdims=True)
    return np.where(mask, budget / np.maximum(n, 1), 0.0)


@dataclass(frozen=True)
class CarryTrendFiltered(Strategy):
    """Funding carry that only shorts high-funding perps already trending down and only buys
    low-funding perps trending up (avoid shorting into squeezes / catching falling knives).

    Registered after carry_top50_4h showed -80% drawdowns driven by short squeezes.
    """

    lookback_days: int = 7
    momentum_days: int = 7
    rebalance_hours: int = 24
    quantile: float = 0.2
    min_symbols: int = 10
    gross: float = 1.0
    family = "carry"
    provenance = "internal:carry+momentum-filter"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        if data.funding is None:
            raise ValueError("CarryTrendFiltered needs funding rates")
        bars = _hours_to_bars(24 * self.lookback_days, data.interval)
        settled = data.funding
        total = settled.fillna(0.0).rolling(bars, min_periods=1).sum()
        count = settled.notna().rolling(bars, min_periods=1).sum()
        avg = (total / count.where(count > 0)).where(count >= max(1, self.lookback_days * 2))
        close = data.close.ffill()
        mom = np.log(close).diff(_hours_to_bars(24 * self.momentum_days, data.interval))
        s = avg.where(data.tradable & mom.notna())
        cnt = s.notna().sum(axis=1).to_numpy()[:, None]
        k = np.maximum(1, np.floor(cnt * self.quantile))
        ordinal = s.rank(axis=1, ascending=False, method="first").to_numpy()
        valid = (cnt >= self.min_symbols) & ~np.isnan(ordinal)
        high_funding = valid & (ordinal <= k)
        low_funding = valid & ((cnt - ordinal) < k) & ~high_funding
        m = mom.to_numpy()
        shorts = high_funding & (m < 0)
        longs = low_funding & (m > 0)
        w = _leg_weights(longs, self.gross / 2) - _leg_weights(shorts, self.gross / 2)
        weights = pd.DataFrame(w, index=data.index, columns=data.symbols)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


@dataclass(frozen=True)
class BtcLeadLag(Strategy):
    """Cross-sectional catch-up: alts whose recent return lags beta * BTC return are bought, leaders sold.

    residual_i = beta_i * r_BTC(L) - r_i(L), beta from trailing 1-bar return covariance.
    """

    lookback_hours: int = 3
    beta_window_days: int = 30
    rebalance_hours: int = 1
    quantile: float = 0.2
    min_symbols: int = 10
    gross: float = 1.0
    anchor: str = "BTCUSDT"
    family = "lead_lag"
    provenance = "literature:crypto-lead-lag (btc leads alts)"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        zero = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        if self.anchor not in data.symbols:
            return zero
        close = data.close.ffill()
        log_close = np.log(close)
        r1 = log_close.diff()
        window = _hours_to_bars(24 * self.beta_window_days, data.interval)
        btc = r1[self.anchor]
        cov = r1.rolling(window, min_periods=window // 2).cov(btc)
        var = btc.rolling(window, min_periods=window // 2).var()
        beta = cov.div(var.where(var > 0), axis=0)
        look = _hours_to_bars(self.lookback_hours, data.interval)
        rl = log_close.diff(look)
        residual = beta.mul(rl[self.anchor], axis=0) - rl
        tradable = data.tradable.copy()
        tradable[self.anchor] = False
        weights = quantile_long_short(residual, tradable & residual.notna(), self.quantile, self.min_symbols, self.gross)
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)


@dataclass(frozen=True)
class CashAndCarry(Strategy):
    """Delta-neutral basis trade: long spot + short perpetual of the same coin while funding is rich.

    Annualised funding = mean settled rate over ``lookback_days`` x settlements per year (observed
    frequency). Open a pair when it exceeds ``entry_apr``, close when it falls below ``exit_apr``
    or when either leg stops being tradable. Each leg gets ``gross_per_pair / 2`` of equity, and
    at most ``max_pairs`` pairs are funded at full size (more active pairs share the budget).
    Columns must follow the ``SYMBOL`` / ``SYMBOL@spot`` convention (see research/carry_data.py).
    """

    entry_apr: float = 0.15
    exit_apr: float = 0.05
    lookback_days: int = 3
    rebalance_hours: int = 24
    max_pairs: int = 10
    gross_per_pair: float = 0.16
    family = "basis_carry"
    provenance = "textbook:cash-and-carry funding capture"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        from quant.signals.latch import latch_positions

        weights = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        if data.funding is None:
            raise ValueError("CashAndCarry needs funding rates")
        perps = [c for c in data.symbols if not c.endswith("@spot") and f"{c}@spot" in data.symbols]
        if not perps:
            return weights
        spots = [f"{c}@spot" for c in perps]
        bars = _hours_to_bars(24 * self.lookback_days, data.interval)
        settled = data.funding[perps]
        total = settled.fillna(0.0).rolling(bars, min_periods=1).sum()
        count = settled.notna().rolling(bars, min_periods=1).sum()
        apr = (total / self.lookback_days * 365.0).where(count >= self.lookback_days)
        tradable = data.tradable
        both = tradable[perps].to_numpy() & tradable[spots].to_numpy()
        both = pd.DataFrame(both, index=data.index, columns=perps)
        on = latch_positions((apr > self.entry_apr) & both, (apr < self.exit_apr) | ~both)
        active = on.sum(axis=1)
        leg = (self.gross_per_pair / 2) / np.maximum(active / self.max_pairs, 1.0)
        sized = on.mul(leg, axis=0)
        weights[perps] = -sized.to_numpy()
        weights[spots] = sized.to_numpy()
        return hold_between_rebalances(weights, self.rebalance_hours, interval=data.interval)
