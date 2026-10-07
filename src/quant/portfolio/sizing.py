"""Position sizing helpers (all causal)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.timeframes import bars_per_year, interval_ms


def rebalance_bars(index: pd.DatetimeIndex, every_hours: int, offset_hours: int = 0) -> np.ndarray:
    """True on bars whose open time is a multiple of ``every_hours`` (UTC) shifted by ``offset_hours``."""
    if every_hours <= 0:
        raise ValueError("every_hours must be positive")
    ns = pd.DatetimeIndex(index).as_unit("ns").asi8
    hours = ns // 3_600_000_000_000
    on_hour = (ns % 3_600_000_000_000) == 0
    return on_hour & ((hours - offset_hours) % every_hours == 0)


def hold_between_rebalances(weights: pd.DataFrame, every_hours: int, offset_hours: int = 0,
                            interval: str | None = None) -> pd.DataFrame:
    """Keep weights decided on rebalance bars until the next rebalance bar.

    For bars longer than ``every_hours`` every bar is a rebalance bar.
    """
    if interval is not None and interval_ms(interval) >= every_hours * 3_600_000:
        return weights
    mask = rebalance_bars(weights.index, every_hours, offset_hours)
    cond = np.broadcast_to(mask[:, None], weights.shape)
    return weights.where(cond, np.nan).ffill().fillna(0.0)


def annualised_vol(close: pd.DataFrame, window_bars: int, interval: str) -> pd.DataFrame:
    rets = np.log(close).diff()
    return rets.rolling(window_bars, min_periods=max(10, window_bars // 2)).std() * np.sqrt(bars_per_year(interval))


def vol_scaled(signal: pd.DataFrame, ann_vol: pd.DataFrame, target_vol: float, max_weight: float) -> pd.DataFrame:
    """signal * target_vol / realised vol, clipped to +/- max_weight; zero where vol is unknown."""
    scale = (target_vol / ann_vol.where(ann_vol > 0)).clip(upper=max_weight)
    return (signal * scale).fillna(0.0)


def normalise_by_count(weights: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    """Divide by the number of active symbols on each bar (equal risk budget across the book)."""
    n = active.sum(axis=1).replace(0, np.nan)
    return weights.div(n, axis=0).fillna(0.0)
