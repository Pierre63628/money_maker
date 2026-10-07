"""Causal feature functions.

Contract: the value at row ``t`` depends only on rows ``<= t`` (i.e. it is known
at the close of bar ``t``). No centred windows, no backward fills, no full-sample
normalisation. ``tests/test_lookahead_and_features.py`` checks every function by truncation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def log_returns(close: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return np.log(close).diff(periods)


def sma(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).mean()


def ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_std(x: pd.DataFrame, window: int) -> pd.DataFrame:
    return x.rolling(window, min_periods=window).std()


def realized_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation of 1-bar log returns (per bar, not annualised)."""
    return rolling_std(log_returns(close), window)


def zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std()
    return (x - mean) / sd.where(sd > 0)


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    # fmax ignores NaN, so the first bar's true range is high - low
    return np.fmax(np.fmax(high - low, (high - prev_close).abs()), (low - prev_close).abs())


@njit(cache=True)
def _wilder(values, window):
    T, N = values.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        count = 0
        acc = 0.0
        prev = np.nan
        for t in range(T):
            x = values[t, j]
            if np.isnan(prev):
                if np.isnan(x):
                    count = 0
                    acc = 0.0
                    continue
                acc += x
                count += 1
                if count == window:
                    prev = acc / window
                    out[t, j] = prev
            else:
                if not np.isnan(x):
                    prev = (prev * (window - 1) + x) / window
                out[t, j] = prev
    return out


def wilder_average(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """Wilder's running average (RMA): seeded with the mean of the first ``window`` values.

    Missing values after the seed carry the previous average forward.
    """
    arr = np.ascontiguousarray(x.to_numpy(dtype=np.float64))
    return pd.DataFrame(_wilder(arr, window), index=x.index, columns=x.columns)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's average true range."""
    return wilder_average(true_range(high, low, close), window)


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI in [0, 100]."""
    delta = close.diff()
    gain = wilder_average(delta.clip(lower=0.0), window)
    loss = wilder_average(-delta.clip(upper=0.0), window)
    rs = gain / loss
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss > 0, 100.0).where(gain.notna())


def donchian_high(high: pd.DataFrame, window: int) -> pd.DataFrame:
    """Highest high of the ``window`` bars *before* bar t (excludes bar t itself)."""
    return high.rolling(window, min_periods=window).max().shift(1)


def donchian_low(low: pd.DataFrame, window: int) -> pd.DataFrame:
    """Lowest low of the ``window`` bars *before* bar t (excludes bar t itself)."""
    return low.rolling(window, min_periods=window).min().shift(1)
