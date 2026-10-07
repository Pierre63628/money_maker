from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.market_data import MarketData


def frame_from_arrays(open_, close=None, high=None, low=None, start="2024-01-01", interval="1h",
                      quote_volume=1e12) -> pd.DataFrame:
    open_ = np.asarray(open_, dtype=float)
    close = open_ + 0.5 if close is None else np.asarray(close, dtype=float)
    high = np.maximum(open_, close) + 1.0 if high is None else np.asarray(high, dtype=float)
    low = np.minimum(open_, close) - 1.0 if low is None else np.asarray(low, dtype=float)
    idx = pd.date_range(start, periods=len(open_), freq=pd.Timedelta(interval), tz="UTC", name="ts").as_unit("ns")
    qv = np.broadcast_to(np.asarray(quote_volume, dtype=float), open_.shape)
    share = np.clip(0.5 + 20 * (close / open_ - 1), 0.05, 0.95)  # aggressive buying on up bars
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": qv / np.where(close > 0, close, 1.0), "quote_volume": qv,
                         "taker_buy_quote_volume": qv * share}, index=idx)


def market(frames: dict[str, pd.DataFrame], interval: str = "1h", funding=None) -> MarketData:
    return MarketData.from_frames(frames, interval, funding)


def gbm_frame(n: int, seed: int, start="2023-01-01", interval="1h", sigma=0.01, drift=0.0,
              quote_volume=5e7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_close = np.log(100.0) + np.cumsum(drift + sigma * rng.standard_normal(n))
    close = np.exp(log_close)
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(0.0005 * rng.standard_normal(n))
    wick = np.abs(sigma * rng.standard_normal(n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick * 0.9
    qv = quote_volume * np.exp(0.3 * rng.standard_normal(n))
    return frame_from_arrays(open_, close, high, low, start=start, interval=interval, quote_volume=qv)


@pytest.fixture
def synthetic_market() -> MarketData:
    frames = {"AAA": gbm_frame(3000, 1), "BBB": gbm_frame(3000, 2, sigma=0.015)}
    return market(frames)
