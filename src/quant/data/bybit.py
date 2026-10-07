"""Bybit public REST klines (used to audit strategies backtested on Bybit prices)."""

from __future__ import annotations

import time

import pandas as pd
import requests

from quant.data.timeframes import interval_ms, to_utc

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
_BYBIT_INTERVAL = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120",
                   "4h": "240", "6h": "360", "12h": "720", "1d": "D"}


def fetch_bybit_klines(symbol: str, interval: str, start, end, category: str = "linear",
                       session: requests.Session | None = None) -> pd.DataFrame:
    """OHLCV bars with open time in [start, end). Bybit returns newest-first pages of <= 1000 bars."""
    session = session or requests.Session()
    step = interval_ms(interval)
    start_ms = int(to_utc(start).timestamp() * 1000)
    end_ms = int(to_utc(end).timestamp() * 1000)
    rows: dict[int, list[float]] = {}
    cursor_end = end_ms - 1
    while cursor_end >= start_ms:
        params = {"category": category, "symbol": symbol, "interval": _BYBIT_INTERVAL[interval],
                  "start": start_ms, "end": cursor_end, "limit": 1000}
        resp = session.get(BYBIT_KLINE_URL, params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()["result"]["list"]
        if not page:
            break
        for r in page:
            rows[int(r[0])] = [float(x) for x in r[1:7]]
        oldest = min(int(r[0]) for r in page)
        if oldest <= start_ms:
            break
        cursor_end = oldest - step
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "turnover"])
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["open", "high", "low", "close", "volume", "turnover"])
    df.index = pd.to_datetime(df.index, unit="ms", utc=True).as_unit("ns")
    df.index.name = "ts"
    return df.sort_index()
