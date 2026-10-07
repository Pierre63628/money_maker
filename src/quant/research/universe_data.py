"""Load market data for a point-in-time universe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.data.market_data import MarketData, load_market_data
from quant.data.store import ParquetStore
from quant.data.universe import UniverseRule, daily_quote_volume, ever_eligible, expand_to_bars, \
    point_in_time_eligibility


def crypto_symbols(store: ParquetStore, market: str = "um", interval: str = "1h") -> list[str]:
    stored = set(store.symbols(market, interval))
    path = store.root / store.exchange / market / "symbol_classification.csv"
    if not path.exists():
        return sorted(stored)
    classes = pd.read_csv(path, index_col=0)
    return sorted(stored & set(classes.index[classes["is_crypto"].astype(bool)]))


def cached_daily_quote_volume(store: ParquetStore, symbols: list[str], interval: str = "1h",
                              market: str = "um") -> pd.DataFrame:
    """Daily quote volume for all symbols, cached and rebuilt when any kline file is newer than the cache."""
    cache = store.root / store.exchange / market / f"daily_quote_volume_{interval}.parquet"
    newest = max((store._path(market, "klines", s, interval).stat().st_mtime for s in symbols), default=0.0)
    if cache.exists() and cache.stat().st_mtime >= newest:
        cached = pd.read_parquet(cache)
        if set(symbols) <= set(cached.columns):
            return cached[symbols]
    table = daily_quote_volume(_ColumnReader(store), symbols, interval, market)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache)
    return table


class _ColumnReader:
    """Store facade reading only the quote volume column."""

    def __init__(self, store: ParquetStore):
        self.store = store

    def read_klines(self, market, symbol, interval, start=None, end=None):
        return self.store.read_klines(market, symbol, interval, start, end, columns=["quote_volume"])


def load_universe(store: ParquetStore, rule: UniverseRule, interval: str, start, end,
                  source_interval: str = "1h", market: str = "um", with_order_flow: bool = False,
                  with_metrics: bool = False) -> MarketData:
    candidates = crypto_symbols(store, market, source_interval)
    daily_qv = cached_daily_quote_volume(store, candidates, source_interval, market)
    mask = point_in_time_eligibility(daily_qv, rule)
    members = ever_eligible(mask, start, end)
    if not members:
        raise ValueError("no symbol is ever eligible in the requested window")
    data = load_market_data(store, members, interval, start, end, market=market, source_interval=source_interval,
                            with_order_flow=with_order_flow, with_metrics=with_metrics)
    return data.with_universe(expand_to_bars(mask[data.symbols], data.index))
