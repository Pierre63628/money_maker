"""Spot/perpetual pairs for basis (cash-and-carry) research.

Columns: ``SYMBOL`` = USDT-M perpetual, ``SYMBOL@spot`` = spot pair of the same ticker. Only tickers
listed identically on both markets are used (multiplier contracts such as 1000PEPEUSDT are skipped
to avoid mismatched price scales). Funding exists only on perpetual columns; both legs share the
perpetual's point-in-time eligibility.
"""

from __future__ import annotations

import pandas as pd

from quant.data.market_data import FIELDS, MarketData, resample_bars
from quant.data.store import ParquetStore
from quant.data.universe import UniverseRule, ever_eligible, expand_to_bars, point_in_time_eligibility
from quant.data.validation import mask_stale_runs
from quant.research.universe_data import cached_daily_quote_volume, crypto_symbols

SPOT_SUFFIX = "@spot"


def _bars(store: ParquetStore, market: str, symbol: str, source_interval: str, interval: str, start, end) -> pd.DataFrame:
    raw = store.read_klines(market, symbol, source_interval, start, end, columns=list(FIELDS))
    cleaned, _ = mask_stale_runs(raw)
    return resample_bars(cleaned.dropna(subset=["close"]), source_interval, interval)


def load_carry_pairs(store: ParquetStore, rule: UniverseRule, interval: str, start, end,
                     source_interval: str = "1h") -> MarketData:
    perps = crypto_symbols(store, "um", source_interval)
    mask = point_in_time_eligibility(cached_daily_quote_volume(store, perps, source_interval, "um"), rule)
    spot_available = set(store.symbols("spot", source_interval))
    pairs = [s for s in ever_eligible(mask, start, end) if s in spot_available]
    if not pairs:
        raise ValueError("no perpetual in the universe has a spot pair")
    frames, funding = {}, {}
    for sym in pairs:
        perp = _bars(store, "um", sym, source_interval, interval, start, end)
        spot = _bars(store, "spot", sym, source_interval, interval, start, end)
        if perp.empty or spot.empty:
            continue
        frames[sym] = perp
        frames[sym + SPOT_SUFFIX] = spot
        if store.has_funding("um", sym):
            funding[sym] = store.read_funding("um", sym, start, end)
    data = MarketData.from_frames(frames, interval, funding)
    daily = mask.reindex(columns=[c.removesuffix(SPOT_SUFFIX) for c in data.symbols], fill_value=False)
    daily.columns = data.symbols
    # a spot leg also needs its own recent liquidity (spot pairs are delisted independently of perpetuals)
    spot_cols = [c for c in data.symbols if c.endswith(SPOT_SUFFIX)]
    spot_daily_qv = data.quote_volume[spot_cols].resample("1D").sum(min_count=1)
    spot_alive = (spot_daily_qv.rolling(7, min_periods=1).median() > 0).shift(1, fill_value=False)
    daily.loc[:, spot_cols] = daily[spot_cols] & spot_alive.reindex(daily.index).fillna(False).astype(bool)
    return data.with_universe(expand_to_bars(daily, data.index))
