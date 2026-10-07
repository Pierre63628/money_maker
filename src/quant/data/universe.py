"""Survivorship-free, point-in-time tradable universe.

1. Candidate symbols = every USDT-M perpetual ever published on Binance Vision
   (delisted ones included), minus non-crypto underlyings (equities, commodities,
   indices, pre-IPO) and stablecoins.
2. Eligibility on day ``d`` uses only information up to the end of day ``d - 1``:
   listing age, recent activity, trailing median daily quote volume, liquidity rank.

Selecting "every symbol ever eligible" for loading bars is safe: it cannot add a
symbol that was not eligible at a given time, and the ranking is computed across
all candidates, not only the loaded ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
STABLE_OR_PEGGED_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USD1", "USDE", "PAXG", "XAUT", "EUR", "GBP"}
CRYPTO_UNDERLYING = "COIN"


def fetch_exchange_info(path: Path | None = None, session: requests.Session | None = None) -> dict:
    session = session or requests.Session()
    resp = session.get(EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info))
    return info


def classify_symbols(vision_symbols: list[str], exchange_info: dict) -> pd.DataFrame:
    """One row per USDT perpetual symbol with its underlying type and eligibility for crypto research.

    Symbols absent from the current exchangeInfo were delisted before it was captured;
    all non-crypto perpetuals were launched long after, so they are classified as COIN.
    """
    meta = {s["symbol"]: s for s in exchange_info.get("symbols", [])}
    rows = []
    for sym in vision_symbols:
        if not sym.endswith("USDT") or "_" in sym:
            continue
        m = meta.get(sym)
        base = m["baseAsset"] if m else sym[: -len("USDT")]
        underlying = m.get("underlyingType", CRYPTO_UNDERLYING) if m else CRYPTO_UNDERLYING
        contract = m.get("contractType", "PERPETUAL") if m else "PERPETUAL"
        rows.append({
            "symbol": sym,
            "base": base,
            "underlying_type": underlying,
            "contract_type": contract,
            "status": m["status"] if m else "DELISTED_UNKNOWN",
            "onboard_date": pd.to_datetime(m["onboardDate"], unit="ms", utc=True) if m else pd.NaT,
            "in_exchange_info": m is not None,
        })
    df = pd.DataFrame(rows).set_index("symbol")
    df["is_crypto"] = (
        (df["underlying_type"] == CRYPTO_UNDERLYING)
        & (df["contract_type"] == "PERPETUAL")
        & ~df["base"].isin(STABLE_OR_PEGGED_BASES)
    )
    return df


def daily_quote_volume(store, symbols: list[str], interval: str = "1h", market: str = "um") -> pd.DataFrame:
    """Wide table (UTC day x symbol) of summed quote volume. NaN = no bars that day."""
    series = {}
    for sym in symbols:
        try:
            df = store.read_klines(market, sym, interval)
        except FileNotFoundError:
            continue
        if df.empty:
            continue
        series[sym] = df["quote_volume"].resample("1D").sum(min_count=1)
    return pd.DataFrame(series).sort_index()


@dataclass(frozen=True)
class UniverseRule:
    min_history_days: int = 30          # days with data before a symbol can be traded
    liquidity_window_days: int = 30     # trailing window for the median daily quote volume
    min_median_quote_volume: float = 20e6
    top_n: int | None = 50              # keep the N most liquid eligible symbols (None = all)
    require_trading_yesterday: bool = True


def point_in_time_eligibility(daily_qv: pd.DataFrame, rule: UniverseRule = UniverseRule()) -> pd.DataFrame:
    """Boolean (day x symbol) mask; the value for day d uses only days < d."""
    has_data = daily_qv.notna()
    age = has_data.cumsum().shift(1).fillna(0)
    liq = (daily_qv.rolling(rule.liquidity_window_days, min_periods=max(1, rule.liquidity_window_days // 2))
           .median().shift(1))
    eligible = (age >= rule.min_history_days) & (liq >= rule.min_median_quote_volume)
    if rule.require_trading_yesterday:
        eligible &= has_data.shift(1, fill_value=False)
    if rule.top_n is not None:
        rank = liq.where(eligible).rank(axis=1, ascending=False, method="first")
        eligible &= rank <= rule.top_n
    return eligible.astype(bool)


def expand_to_bars(daily_mask: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map a daily mask onto intraday bars: every bar of day d gets the mask of day d."""
    days = bar_index.floor("1D")
    out = daily_mask.reindex(days).fillna(False)
    out.index = bar_index
    return out.astype(bool)


def ever_eligible(daily_mask: pd.DataFrame, start=None, end=None) -> list[str]:
    window = daily_mask.loc[start:end] if (start is not None or end is not None) else daily_mask
    return sorted(window.columns[window.any(axis=0)])
