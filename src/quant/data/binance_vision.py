"""Binance public historical data (https://data.binance.vision).

Why this source:
- free, exchange-native OHLCV for spot and USDT-M perpetuals, plus funding rates;
- every file ships with a SHA256 checksum (integrity is verified, never assumed);
- delisted symbols stay in the bucket, so a universe built from the bucket
  listing is free of survivorship bias.

Only complete days/months are published, so there are no partial bars.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quant.data.timeframes import interval_ms, to_utc

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision"
S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
MARKET_PREFIX = {"um": "data/futures/um", "spot": "data/spot"}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume",
                 "taker_buy_volume", "taker_buy_quote_volume"]
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_DATE_IN_KEY = re.compile(r"-(\d{4}-\d{2}(?:-\d{2})?)\.zip$")


class ChecksumError(RuntimeError):
    pass


def _to_utc_ms_index(raw: pd.Series) -> pd.DatetimeIndex:
    """Epoch integers -> UTC DatetimeIndex. Binance switched some files to microseconds."""
    values = raw.astype("int64").to_numpy()
    if len(values) and values.max() > 10**14:
        values = values // 1000
    return pd.DatetimeIndex(pd.to_datetime(values, unit="ms", utc=True)).as_unit("ns")


def _read_single_csv(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected exactly one csv in archive, got {names}")
        text = zf.read(names[0])
    first_char = text.lstrip()[:1]
    header = None if first_char.isdigit() else 0
    return pd.read_csv(io.BytesIO(text), header=header)


def parse_klines_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a kline archive into the canonical OHLCV frame (index ``ts`` = bar open time, UTC)."""
    df = _read_single_csv(zip_bytes)
    if df.shape[1] != len(KLINE_COLUMNS):
        raise ValueError(f"unexpected kline column count {df.shape[1]}")
    df.columns = KLINE_COLUMNS
    out = pd.DataFrame(index=_to_utc_ms_index(df["open_time"]))
    out.index.name = "ts"
    for col in FLOAT_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    return out


def parse_funding_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a funding-rate archive: index ``ts`` = funding settlement time (UTC)."""
    df = _read_single_csv(zip_bytes)
    if df.shape[1] != 3:
        raise ValueError(f"unexpected funding column count {df.shape[1]}")
    df.columns = ["calc_time", "funding_interval_hours", "funding_rate"]
    out = pd.DataFrame(index=_to_utc_ms_index(df["calc_time"]))
    out.index.name = "ts"
    out["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce").to_numpy(dtype=np.float64)
    out["funding_interval_hours"] = pd.to_numeric(df["funding_interval_hours"], errors="coerce").to_numpy()
    return out


METRICS_RENAME = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "top_trader_account_ls",
    "sum_toptrader_long_short_ratio": "top_trader_position_ls",
    "count_long_short_ratio": "global_account_ls",
    "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
}


def parse_metrics_zip(zip_bytes: bytes) -> pd.DataFrame:
    """Parse a futures ``metrics`` archive: open interest and positioning ratios (UTC timestamps)."""
    df = _read_single_csv(zip_bytes)
    if "create_time" not in df.columns:
        raise ValueError("metrics archive without a create_time column")
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df["create_time"], utc=True), name="ts").as_unit("ns"))
    for source, target in METRICS_RENAME.items():
        out[target] = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=np.float64) if source in df else np.nan
    return out[~out.index.duplicated(keep="last")].sort_index()


def verify_checksum(data: bytes, checksum_text: str) -> None:
    expected = checksum_text.strip().split()[0].lower()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumError(f"sha256 mismatch: expected {expected}, got {actual}")


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate archives, sort by time and drop exact duplicates.

    Conflicting duplicates (same timestamp, different values) are kept so the
    validation step reports them instead of silently choosing one.
    """
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index(kind="stable")
    dup_exact = df.reset_index().duplicated(keep="first").to_numpy()
    return df[~dup_exact]


@dataclass
class BinanceVisionClient:
    raw_dir: Path
    market: str = "um"
    timeout: float = 60.0
    retries: int = 4
    max_workers: int = 8
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        if self.market not in MARKET_PREFIX:
            raise ValueError(f"market must be one of {sorted(MARKET_PREFIX)}")
        self.raw_dir = Path(self.raw_dir)
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=64)
        self.session.mount("https://", adapter)

    # ------------------------------------------------------------------ http
    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 404:
                    return resp
                if resp.status_code in (418, 429) or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp
            except (requests.RequestException, requests.HTTPError) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GET failed after {self.retries} attempts: {url}") from last_exc

    def _list(self, prefix: str) -> tuple[list[str], list[str]]:
        """List common prefixes (sub-directories) and keys under an S3 prefix, following pagination."""
        prefixes: list[str] = []
        keys: list[str] = []
        marker = ""
        while True:
            params = {"delimiter": "/", "prefix": prefix}
            if marker:
                params["marker"] = marker
            root = ET.fromstring(self._get(S3_LIST_URL, params=params).content)
            page_prefixes = [p.text for p in root.iter(f"{_S3_NS}Prefix") if p.text and p.text != prefix]
            page_keys = [k.text for k in root.iter(f"{_S3_NS}Key") if k.text]
            prefixes.extend(page_prefixes)
            keys.extend(page_keys)
            truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "false").lower() == "true"
            if not truncated:
                break
            marker = root.findtext(f"{_S3_NS}NextMarker") or (page_keys or page_prefixes)[-1]
        return prefixes, keys

    # --------------------------------------------------------------- listing
    def list_symbols(self, dataset: str = "klines") -> list[str]:
        """All symbols ever published (including delisted ones)."""
        prefixes, _ = self._list(f"{MARKET_PREFIX[self.market]}/monthly/{dataset}/")
        return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes)

    def list_archives(self, symbol: str, dataset: str, interval: str | None, freq: str) -> list[str]:
        parts = [MARKET_PREFIX[self.market], freq, dataset, symbol]
        if interval:
            parts.append(interval)
        _, keys = self._list("/".join(parts) + "/")
        return sorted(k for k in keys if k.endswith(".zip"))

    # -------------------------------------------------------------- download
    def fetch_archive(self, key: str) -> bytes:
        """Return archive bytes, verified against the published SHA256. Cached under ``raw_dir``."""
        local = self.raw_dir / key
        marker = local.with_suffix(local.suffix + ".verified")
        if local.exists() and marker.exists():
            return local.read_bytes()
        resp = self._get(f"{BASE_URL}/{key}")
        if resp.status_code == 404:
            raise FileNotFoundError(key)
        data = resp.content
        checksum = self._get(f"{BASE_URL}/{key}.CHECKSUM")
        if checksum.status_code == 404:
            raise ChecksumError(f"no checksum published for {key}; refusing unverified data")
        verify_checksum(data, checksum.text)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        marker.write_text(hashlib.sha256(data).hexdigest())
        return data

    def _archive_keys(self, symbol: str, dataset: str, interval: str | None,
                      start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[str]:
        monthly = self.list_archives(symbol, dataset, interval, "monthly")
        keys = list(monthly)
        if dataset == "klines":
            keys += self._recent_daily_keys(symbol, interval, monthly)

        def in_range(key: str) -> bool:
            stamp = _DATE_IN_KEY.search(key).group(1)
            period_start = pd.Timestamp(stamp + ("-01" if len(stamp) == 7 else ""), tz="UTC")
            period_end = period_start + (pd.offsets.MonthBegin(1) if len(stamp) == 7 else pd.Timedelta(days=1))
            if start is not None and period_end <= start:
                return False
            if end is not None and period_start >= end:
                return False
            return True

        return [k for k in keys if in_range(k)]

    def _recent_daily_keys(self, symbol: str, interval: str, monthly: list[str]) -> list[str]:
        """Daily archives for days after the last monthly archive, up to yesterday (UTC).

        Keys are generated rather than listed (listing every daily key is slow); days that
        were never published are skipped at download time. Symbols without any monthly
        archive (very recent listings) fall back to a listing.
        """
        if not monthly:
            return self.list_archives(symbol, "klines", interval, "daily")
        last_month = max(_DATE_IN_KEY.search(k).group(1) for k in monthly)
        first_day = pd.Timestamp(last_month + "-01", tz="UTC") + pd.offsets.MonthBegin(1)
        yesterday = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        if yesterday - first_day > pd.Timedelta(days=62):
            return []  # no recent monthly archive: the symbol stopped trading, monthly files are complete
        prefix = f"{MARKET_PREFIX[self.market]}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-"
        return [f"{prefix}{day:%Y-%m-%d}.zip" for day in pd.date_range(first_day, yesterday, freq="D")]

    def _fetch_optional(self, key: str) -> bytes | None:
        try:
            return self.fetch_archive(key)
        except FileNotFoundError:
            return None

    def fetch_klines(self, symbol: str, interval: str, start: str | pd.Timestamp | None = None,
                     end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        interval_ms(interval)  # validates
        start_ts = to_utc(start) if start is not None else None
        end_ts = to_utc(end) if end is not None else None
        keys = self._archive_keys(symbol, "klines", interval, start_ts, end_ts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            blobs = list(pool.map(self._fetch_optional, keys))
        frames = [parse_klines_zip(b) for b in blobs if b is not None]
        df = merge_frames(frames)
        if df.empty:
            return df
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index < end_ts]
        return df

    def fetch_metrics(self, symbol: str, start, end) -> pd.DataFrame:
        """Daily ``metrics`` archives for [start, end]. Days never published are skipped (404)."""
        start_ts, end_ts = to_utc(start).normalize(), to_utc(end).normalize()
        prefix = f"{MARKET_PREFIX[self.market]}/daily/metrics/{symbol}/{symbol}-metrics-"
        keys = [f"{prefix}{day:%Y-%m-%d}.zip" for day in pd.date_range(start_ts, end_ts, freq="D")]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            blobs = list(pool.map(self._fetch_optional, keys))
        frames = [parse_metrics_zip(b) for b in blobs if b is not None]
        df = merge_frames(frames)
        return df if df.empty else df[(df.index >= start_ts) & (df.index < end_ts + pd.Timedelta(days=1))]

    def fetch_funding(self, symbol: str, start: str | pd.Timestamp | None = None,
                      end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if self.market != "um":
            raise ValueError("funding rates exist only for futures markets")
        start_ts = to_utc(start) if start is not None else None
        end_ts = to_utc(end) if end is not None else None
        keys = self._archive_keys(symbol, "fundingRate", None, start_ts, end_ts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            frames = list(pool.map(lambda k: parse_funding_zip(self.fetch_archive(k)), keys))
        df = merge_frames(frames)
        if df.empty:
            return df
        if start_ts is not None:
            df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index < end_ts]
        return df
