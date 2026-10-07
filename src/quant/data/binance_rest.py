"""Binance USDT-M public REST API: fills gaps left by Binance Vision and extends recent history.

Every row added from REST carries ``source = 1`` (Vision rows are ``0``) so repaired
data stays auditable. Before inserting, overlapping bars are cross-checked against
Vision; any disagreement aborts the repair for that gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from quant.data.timeframes import interval_ms, interval_to_timedelta, to_utc

FAPI_URL = "https://fapi.binance.com"
SPOT_URL = "https://api.binance.com"
SOURCE_VISION, SOURCE_REST = 0, 1


@dataclass
class BinanceRestClient:
    timeout: float = 30.0
    pause: float = 0.12
    session: requests.Session = field(default_factory=requests.Session)
    market: str = "um"   # "um" (USDT-M futures) or "spot"

    @property
    def _base(self) -> str:
        return SPOT_URL if self.market == "spot" else FAPI_URL

    @property
    def _klines_path(self) -> str:
        return "/api/v3/klines" if self.market == "spot" else "/fapi/v1/klines"

    def _get(self, path: str, params: dict) -> list:
        for attempt in range(5):
            try:
                resp = self.session.get(self._base + path, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):  # transient: back off and retry
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code in (418, 429) or resp.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            time.sleep(self.pause)
            return resp.json()
        raise RuntimeError(f"REST request failed repeatedly: {path} {params}")

    def klines(self, symbol: str, interval: str, start, end) -> pd.DataFrame:
        """Bars with open time in [start, end)."""
        step = interval_ms(interval)
        cur = int(to_utc(start).timestamp() * 1000)
        stop = int(to_utc(end).timestamp() * 1000)
        rows: list[list] = []
        while cur < stop:
            page = self._get(self._klines_path, {"symbol": symbol, "interval": interval, "startTime": cur,
                                                 "endTime": stop - 1, "limit": 1000 if self.market == "spot" else 1500})
            if not page:
                break
            rows.extend(page)
            last_open = int(page[-1][0])
            if last_open + step <= cur:
                break
            cur = last_open + step
        if not rows:
            return _empty_klines()
        arr = pd.DataFrame(rows).iloc[:, :11]
        arr.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                       "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
        out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(arr["open_time"].astype("int64"), unit="ms", utc=True),
                                                  name="ts").as_unit("ns"))
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            out[col] = arr[col].astype(float).to_numpy()
        out["trades"] = arr["trades"].astype("int64").to_numpy()
        out["source"] = np.int8(SOURCE_REST)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out[out.index < to_utc(end)]

    def metrics(self, symbol: str, start, end, period: str = "5m", fields: list[str] | None = None) -> pd.DataFrame:
        """Recent open interest and positioning ratios (public endpoints keep about 30 days).

        Each endpoint returns at most 500 points, so the window is walked in chunks.
        """
        start_ms, end_ms = int(to_utc(start).timestamp() * 1000), int(to_utc(end).timestamp() * 1000)
        step_ms = 500 * int(pd.Timedelta(period).total_seconds() * 1000)
        frames = []
        for path, mapping in METRICS_ENDPOINTS.items():
            if fields is not None and not set(mapping.values()) & set(fields):
                continue  # endpoint not needed for the requested columns
            parts = []
            cursor = start_ms
            while cursor < end_ms:
                chunk = _metrics_page(self, path, mapping, symbol, period, cursor, min(cursor + step_ms, end_ms))
                if not chunk.empty:
                    parts.append(chunk)
                cursor += step_ms
            if parts:
                merged_endpoint = pd.concat(parts).sort_index()
                # chunk boundaries repeat timestamps; axis=1 concat needs unique indexes
                frames.append(merged_endpoint[~merged_endpoint.index.duplicated(keep="last")])
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, axis=1).sort_index()
        return merged[~merged.index.duplicated(keep="last")]

    def funding(self, symbol: str, start, end) -> pd.DataFrame:
        cur = int(to_utc(start).timestamp() * 1000)
        stop = int(to_utc(end).timestamp() * 1000)
        rows: list[dict] = []
        while cur < stop:
            page = self._get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cur, "endTime": stop - 1,
                                                      "limit": 1000})
            if not page:
                break
            rows.extend(page)
            last = int(page[-1]["fundingTime"])
            if last < cur or len(page) < 1000:
                break
            cur = last + 1
        if not rows:
            return pd.DataFrame(columns=["funding_rate", "funding_interval_hours"])
        df = pd.DataFrame(rows)
        out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True),
                                                  name="ts").as_unit("ns"))
        out["funding_rate"] = df["fundingRate"].astype(float).to_numpy()
        out["funding_interval_hours"] = np.nan
        return out[~out.index.duplicated(keep="last")].sort_index()


METRICS_ENDPOINTS = {
    "/futures/data/openInterestHist": {"sumOpenInterest": "open_interest", "sumOpenInterestValue": "open_interest_value"},
    "/futures/data/globalLongShortAccountRatio": {"longShortRatio": "global_account_ls"},
    "/futures/data/topLongShortAccountRatio": {"longShortRatio": "top_trader_account_ls"},
    "/futures/data/topLongShortPositionRatio": {"longShortRatio": "top_trader_position_ls"},
    "/futures/data/takerlongshortRatio": {"buySellRatio": "taker_buy_sell_ratio"},
}


def _empty_klines() -> pd.DataFrame:
    cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
            "trades", "source"]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz="UTC", name="ts"))


def _metrics_page(client: "BinanceRestClient", path: str, mapping: dict, symbol: str, period: str,
                  start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = client._get(path, {"symbol": symbol, "period": period, "startTime": start_ms, "endTime": end_ms,
                              "limit": 500})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True),
                                              name="ts").as_unit("ns"))
    for source, target in mapping.items():
        out[target] = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=np.float64)
    return out[~out.index.duplicated(keep="last")].sort_index()


def missing_ranges(df: pd.DataFrame, interval: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous missing [start, end) ranges on the interval grid between the first and last bar."""
    if len(df) < 2:
        return []
    step = interval_to_timedelta(interval)
    idx = df.index.sort_values()
    gaps = idx[1:] - idx[:-1]
    out = []
    for prev, gap in zip(idx[:-1], gaps):
        if gap > step:
            out.append((prev + step, prev + gap))
    return out


@dataclass
class RepairLog:
    symbol: str
    gap_start: pd.Timestamp
    gap_end: pd.Timestamp
    missing_bars: int
    recovered_bars: int
    status: str


def repair_gaps(df: pd.DataFrame, symbol: str, interval: str, client: BinanceRestClient,
                max_gap: pd.Timedelta = pd.Timedelta(days=45), check_bars: int = 3) -> tuple[pd.DataFrame, list[RepairLog]]:
    """Fill grid gaps from REST after verifying REST agrees with the stored bars around each gap.

    Gaps longer than ``max_gap`` are left alone: they are usually delisting/relisting
    of a reused ticker, which must not be stitched into one price series.
    """
    step = interval_to_timedelta(interval)
    if "source" not in df.columns:
        df = df.assign(source=np.int8(SOURCE_VISION))
    logs: list[RepairLog] = []
    patches = []
    for start, end in missing_ranges(df, interval):
        n_missing = int((end - start) / step)
        if end - start > max_gap:
            logs.append(RepairLog(symbol, start, end, n_missing, 0, "skipped_long_gap"))
            continue
        try:
            fetched = client.klines(symbol, interval, start - check_bars * step, end + check_bars * step)
        except requests.HTTPError as exc:  # delisted symbols are unknown to the live API
            logs.append(RepairLog(symbol, start, end, n_missing, 0, f"rest_unavailable ({exc.response.status_code})"))
            continue
        overlap = df.index.intersection(fetched.index)
        if len(overlap) == 0:
            logs.append(RepairLog(symbol, start, end, n_missing, 0, "no_overlap_to_verify"))
            continue
        cols = ["open", "high", "low", "close"]
        a = df.loc[overlap, cols].to_numpy(dtype=float)
        b = fetched.loc[overlap, cols].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-9, atol=0.0):
            logs.append(RepairLog(symbol, start, end, n_missing, 0, "rest_disagrees_with_vision"))
            continue
        patch = fetched[(fetched.index >= start) & (fetched.index < end)]
        patches.append(patch)
        status = ("repaired" if len(patch) == n_missing else
                  "absent_from_rest_too" if len(patch) == 0 else "partially_repaired")
        logs.append(RepairLog(symbol, start, end, n_missing, len(patch), status))
    if patches:
        df = pd.concat([df, *patches]).sort_index()
        df = df[~df.index.duplicated(keep="first")]
    df["source"] = df["source"].fillna(SOURCE_VISION).astype(np.int8)
    return df, logs
