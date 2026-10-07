"""Market data for paper trading, from public endpoints only (no account, no keys).

``HybridBarSource`` serves verified Binance Vision history from the research store and
appends the most recent closed bars from REST into a separate live cache, so the research
store stays pure and live decisions see the same bar definitions as backtests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from quant.data.binance_rest import FAPI_URL, SPOT_URL, BinanceRestClient
from quant.data.market_data import (FIELDS, METRIC_LAST_COLUMNS, METRIC_MEAN_COLUMNS, clean_metrics,
                                    metrics_available_at, snapshots_to_bars)
from quant.data.store import ParquetStore
from quant.data.timeframes import interval_to_timedelta


@dataclass
class Quote:
    bid: float
    ask: float
    time: pd.Timestamp

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def half_spread_bps(self) -> float:
        return (self.ask - self.bid) / 2 / self.mid * 1e4


class MarketFeed:
    """Interface used by the trading loop (a fake implementation is used in tests)."""

    def bars(self, symbols: list[str], interval: str, start: pd.Timestamp, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        raise NotImplementedError

    def funding_since(self, symbol: str, start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    def funding_history(self, symbols: list[str], start: pd.Timestamp, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def quote_volume_24h(self) -> pd.Series:
        raise NotImplementedError

    def metrics_history(self, symbols: list[str], start: pd.Timestamp, now: pd.Timestamp,
                        interval: str = "1h") -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    def funding_now(self) -> pd.Series:
        raise NotImplementedError

    def retail_positioning(self, symbols: list[str], now: pd.Timestamp, days: int = 25) -> pd.DataFrame:
        raise NotImplementedError


SPOT_SUFFIX = "@spot"


def _write_cache(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Atomic parquet write. Several loops share one cache directory and wake at the same second: a
    plain ``to_parquet`` lets one read a file the other is half-way through writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, **kwargs)
    tmp.replace(path)


def _read_cache(path: Path) -> pd.DataFrame:
    """A cache is disposable: an unreadable file is dropped and rebuilt from the exchange."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        path.unlink(missing_ok=True)
        return pd.DataFrame()


def split_symbol(symbol: str) -> tuple[str, str]:
    """``BTCUSDT@spot`` -> ("spot", "BTCUSDT"); ``BTCUSDT`` -> ("um", "BTCUSDT")."""
    return ("spot", symbol[: -len(SPOT_SUFFIX)]) if symbol.endswith(SPOT_SUFFIX) else ("um", symbol)


@dataclass
class BinancePublicFeed(MarketFeed):
    store: ParquetStore
    cache_dir: Path
    rest: BinanceRestClient = field(default_factory=BinanceRestClient)
    spot_rest: BinanceRestClient = field(default_factory=lambda: BinanceRestClient(market="spot"))
    timeout: float = 20.0

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / interval / f"{symbol}.parquet"

    def bars(self, symbols: list[str], interval: str, start: pd.Timestamp, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Closed bars in [start, last closed bar]. Source interval is always 1h (resampled by the caller)."""
        step = interval_to_timedelta(interval)
        last_open = now.floor(step) - step  # the bar that opened one step ago has closed
        out = {}
        for sym in symbols:
            market, ticker = split_symbol(sym)
            client = self.spot_rest if market == "spot" else self.rest
            try:
                hist = self.store.read_klines(market, ticker, interval, start, columns=list(FIELDS))
            except FileNotFoundError:
                hist = pd.DataFrame(columns=list(FIELDS))
            path = self._cache_path(sym, interval)
            cached = _read_cache(path)
            frames = [f for f in (hist, cached) if len(f)]
            merged = pd.concat(frames) if frames else pd.DataFrame(columns=list(FIELDS))
            if len(merged):
                merged.index = pd.DatetimeIndex(merged.index).as_unit("ns")
                merged = merged[~merged.index.duplicated(keep="first")].sort_index()
            fetch_from = merged.index.max() + step if len(merged) else start
            if fetch_from <= last_open:
                try:
                    tail = client.klines(ticker, interval, fetch_from, last_open + step)[list(FIELDS)]
                except requests.HTTPError:
                    tail = pd.DataFrame(columns=list(FIELDS))
                if len(tail):
                    new_cache = pd.concat([cached, tail]) if len(cached) else tail
                    new_cache.index = pd.DatetimeIndex(new_cache.index).as_unit("ns")
                    new_cache = new_cache[~new_cache.index.duplicated(keep="last")].sort_index()
                    _write_cache(new_cache, path)
                    merged = pd.concat([merged, tail]) if len(merged) else tail
                    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
            if merged.empty:
                continue  # no history and nothing from REST (e.g. no spot pair): symbol unavailable
            merged.index = pd.DatetimeIndex(merged.index).as_unit("ns")
            merged = merged[(merged.index >= start) & (merged.index <= last_open)]
            out[sym] = merged.astype(float)
        return out

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        out = {}
        now = pd.Timestamp.now(tz="UTC")
        perps = {s for s in symbols if not s.endswith(SPOT_SUFFIX)}
        spots = {split_symbol(s)[1] for s in symbols if s.endswith(SPOT_SUFFIX)}
        if perps:
            resp = requests.get(f"{FAPI_URL}/fapi/v1/ticker/bookTicker", timeout=self.timeout)
            resp.raise_for_status()
            for row in resp.json():
                if row["symbol"] in perps:
                    out[row["symbol"]] = Quote(float(row["bidPrice"]), float(row["askPrice"]),
                                               pd.Timestamp(int(row["time"]), unit="ms", tz="UTC"))
        if spots:
            resp = requests.get(f"{SPOT_URL}/api/v3/ticker/bookTicker", timeout=self.timeout)
            resp.raise_for_status()
            for row in resp.json():
                if row["symbol"] in spots and float(row["bidPrice"]) > 0:
                    out[row["symbol"] + SPOT_SUFFIX] = Quote(float(row["bidPrice"]), float(row["askPrice"]), now)
        return out

    def funding_since(self, symbol: str, start: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
        if symbol.endswith(SPOT_SUFFIX):
            return pd.DataFrame(columns=["funding_rate"])
        return self.rest.funding(symbol, start, now)

    def funding_history(self, symbols: list[str], start: pd.Timestamp, now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Settled funding in [start, now]: Vision archives from the store, REST for the recent tail (cached)."""
        out = {}
        for sym in [x for x in symbols if not x.endswith(SPOT_SUFFIX)]:
            try:
                hist = self.store.read_funding("um", sym, start)[["funding_rate"]]
            except FileNotFoundError:
                hist = pd.DataFrame(columns=["funding_rate"])
            path = self.cache_dir / "funding" / f"{sym}.parquet"
            cached = _read_cache(path)
            frames = [f for f in (hist, cached) if len(f)]
            merged = pd.concat(frames) if frames else pd.DataFrame(columns=["funding_rate"])
            if len(merged):
                merged.index = pd.DatetimeIndex(merged.index).as_unit("ns")
                merged = merged[~merged.index.duplicated(keep="first")].sort_index()
            since = merged.index.max() + pd.Timedelta(milliseconds=1) if len(merged) else start
            if now - since > pd.Timedelta(hours=1):
                try:
                    tail = self.rest.funding(sym, since, now)[["funding_rate"]]
                except requests.HTTPError:
                    tail = pd.DataFrame(columns=["funding_rate"])
                if len(tail):
                    new_cache = pd.concat([cached, tail]) if len(cached) else tail
                    new_cache.index = pd.DatetimeIndex(new_cache.index).as_unit("ns")
                    new_cache = new_cache[~new_cache.index.duplicated(keep="last")].sort_index()
                    _write_cache(new_cache.astype(float), path)
                    merged = pd.concat([merged, tail]) if len(merged) else tail
                    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
            out[sym] = merged[(merged.index >= start) & (merged.index <= now)].astype(float)
        return out

    FINE_WINDOW = pd.Timedelta(hours=40)       # one 5-minute request covers 500 points (41h40)
    REST_RETENTION = pd.Timedelta(days=28)     # the public endpoints keep about 30 days

    def metrics_history(self, symbols: list[str], start: pd.Timestamp, now: pd.Timestamp,
                        interval: str = "1h", fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Open interest and positioning on the bar grid, with the SAME alignment as research.

        Everything is first expressed as snapshots indexed by the time they become observable (see the
        convention note in ``market_data``): archive rows are shifted to their availability time, REST
        rows already carry it. Bar [T, T+D) then takes the last snapshot in ``(T, T+D]``, exactly what
        the backtest used. The recent window is requested at 5 minutes so the decision bar gets the
        freshest snapshot published; an older gap (first run after a data refresh) is bridged at 1 hour.
        """
        per_symbol: dict[str, pd.DataFrame] = {}
        warnings: list[str] = []
        ages: dict[str, float] = {}
        last_close = now.floor(interval_to_timedelta(interval))     # close of the bar being decided
        for sym in [s for s in symbols if not s.endswith(SPOT_SUFFIX)]:
            try:
                hist = self.store.read_metrics("um", sym, start)
            except FileNotFoundError:
                hist = pd.DataFrame()
            snapshots = metrics_available_at(hist) if len(hist) >= 2 else pd.DataFrame()
            path = self.cache_dir / "metric_snapshots" / f"{sym}.parquet"
            cached = _read_cache(path)
            frames = [f for f in (snapshots, cached) if len(f)]
            merged = pd.concat(frames) if frames else pd.DataFrame()
            if len(merged):
                merged.index = pd.DatetimeIndex(merged.index).as_unit("ns")
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            known_until = merged.index.max() if len(merged) else now - self.REST_RETENTION
            if now - known_until > self.REST_RETENTION:
                warnings.append(f"{sym}: metrics known until {known_until:%Y-%m-%d}, beyond the REST history; "
                                "the z-score window has a hole")
                known_until = now - self.REST_RETENTION
            fine_start = max(known_until, now - self.FINE_WINDOW)
            parts = []
            try:
                if known_until < fine_start:
                    parts.append(self.rest.metrics(sym, known_until, fine_start, period="1h", fields=fields))
                if now - fine_start >= pd.Timedelta(minutes=5):
                    parts.append(self.rest.metrics(sym, fine_start, now, period="5m", fields=fields))
            except (requests.HTTPError, RuntimeError) as exc:
                warnings.append(f"{sym}: metrics tail unavailable ({str(exc)[:80]})")
            parts = [p for p in parts if len(p)]
            if parts:
                tail = clean_metrics(pd.concat(parts).sort_index())
                tail = tail[~tail.index.duplicated(keep="last")]
                new_cache = pd.concat([cached, tail]) if len(cached) else tail
                new_cache.index = pd.DatetimeIndex(new_cache.index).as_unit("ns")
                new_cache = new_cache[~new_cache.index.duplicated(keep="last")].sort_index()
                new_cache = new_cache[new_cache.index >= start - pd.Timedelta(days=7)]   # bounded cache
                _write_cache(new_cache.astype("float32"), path, compression="zstd")
                merged = pd.concat([merged, tail]) if len(merged) else tail
                merged.index = pd.DatetimeIndex(merged.index).as_unit("ns")
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if len(merged):
                per_symbol[sym] = snapshots_to_bars(clean_metrics(merged[merged.index <= now]), interval)
                usable = merged.index[merged.index <= last_close]
                if len(usable):   # how old is the snapshot the decision bar ends up with (0 = taken at its close)
                    ages[sym] = float((last_close - usable.max()).total_seconds())
        self.last_metrics_warnings = warnings
        self.last_snapshot_age_seconds = ages
        if not per_symbol:
            return {}
        columns = METRIC_LAST_COLUMNS + METRIC_MEAN_COLUMNS
        return {col: pd.DataFrame({s: df[col] for s, df in per_symbol.items() if col in df})
                for col in columns}

    def quote_volume_24h(self) -> pd.Series:
        resp = requests.get(f"{FAPI_URL}/fapi/v1/ticker/24hr", timeout=self.timeout)
        resp.raise_for_status()
        return pd.Series({r["symbol"]: float(r["quoteVolume"]) for r in resp.json()})

    def funding_now(self) -> pd.Series:
        """Last funding rate per perpetual, per settlement interval (one request for the whole venue)."""
        resp = requests.get(f"{FAPI_URL}/fapi/v1/premiumIndex", timeout=self.timeout)
        resp.raise_for_status()
        return pd.Series({r["symbol"]: float(r["lastFundingRate"]) for r in resp.json()})

    def retail_positioning(self, symbols: list[str], now: pd.Timestamp, days: int = 25) -> pd.DataFrame:
        """Long/short ratio of retail accounts and its z-score over ``days``, from REST alone (4h points).

        Context for whoever reads the market, not the research signal: ``strategies/positioning.py``
        standardises over 90 days of stored history. Symbols the endpoint does not cover are left out.
        """
        rows = {}
        for sym in symbols:
            try:
                frame = clean_metrics(self.rest.metrics(sym, now - pd.Timedelta(days=days), now, period="4h",
                                                        fields=["global_account_ls"]))
            except (requests.HTTPError, RuntimeError):
                continue
            ratio = frame["global_account_ls"].dropna() if "global_account_ls" in frame else pd.Series(dtype=float)
            if len(ratio) >= 30 and ratio.std() > 0:
                rows[sym] = {"ratio": float(ratio.iloc[-1]),
                             "z": float((ratio.iloc[-1] - ratio.mean()) / ratio.std())}
        return pd.DataFrame.from_dict(rows, orient="index", columns=["ratio", "z"])
