"""Aligned multi-asset market data (wide frames: index = bar open time, columns = symbols)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from quant.data.timeframes import interval_ms, interval_to_timedelta, to_utc
from quant.data.validation import mask_stale_runs

FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


@dataclass(frozen=True)
class MarketData:
    """Bars on a regular grid. Missing bars / unlisted periods are NaN.

    Timing convention: a row stamped ``t`` is the bar [t, t + interval). Its
    close is known only at ``t + interval``. ``funding`` holds the funding rate
    settled at the bar open ``t`` (NaN when no settlement happens in that bar;
    several settlements inside one bar are summed).
    """

    interval: str
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    quote_volume: pd.DataFrame
    funding: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None  # bool: symbol tradable at bar t (point-in-time eligibility)
    taker_buy_quote_volume: pd.DataFrame | None = None  # aggressive-buy notional per bar (order flow)
    extras: dict | None = None  # extra aligned wide frames, e.g. open interest and positioning ratios

    def __post_init__(self) -> None:
        interval_ms(self.interval)
        ref = self.close
        for name in FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns):
                raise ValueError(f"field {name!r} is not aligned with close")
        if self.funding is not None and (not self.funding.index.equals(ref.index)
                                         or list(self.funding.columns) != list(ref.columns)):
            raise ValueError("funding is not aligned with close")
        for optional in ("universe", "taker_buy_quote_volume"):
            frame = getattr(self, optional)
            if frame is not None and (not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns)):
                raise ValueError(f"{optional} is not aligned with close")
        for name, frame in (self.extras or {}).items():
            if not frame.index.equals(ref.index) or list(frame.columns) != list(ref.columns):
                raise ValueError(f"extra {name!r} is not aligned with close")
        if not ref.index.is_monotonic_increasing or ref.index.has_duplicates:
            raise ValueError("index must be strictly increasing")

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    STALE_LIMIT_BARS = 24

    @property
    def tradable(self) -> pd.DataFrame:
        """Symbols a strategy may hold at bar t: priced within the last ``STALE_LIMIT_BARS`` bars (short gaps
        tolerated, delisted/ended series excluded) and inside the universe mask if one is set."""
        listed = self.close.ffill(limit=self.STALE_LIMIT_BARS).notna()
        return listed & self.universe if self.universe is not None else listed

    def extra(self, name: str) -> pd.DataFrame:
        """An aligned extra frame (raises if the data was not loaded)."""
        if not self.extras or name not in self.extras:
            raise ValueError(f"extra {name!r} not loaded; load the market data with it")
        return self.extras[name]

    def with_extras(self, **frames: pd.DataFrame) -> "MarketData":
        aligned = {k: v.reindex(index=self.index, columns=self.symbols).astype(float) for k, v in frames.items()}
        return replace(self, extras={**(self.extras or {}), **aligned})

    def with_universe(self, mask: pd.DataFrame) -> "MarketData":
        aligned = mask.reindex(index=self.index, columns=self.symbols).fillna(False).astype(bool)
        return replace(self, universe=aligned)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.close)

    def _map(self, fn) -> "MarketData":
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        for name in (*FIELDS, "funding", "universe", "taker_buy_quote_volume"):
            if kwargs[name] is not None:
                kwargs[name] = fn(kwargs[name])
        if kwargs["extras"]:
            kwargs["extras"] = {k: fn(v) for k, v in kwargs["extras"].items()}
        return replace(self, **{k: v for k, v in kwargs.items() if k != "interval"})

    def head(self, n: int) -> "MarketData":
        """First ``n`` bars. Used by look-ahead tests: a causal strategy must give identical output."""
        return self._map(lambda f: f.iloc[:n])

    def slice(self, start=None, end=None) -> "MarketData":
        """Bars with ``start <= t < end``."""
        idx = self.index
        mask = np.ones(len(idx), dtype=bool)
        if start is not None:
            mask &= idx >= to_utc(start)
        if end is not None:
            mask &= idx < to_utc(end)
        return self._map(lambda f: f.loc[mask])

    def select(self, symbols: list[str]) -> "MarketData":
        return self._map(lambda f: f[list(symbols)])

    @classmethod
    def from_frames(cls, bars: dict[str, pd.DataFrame], interval: str,
                    funding: dict[str, pd.DataFrame] | None = None) -> "MarketData":
        """Build from per-symbol canonical kline frames (and optional funding frames)."""
        if not bars:
            raise ValueError("no bars supplied")
        non_empty = [df for df in bars.values() if len(df)]
        if not non_empty:
            raise ValueError("all supplied bar frames are empty")
        start = min(df.index.min() for df in non_empty)
        end = max(df.index.max() for df in non_empty)
        index = pd.date_range(start, end, freq=interval_to_timedelta(interval), name="ts").as_unit("ns")
        gridded = {s: df[~df.index.duplicated(keep="first")].reindex(index) for s, df in bars.items()}
        wide = {name: pd.DataFrame({s: g[name] if name in g else np.nan for s, g in gridded.items()}, index=index)
                for name in FIELDS}
        if all("taker_buy_quote_volume" in g for g in gridded.values()):
            wide["taker_buy_quote_volume"] = pd.DataFrame({s: g["taker_buy_quote_volume"] for s, g in gridded.items()},
                                                          index=index)
        funding_wide = None
        if funding is not None:
            funding_wide = pd.DataFrame(np.nan, index=index, columns=list(bars))
            step = interval_to_timedelta(interval)
            for sym, fdf in funding.items():
                if sym not in funding_wide.columns or fdf is None or fdf.empty:
                    continue
                bar_ts = fdf.index.floor(step)
                summed = fdf["funding_rate"].groupby(bar_ts).sum(min_count=1)
                summed = summed[summed.index.isin(index)]
                funding_wide.loc[summed.index, sym] = summed.to_numpy()
        return cls(interval=interval, funding=funding_wide, **wide)


def resample_bars(df: pd.DataFrame, source_interval: str, target_interval: str) -> pd.DataFrame:
    """Aggregate canonical bars to a coarser interval. A target bar exists only if all its
    source bars exist (no partially observed bars)."""
    src, dst = interval_ms(source_interval), interval_ms(target_interval)
    if dst % src:
        raise ValueError(f"{target_interval} is not a multiple of {source_interval}")
    if dst == src:
        return df
    rule = interval_to_timedelta(target_interval)
    grouped = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": grouped["open"].first(),
        "high": grouped["high"].max(),
        "low": grouped["low"].min(),
        "close": grouped["close"].last(),
        "volume": grouped["volume"].sum(min_count=1),
        "quote_volume": grouped["quote_volume"].sum(min_count=1),
    })
    if "taker_buy_quote_volume" in df:
        out["taker_buy_quote_volume"] = grouped["taker_buy_quote_volume"].sum(min_count=1)
    complete = grouped["close"].count() == dst // src
    out = out[complete]
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "ts"
    return out


METRIC_LAST_COLUMNS = ["open_interest", "open_interest_value", "top_trader_account_ls", "top_trader_position_ls",
                       "global_account_ls"]
METRIC_MEAN_COLUMNS = ["taker_buy_sell_ratio"]


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Non-positive open interest or ratios are corrupt rows in the published metrics: mask them.

    (Binance publishes occasional zero open-interest snapshots; a zero would make percentage changes
    infinite and fabricate signals.)
    """
    if df.empty:
        return df
    out = df.copy()
    for col in METRIC_LAST_COLUMNS + METRIC_MEAN_COLUMNS:
        if col in out:
            out[col] = out[col].where(out[col] > 0)
    return out


# --- timestamp conventions of the positioning data (verified against the live API, 2026-10-06) -------
# The REST endpoints stamp a point with the END of its period: the value stamped ``tau`` is the
# snapshot observable at ``tau`` (Binance docs: "End time of the period"; a 1h point equals the 5m
# point carrying the same stamp).
# The Binance Vision archive stamps the same value with the START of its 5-minute period:
# archive ``create_time = t``  <=>  REST ``timestamp = t + 5min``.
# Research therefore gives bar [T, T+D) the snapshot observable at T+D (archive row T+D-5min), i.e.
# what is known at the bar close. Anything built from REST must bin on ``(T, T+D]`` to match;
# binning REST stamps on ``[T, T+D)`` silently feeds the strategy data that is one bar older.


def metrics_available_at(df: pd.DataFrame, step: pd.Timedelta | None = None) -> pd.DataFrame:
    """Archive rows (stamped at the start of their period) re-indexed by the time they become observable.

    ``step`` is the archive resolution (5 minutes natively, 1 hour once resampled for storage); it is
    inferred from the index spacing when omitted. A stored hourly row ``h`` holds the last 5-minute
    row of that hour, observable at ``h + 1h`` -- the same rule.
    """
    if df.empty:
        return df
    if step is None:
        if len(df) < 2:
            raise ValueError("cannot infer the archive resolution from fewer than two rows")
        step = pd.Series(df.index).diff().median()
    out = df.copy()
    out.index = pd.DatetimeIndex(df.index + step).as_unit("ns")
    out.index.name = "ts"
    return out


def snapshots_to_bars(snapshots: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Snapshots indexed by availability time -> bars: bar [T, T+D) takes what is observable in
    ``(T, T+D]`` (last value for stock variables, mean for flow ratios). Mixed densities are fine."""
    if snapshots.empty:
        return snapshots
    grouped = snapshots.resample(interval_to_timedelta(interval), label="left", closed="right")
    out = pd.DataFrame(index=grouped.size().index)
    for col in METRIC_LAST_COLUMNS:
        if col in snapshots:
            out[col] = grouped[col].last()
    for col in METRIC_MEAN_COLUMNS:
        if col in snapshots:
            out[col] = grouped[col].mean()
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "ts"
    return out


def resample_metrics(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Archive-stamped metrics -> bars. Stock variables take the last row of the bar (observable at
    its close), flow ratios take the bar average. Equivalent to
    ``snapshots_to_bars(metrics_available_at(df), interval)``; see the convention note above."""
    if df.empty:
        return df
    grouped = df.resample(interval_to_timedelta(interval), label="left", closed="left")
    out = pd.DataFrame(index=grouped.size().index)
    for col in METRIC_LAST_COLUMNS:
        if col in df:
            out[col] = grouped[col].last()
    for col in METRIC_MEAN_COLUMNS:
        if col in df:
            out[col] = grouped[col].mean()
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "ts"
    return out


def load_metric_frames(store, symbols: list[str], interval: str, start=None, end=None,
                       market: str = "um") -> dict[str, pd.DataFrame]:
    """Wide frames (index bars, columns symbols) for every metrics column available."""
    per_symbol = {}
    for sym in symbols:
        if not store.has_metrics(market, sym):
            continue
        df = store.read_metrics(market, sym, start, end)
        if len(df):
            per_symbol[sym] = resample_metrics(clean_metrics(df), interval)
    if not per_symbol:
        return {}
    columns = METRIC_LAST_COLUMNS + METRIC_MEAN_COLUMNS
    return {col: pd.DataFrame({s: df[col] for s, df in per_symbol.items() if col in df})
            for col in columns}


def load_market_data(store, symbols: list[str], interval: str, start=None, end=None,
                     market: str = "um", with_funding: bool = True, source_interval: str | None = None,
                     with_order_flow: bool = False, with_metrics: bool = False) -> MarketData:
    """Load bars for ``symbols``; with ``source_interval`` the bars are resampled from that stored interval."""
    src = source_interval or interval
    bars = {}
    columns = list(FIELDS) + (["taker_buy_quote_volume"] if with_order_flow else [])
    for s in symbols:
        raw = store.read_klines(market, s, src, start, end, columns=columns)
        cleaned, _ = mask_stale_runs(raw)
        bars[s] = resample_bars(cleaned.dropna(subset=["close"]), src, interval)
    funding = None
    if with_funding and market == "um":
        funding = {s: store.read_funding(market, s, start, end) for s in symbols if store.has_funding(market, s)}
    data = MarketData.from_frames(bars, interval, funding)
    if with_metrics:
        frames = load_metric_frames(store, symbols, interval, start, end, market)
        if frames:
            data = data.with_extras(**frames)
    return data
