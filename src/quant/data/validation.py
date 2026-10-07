"""Data-quality checks for OHLCV bars.

Policy: report, never silently repair. The only automatic change is dropping
*exact* duplicate rows (done at ingestion). Missing bars stay missing (NaN after
reindexing) so the backtester cannot trade on prices that never existed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.data.timeframes import interval_ms, interval_to_timedelta, to_utc

PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass
class DataQualityReport:
    symbol: str
    interval: str
    n_bars: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    duplicate_timestamps: int = 0
    conflicting_duplicates: int = 0
    non_monotonic: int = 0
    misaligned_timestamps: int = 0
    missing_bars: int = 0
    missing_ratio: float = 0.0
    largest_gap: pd.Timedelta = pd.Timedelta(0)
    nan_prices: int = 0
    non_positive_prices: int = 0
    ohlc_violations: int = 0
    zero_volume_bars: int = 0
    stale_bars: int = 0
    extreme_moves: int = 0
    extreme_move_samples: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    max_missing_ratio: float = 0.02

    @property
    def errors(self) -> list[str]:
        """Issues that make the series unusable for backtesting until resolved."""
        out = []
        if self.n_bars == 0:
            out.append("empty series")
        if self.conflicting_duplicates:
            out.append(f"{self.conflicting_duplicates} conflicting duplicate timestamps")
        if self.non_monotonic:
            out.append(f"{self.non_monotonic} non-monotonic timestamps")
        if self.misaligned_timestamps:
            out.append(f"{self.misaligned_timestamps} timestamps off the {self.interval} grid")
        if self.non_positive_prices:
            out.append(f"{self.non_positive_prices} non-positive prices")
        if self.ohlc_violations:
            out.append(f"{self.ohlc_violations} OHLC consistency violations")
        if self.missing_ratio > self.max_missing_ratio:
            out.append(f"missing ratio {self.missing_ratio:.2%} > {self.max_missing_ratio:.2%}")
        return out

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.missing_bars:
            out.append(f"{self.missing_bars} missing bars (largest gap {self.largest_gap})")
        if self.nan_prices:
            out.append(f"{self.nan_prices} bars with NaN prices")
        if self.zero_volume_bars:
            out.append(f"{self.zero_volume_bars} zero-volume bars")
        if self.stale_bars:
            out.append(f"{self.stale_bars} stale bars (flat OHLC, zero volume)")
        if self.extreme_moves:
            out.append(f"{self.extreme_moves} extreme moves flagged")
        return out

    @property
    def is_usable(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "OK" if self.is_usable else "UNUSABLE"
        lines = [f"[{status}] {self.symbol} {self.interval}: {self.n_bars} bars {self.start} -> {self.end}"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  warn: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_klines(df: pd.DataFrame, interval: str, symbol: str = "",
                    max_missing_ratio: float = 0.02, extreme_threshold: float = 15.0) -> DataQualityReport:
    """Run all checks on a canonical kline frame (index = bar open time, UTC)."""
    step = interval_ms(interval)
    report = DataQualityReport(symbol=symbol, interval=interval, n_bars=len(df),
                               start=df.index.min() if len(df) else None,
                               end=df.index.max() if len(df) else None,
                               max_missing_ratio=max_missing_ratio)
    if df.empty:
        return report

    idx = pd.DatetimeIndex(df.index)
    ns = idx.as_unit("ns").asi8
    report.non_monotonic = int((np.diff(ns) < 0).sum())
    dup_mask = idx.duplicated(keep=False)
    report.duplicate_timestamps = int(idx.duplicated().sum())
    if dup_mask.any():
        dups = df[dup_mask]
        distinct_rows = dups.reset_index().drop_duplicates()
        report.conflicting_duplicates = int(distinct_rows.duplicated(subset=[distinct_rows.columns[0]]).sum())
    report.misaligned_timestamps = int(((ns // 1_000_000) % step != 0).sum())

    unique_sorted = np.unique(ns)
    expected = (unique_sorted[-1] - unique_sorted[0]) // (step * 1_000_000) + 1
    report.missing_bars = int(expected - len(unique_sorted))
    report.missing_ratio = report.missing_bars / expected if expected else 0.0
    if len(unique_sorted) > 1:
        report.largest_gap = pd.Timedelta(int(np.diff(unique_sorted).max()), unit="ns") - interval_to_timedelta(interval)

    prices = df[PRICE_COLUMNS]
    report.nan_prices = int(prices.isna().any(axis=1).sum())
    report.non_positive_prices = int((prices <= 0).any(axis=1).sum())
    high_ok = df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-12
    low_ok = df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-12
    valid = prices.notna().all(axis=1)
    report.ohlc_violations = int((valid & ~(high_ok & low_ok)).sum())

    if "volume" in df:
        zero_vol = df["volume"].fillna(0) <= 0
        report.zero_volume_bars = int(zero_vol.sum())
        flat = (df["high"] == df["low"]) & (df["open"] == df["close"]) & (df["open"] == df["high"])
        report.stale_bars = int((zero_vol & flat).sum())

    # Flag (not remove) moves far outside the robust distribution. QA only, so
    # using the full sample is fine here; signals must never do this.
    log_ret = np.log(df["close"]).diff().dropna()
    if len(log_ret) > 50:
        med = log_ret.median()
        mad = (log_ret - med).abs().median() * 1.4826
        if mad > 0:
            z = (log_ret - med).abs() / mad
            flagged = z[z > extreme_threshold]
            report.extreme_moves = int(len(flagged))
            top = flagged.sort_values(ascending=False).head(5)
            report.extreme_move_samples = [(ts, float(log_ret.loc[ts])) for ts in top.index]
    return report


def reindex_to_grid(df: pd.DataFrame, interval: str, start=None, end=None) -> pd.DataFrame:
    """Place bars on the regular interval grid. Missing bars become explicit NaN rows."""
    if df.empty:
        return df
    first = to_utc(start) if start is not None else df.index.min()
    last = to_utc(end) if end is not None else df.index.max()
    grid = pd.date_range(first, last, freq=interval_to_timedelta(interval), name="ts").as_unit("ns")
    return df[~df.index.duplicated(keep="first")].reindex(grid)


def mask_stale_runs(df: pd.DataFrame, min_run: int = 24) -> tuple[pd.DataFrame, int]:
    """Replace runs of >= ``min_run`` consecutive zero-volume, flat bars with NaN and drop a stale tail.

    Binance keeps publishing frozen bars (constant price, zero volume) after a contract is
    settled/delisted, sometimes for years. Those prices are not tradable and must not be
    used for signals, marks or fills. Short runs (exchange maintenance) are kept.
    Returns the cleaned frame and the number of masked bars.
    """
    if df.empty or "volume" not in df:
        return df, 0
    flat = (df["high"] == df["low"]) & (df["open"] == df["close"]) & (df["open"] == df["high"])
    stale = ((df["volume"].fillna(0) <= 0) & flat).to_numpy()
    if not stale.any():
        return df, 0
    run_id = np.cumsum(np.concatenate([[True], stale[1:] != stale[:-1]]))
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    to_mask = stale & (run_len >= min_run)
    if not to_mask.any():
        return df, 0
    out = df.copy()
    price_cols = [c for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote_volume"]
                  if c in out]
    out.loc[to_mask, price_cols] = np.nan
    valid = out["close"].notna().to_numpy()
    last_valid = np.flatnonzero(valid)
    if len(last_valid):
        out = out.iloc[: last_valid[-1] + 1]
    return out, int(to_mask.sum())


@dataclass
class MetricsQualityReport:
    symbol: str
    n_rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    expected_rows: int = 0
    coverage: float = 0.0
    non_positive: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.coverage < 0.95:
            out.append(f"coverage {self.coverage:.1%} of the 5-minute grid")
        for col, n in self.non_positive.items():
            if n:
                out.append(f"{n} non-positive {col} values (masked at load)")
        for col, n in self.missing.items():
            if n / max(self.n_rows, 1) > 0.05:
                out.append(f"{col} missing in {n / max(self.n_rows, 1):.0%} of rows")
        return out

    def summary(self) -> str:
        head = f"[metrics] {self.symbol}: {self.n_rows} rows {self.start} -> {self.end}, coverage {self.coverage:.1%}"
        return "\n".join([head] + [f"  warn: {w}" for w in self.warnings])


def validate_metrics(df: pd.DataFrame, symbol: str = "") -> MetricsQualityReport:
    """Quality of a 5-minute metrics series (open interest and positioning ratios)."""
    report = MetricsQualityReport(symbol=symbol, n_rows=len(df),
                                  start=df.index.min() if len(df) else None,
                                  end=df.index.max() if len(df) else None)
    if df.empty:
        return report
    span = (report.end - report.start).total_seconds() / 300 + 1
    report.expected_rows = int(span)
    report.coverage = len(df) / span
    report.non_positive = {c: int((df[c] <= 0).sum()) for c in df.columns}
    report.missing = {c: int(df[c].isna().sum()) for c in df.columns}
    return report
