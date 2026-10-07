"""Bar interval helpers. Crypto trades 24/7, so a year is 365 days of bars."""

from __future__ import annotations

import pandas as pd

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

MS_PER_YEAR = 365 * 86_400_000


def interval_ms(interval: str) -> int:
    try:
        return INTERVAL_MS[interval]
    except KeyError as exc:
        raise ValueError(f"unsupported interval {interval!r}; expected one of {sorted(INTERVAL_MS)}") from exc


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    return pd.Timedelta(milliseconds=interval_ms(interval))


def bars_per_year(interval: str) -> float:
    return MS_PER_YEAR / interval_ms(interval)


def bars_per_day(interval: str) -> float:
    return 86_400_000 / interval_ms(interval)


def to_utc(value) -> pd.Timestamp:
    """Coerce a string/datetime/Timestamp (naive = UTC) to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
