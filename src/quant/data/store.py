"""Local parquet store: one file per (exchange, market, dataset, interval, symbol)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.data.timeframes import to_utc


@dataclass
class ParquetStore:
    root: Path
    exchange: str = "binance"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _path(self, market: str, dataset: str, symbol: str, interval: str | None = None) -> Path:
        parts = [self.exchange, market, dataset]
        if interval:
            parts.append(interval)
        return self.root.joinpath(*parts, f"{symbol}.parquet")

    @staticmethod
    def _write(df: pd.DataFrame, path: Path, compression: str = "snappy") -> None:
        if not df.index.is_monotonic_increasing:
            raise ValueError("refusing to store an unsorted frame")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, compression=compression)
        tmp.replace(path)  # atomic: a crash never leaves a half-written file

    @staticmethod
    def _read(path: Path, start=None, end=None, columns: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path, columns=columns)
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        if start is not None:
            df = df[df.index >= to_utc(start)]
        if end is not None:
            df = df[df.index < to_utc(end)]
        return df

    def write_klines(self, df: pd.DataFrame, market: str, symbol: str, interval: str) -> Path:
        path = self._path(market, "klines", symbol, interval)
        self._write(df, path)
        return path

    def read_klines(self, market: str, symbol: str, interval: str, start=None, end=None,
                    columns: list[str] | None = None) -> pd.DataFrame:
        return self._read(self._path(market, "klines", symbol, interval), start, end, columns)

    def write_funding(self, df: pd.DataFrame, market: str, symbol: str) -> Path:
        path = self._path(market, "funding", symbol)
        self._write(df, path)
        return path

    def read_funding(self, market: str, symbol: str, start=None, end=None) -> pd.DataFrame:
        return self._read(self._path(market, "funding", symbol), start, end)

    def has_funding(self, market: str, symbol: str) -> bool:
        return self._path(market, "funding", symbol).exists()

    def write_metrics(self, df: pd.DataFrame, market: str, symbol: str) -> Path:
        """Metrics are stored as float32 with zstd compression: they are bulky and only used as
        features (the raw archives can always be re-downloaded and re-verified)."""
        path = self._path(market, "metrics", symbol)
        self._write(df.astype("float32"), path, compression="zstd")
        return path

    def read_metrics(self, market: str, symbol: str, start=None, end=None, columns: list[str] | None = None) -> pd.DataFrame:
        return self._read(self._path(market, "metrics", symbol), start, end, columns)

    def has_metrics(self, market: str, symbol: str) -> bool:
        return self._path(market, "metrics", symbol).exists()

    def symbols(self, market: str, interval: str) -> list[str]:
        directory = self.root / self.exchange / market / "klines" / interval
        return sorted(p.stem for p in directory.glob("*.parquet"))
