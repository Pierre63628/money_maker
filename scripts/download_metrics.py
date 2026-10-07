"""Download Binance USDT-M ``metrics`` (5-minute open interest and positioning ratios).

Only daily archives exist for this dataset, so the download is request-heavy: days are bounded by
each symbol's actual trading range (from the stored klines) to avoid pointless 404s, and an existing
store is extended rather than re-downloaded.

Example:
    .venv/bin/python scripts/download_metrics.py --top-n 30 --start 2022-01-01 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.binance_vision import BinanceVisionClient  # noqa: E402
from quant.data.market_data import clean_metrics, resample_metrics  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.universe import UniverseRule, ever_eligible, point_in_time_eligibility  # noqa: E402
from quant.research.universe_data import cached_daily_quote_volume, crypto_symbols  # noqa: E402

_lock = threading.Lock()


def purge_raw(raw_dir: Path, symbol: str) -> int:
    """Delete the cached daily archives of a symbol once its parquet is written (they are large and
    can be re-downloaded and re-verified on demand)."""
    folder = raw_dir / "data" / "futures" / "um" / "daily" / "metrics" / symbol
    if not folder.exists():
        return 0
    removed = 0
    for path in folder.iterdir():
        path.unlink()
        removed += 1
    folder.rmdir()
    return removed


def process(client: BinanceVisionClient, store: ParquetStore, symbol: str, start: pd.Timestamp,
            end: pd.Timestamp, purge: bool = False, resample: str | None = "1h") -> dict:
    try:
        klines = store.read_klines("um", symbol, "1h", columns=["close"])
    except FileNotFoundError:
        return {"symbol": symbol, "status": "no_klines"}
    first, last = klines.index.min().normalize(), klines.index.max().normalize()
    lo, hi = max(start, first), min(end, last)
    existing = store.read_metrics("um", symbol) if store.has_metrics("um", symbol) else pd.DataFrame()
    # fetch what is missing on both sides of the stored range (backfilling an earlier window is common)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if len(existing):
        first_stored, last_stored = existing.index.min().normalize(), existing.index.max().normalize()
        if lo < first_stored:
            ranges.append((lo, first_stored - pd.Timedelta(days=1)))
        if hi > last_stored:
            ranges.append((last_stored + pd.Timedelta(days=1), hi))
    elif lo <= hi:
        ranges.append((lo, hi))
    if not ranges:
        return {"symbol": symbol, "status": "up_to_date", "rows": len(existing)}
    parts = [client.fetch_metrics(symbol, a, b) for a, b in ranges if a <= b]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return {"symbol": symbol, "status": "no_data", "rows": len(existing)}
    fetched = pd.concat(parts).sort_index()
    raw_rows = len(fetched)
    if resample:
        # 5-minute snapshots are 12x bigger than the bars any strategy uses; store them at bar resolution
        fetched = resample_metrics(clean_metrics(fetched), resample)
    merged = pd.concat([existing, fetched]) if len(existing) else fetched
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    store.write_metrics(merged, "um", symbol)
    if purge:
        purge_raw(Path(client.raw_dir), symbol)
    days = (merged.index.max() - merged.index.min()).days + 1
    per_day = 288 if not resample else int(pd.Timedelta(days=1) / pd.Timedelta(resample))
    expected = days * per_day
    return {"symbol": symbol, "status": "ok", "rows": len(merged), "raw_rows": raw_rows, "start": merged.index.min(),
            "end": merged.index.max(), "coverage": len(merged) / expected if expected else float("nan")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-n", type=int, default=30, help="symbols ever in this liquidity rank")
    parser.add_argument("--min-quote-volume", type=float, default=5e6)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--symbols", nargs="*", default=None, help="explicit symbols instead of the universe")
    parser.add_argument("--workers", type=int, default=8, help="symbols in parallel")
    parser.add_argument("--archive-workers", type=int, default=16, help="daily archives in parallel per symbol")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    parser.add_argument("--raw-dir", default=str(ROOT / "var" / "raw"))
    parser.add_argument("--purge-raw", action="store_true", help="delete cached daily archives after ingestion")
    parser.add_argument("--resample", default="1h", help="store metrics at this bar size ('' keeps 5-minute rows)")
    args = parser.parse_args()

    store = ParquetStore(Path(args.data_dir))
    client = BinanceVisionClient(raw_dir=Path(args.raw_dir), market="um", max_workers=args.archive_workers)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    if args.symbols:
        symbols = args.symbols
    else:
        qv = cached_daily_quote_volume(store, crypto_symbols(store))
        mask = point_in_time_eligibility(qv, UniverseRule(top_n=args.top_n, min_median_quote_volume=args.min_quote_volume))
        symbols = ever_eligible(mask, start, end)
    print(f"{len(symbols)} symbols, {start:%Y-%m-%d} -> {end:%Y-%m-%d}", flush=True)

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, client, store, s, start, end, args.purge_raw, args.resample or None): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append({"symbol": futures[fut], "status": f"failed: {exc}"})
            if i % 10 == 0:
                with _lock:
                    print(f"--- {i}/{len(symbols)} symbols, {time.time() - t0:.0f}s", flush=True)
    report = pd.DataFrame(rows)
    out = ROOT / "var" / "reports" / "metrics_coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print("status:", report["status"].str.split(":").str[0].value_counts().to_dict())
    if "coverage" in report:
        ok = report[report["status"] == "ok"]
        print(f"coverage median {ok['coverage'].median():.3f}, below 0.9: {(ok['coverage'] < 0.9).sum()} symbols")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
