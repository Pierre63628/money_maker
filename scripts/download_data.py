"""Download, verify, validate and store Binance USDT-M klines (+ funding rates).

Examples:
    .venv/bin/python scripts/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --intervals 1h 15m --start 2020-01-01
    .venv/bin/python scripts/download_data.py --universe crypto --intervals 1h --workers 4 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.binance_vision import BinanceVisionClient  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.universe import classify_symbols, fetch_exchange_info  # noqa: E402
from quant.data.validation import validate_klines  # noqa: E402

_print_lock = threading.Lock()


def process_symbol(client, store, symbol, intervals, market, start, end, funding, skip_existing,
                   extend=False) -> list[dict]:
    """``extend``: append what was published since the stored series ends (weekly refresh).
    ``skip_existing``: leave any stored symbol untouched (resume an interrupted bulk download)."""
    rows = []
    for interval in intervals:
        path = store._path(market, "klines", symbol, interval)
        existing, fetch_start = None, start
        if path.exists():
            if extend:
                existing = store.read_klines(market, symbol, interval)
                if len(existing):   # re-read the last stored day: it may have been published incomplete
                    fetch_start = existing.index.max().normalize()
            elif skip_existing:
                rows.append({"symbol": symbol, "interval": interval, "status": "skipped_existing"})
                continue
        try:
            df = client.fetch_klines(symbol, interval, fetch_start, end)
        except Exception as exc:
            rows.append({"symbol": symbol, "interval": interval, "status": f"download_failed: {exc}"})
            continue
        if existing is not None and len(existing):
            if df.empty or df.index.max() <= existing.index.max():
                rows.append({"symbol": symbol, "interval": interval, "status": "up_to_date"})
                continue
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.empty:
            rows.append({"symbol": symbol, "interval": interval, "status": "no_data"})
            continue
        report = validate_klines(df, interval, symbol)
        store.write_klines(df, market, symbol, interval)
        rows.append({"symbol": symbol, "interval": interval, "status": "ok" if report.is_usable else "unusable",
                     "n_bars": report.n_bars, "start": report.start, "end": report.end,
                     "missing_bars": report.missing_bars, "missing_ratio": report.missing_ratio,
                     "largest_gap_hours": report.largest_gap.total_seconds() / 3600,
                     "ohlc_violations": report.ohlc_violations, "conflicting_duplicates": report.conflicting_duplicates,
                     "zero_volume_bars": report.zero_volume_bars, "extreme_moves": report.extreme_moves,
                     "errors": "; ".join(report.errors)})
        with _print_lock:
            print(report.summary(), flush=True)
    has_funding = store.has_funding(market, symbol)
    if market == "um" and funding and not (skip_existing and has_funding and not extend):
        try:
            stored = store.read_funding(market, symbol) if (extend and has_funding) else None
            funding_start = stored.index.max().normalize().replace(day=1) if stored is not None and len(stored) else start
            fdf = client.fetch_funding(symbol, funding_start, end)
            if stored is not None and len(stored) and not fdf.empty:
                fdf = pd.concat([stored, fdf])
                fdf = fdf[~fdf.index.duplicated(keep="last")].sort_index()
            if not fdf.empty:
                store.write_funding(fdf, market, symbol)
        except Exception as exc:
            rows.append({"symbol": symbol, "interval": "funding", "status": f"download_failed: {exc}"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--universe", choices=["crypto", "all"], default=None,
                        help="every USDT perpetual in the bucket (incl. delisted); 'crypto' excludes TradFi/index/stables")
    parser.add_argument("--intervals", nargs="+", default=["1h"])
    parser.add_argument("--market", default="um", choices=["um", "spot"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave stored symbols untouched (resume a bulk download)")
    parser.add_argument("--extend", action="store_true",
                        help="append what was published since each stored series ends (periodic refresh)")
    parser.add_argument("--workers", type=int, default=1, help="symbols processed in parallel")
    parser.add_argument("--archive-workers", type=int, default=8, help="archives downloaded in parallel per symbol")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    parser.add_argument("--raw-dir", default=str(ROOT / "var" / "raw"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = Path(args.data_dir)
    client = BinanceVisionClient(raw_dir=Path(args.raw_dir), market=args.market, max_workers=args.archive_workers)
    store = ParquetStore(data_dir)
    if args.universe:
        info = fetch_exchange_info(data_dir / "binance" / args.market / "exchange_info.json")
        classes = classify_symbols(client.list_symbols(), info)
        classes.to_csv(data_dir / "binance" / args.market / "symbol_classification.csv")
        symbols = list(classes.index[classes["is_crypto"]]) if args.universe == "crypto" else list(classes.index)
        print(f"universe={args.universe}: {len(symbols)} symbols "
              f"({(~classes['is_crypto']).sum()} non-crypto excluded)" if args.universe == "crypto" else "")
    else:
        symbols = args.symbols

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_symbol, client, store, s, args.intervals, args.market, args.start, args.end,
                               not args.no_funding, args.skip_existing, args.extend): s for s in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.extend(fut.result())
            if i % 25 == 0:
                with _print_lock:
                    print(f"--- progress {i}/{len(symbols)} symbols", flush=True)

    report = pd.DataFrame(rows)
    out = ROOT / "var" / "reports" / f"data_quality_{args.market}_{'_'.join(args.intervals)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and args.skip_existing:
        previous = pd.read_csv(out)
        report = pd.concat([previous, report[report["status"] != "skipped_existing"]]).drop_duplicates(
            subset=["symbol", "interval"], keep="last")
    report.to_csv(out, index=False)
    print("\nstatus counts:", report["status"].str.split(":").str[0].value_counts().to_dict())
    print(f"quality report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
