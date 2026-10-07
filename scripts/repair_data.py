"""Repair stored klines from the Binance REST API and extend funding rates past the last monthly archive.

- Gaps up to --max-gap-days are filled only after REST bars adjacent to the gap match
  the stored Vision bars exactly; repaired rows get source=1.
- Funding after the last stored settlement is appended from REST (Vision publishes monthly).

Example:
    .venv/bin/python scripts/repair_data.py --interval 1h --all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.binance_rest import BinanceRestClient, missing_ranges, repair_gaps  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.validation import validate_klines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="every stored symbol for --interval")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--max-gap-days", type=float, default=45)
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    args = parser.parse_args()

    store = ParquetStore(Path(args.data_dir))
    client = BinanceRestClient()
    symbols = store.symbols("um", args.interval) if args.all else args.symbols
    logs, funding_rows = [], []
    for i, sym in enumerate(symbols, 1):
        df = store.read_klines("um", sym, args.interval)
        if missing_ranges(df, args.interval):
            fixed, sym_logs = repair_gaps(df, sym, args.interval, client, max_gap=pd.Timedelta(days=args.max_gap_days))
            logs += [asdict(entry) for entry in sym_logs]
            if len(fixed) > len(df):
                report = validate_klines(fixed, args.interval, sym)
                if report.conflicting_duplicates or report.ohlc_violations:
                    print(f"  {sym}: repaired series failed validation, not stored: {report.errors}")
                else:
                    store.write_klines(fixed, "um", sym, args.interval)
                    print(f"  {sym}: +{len(fixed) - len(df)} bars from REST")
        if not args.no_funding and store.has_funding("um", sym):
            fr = store.read_funding("um", sym)
            last = fr.index.max()
            recent_bar = df.index.max()
            if recent_bar - last > pd.Timedelta(hours=8):
                extra = client.funding(sym, last + pd.Timedelta(milliseconds=1), recent_bar + pd.Timedelta(hours=1))
                if not extra.empty:
                    merged = pd.concat([fr, extra])
                    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
                    store.write_funding(merged, "um", sym)
                    funding_rows.append({"symbol": sym, "added": len(extra), "until": extra.index.max()})
        if i % 50 == 0:
            print(f"--- {i}/{len(symbols)}", flush=True)
    out = ROOT / "var" / "reports" / f"repair_{args.interval}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(out, index=False)
    if logs:
        print("gap repair status:", pd.DataFrame(logs)["status"].value_counts().to_dict())
    print(f"funding extended for {len(funding_rows)} symbols; log {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
