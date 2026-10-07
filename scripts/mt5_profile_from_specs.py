"""Turn an MT5 contract-specification dump into a broker profile this project can test.

Workflow:
1. open a DEMO account with the broker and install MetaTrader 5;
2. run ``scripts/mt5/DumpSymbolSpecs.mq5`` once (Navigator -> Scripts), which writes
   ``MQL5/Files/symbol_specs.csv``;
3. ``.venv/bin/python scripts/mt5_profile_from_specs.py symbol_specs.csv -o var/my_broker.json``;
4. ``.venv/bin/python scripts/broker_feasibility.py --profile var/my_broker.json``.

The conversions that matter:
- MT5 quotes a spread in *points*; the cost per side is half of it, expressed in basis points of price;
- swap is charged per lot per night in points, or as an annual percentage, depending on
  ``SYMBOL_SWAP_MODE``; both are converted to basis points of notional per day;
- the minimum tradable notional is ``volume_min x contract_size x price``, which decides the smallest
  account that can hold a diversified book.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.store import ParquetStore  # noqa: E402

# ENUM_SYMBOL_SWAP_MODE values that express swap as an annual percentage of the position value
PERCENT_SWAP_MODES = {5, 6}
POINT_SWAP_MODES = {1}
TRADE_MODE_FULL = 4
BASE_ALIASES = {"XBT": "BTC", "BCHABC": "BCH", "MIOTA": "IOTA"}


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    """CSV blanks arrive as NaN, which would silently poison every computed cost."""
    value = row.get(key, default)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(number) else number


def _text(row: pd.Series, key: str) -> str:
    value = row.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def base_currency(row: pd.Series) -> str:
    base = _text(row, "currency_base").upper()
    if not base:
        name = str(row["symbol"]).upper()
        for quote in ("USDT", "USD", "EUR", "USDC"):
            if quote in name:
                base = name.split(quote)[0]
                break
        base = "".join(ch for ch in base if ch.isalnum())
    return BASE_ALIASES.get(base, base)


def swap_bps_per_day(value: float, swap_mode: int, point: float, price: float) -> float:
    """Cost of carrying one side for a day, in basis points of notional (positive = you pay)."""
    if price <= 0 or value == 0:
        return 0.0
    if swap_mode in PERCENT_SWAP_MODES:
        return -value / 365.0 * 100.0      # annual percent -> bps/day, sign flipped so cost > 0
    if swap_mode in POINT_SWAP_MODES:
        return -value * point / price * 1e4
    return -value / price * 1e4            # deposit/margin currency per lot: approximate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs_csv")
    parser.add_argument("-o", "--out", default=str(ROOT / "var" / "broker_profile.json"))
    parser.add_argument("--name", default="mt5-demo")
    parser.add_argument("--commission-bps-per-side", type=float, default=0.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=10_000.0, help="planned account size, for the lot check")
    parser.add_argument("--max-weight", type=float, default=0.05, help="per-symbol weight cap of the strategy")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    args = parser.parse_args()

    raw = pd.read_csv(args.specs_csv, sep=";")
    raw.columns = [c.strip() for c in raw.columns]
    store = ParquetStore(Path(args.data_dir))
    known = set(store.symbols("um", "1h"))

    instruments, skipped = {}, []
    for _, row in raw.iterrows():
        symbol = _text(row, "symbol")
        if not symbol:
            continue
        bid, ask = _num(row, "bid"), _num(row, "ask")
        price = (bid + ask) / 2 if bid and ask else max(bid, ask)
        source = f"{base_currency(row)}USDT"
        if source not in known:
            skipped.append((symbol, f"no research data for {source}"))
            continue
        if int(_num(row, "trade_mode", TRADE_MODE_FULL)) != TRADE_MODE_FULL:
            skipped.append((symbol, "not fully tradable"))
            continue
        if price <= 0 or pd.isna(price):
            skipped.append((symbol, "no quote (open the symbol in Market Watch first)"))
            continue
        point = _num(row, "point")
        spread_points = _num(row, "spread_points")
        if spread_points <= 0 and ask > bid:
            spread_points = (ask - bid) / point if point else 0.0
        half_spread_bps = (spread_points * point) / 2 / price * 1e4
        swap_mode = int(_num(row, "swap_mode"))
        instruments[symbol] = {
            "source": source,
            "half_spread_bps": round(half_spread_bps, 2),
            "swap_long_bps_per_day": round(swap_bps_per_day(_num(row, "swap_long"), swap_mode, point, price), 3),
            "swap_short_bps_per_day": round(swap_bps_per_day(_num(row, "swap_short"), swap_mode, point, price), 3),
            "min_notional": round(_num(row, "volume_min") * _num(row, "contract_size", 1.0) * price, 2),
        }

    profile = {
        "name": args.name,
        "account_currency": "USD",
        "commission_bps_per_side": args.commission_bps_per_side,
        "slippage_bps": 2.0,
        "max_leverage": args.max_leverage,
        "notes": f"generated from {Path(args.specs_csv).name}; spreads are a single snapshot, not an average",
        "instruments": instruments,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(profile, indent=2))

    print(f"{len(instruments)} usable instruments written to {args.out}")
    if instruments:
        table = pd.DataFrame(instruments).T.sort_values("half_spread_bps")
        print(table.to_string())
        worst = table["half_spread_bps"].max()
        print(f"\nhalf spread: median {table['half_spread_bps'].median():.1f} bps, worst {worst:.1f} bps")
        need = table["min_notional"].max() / args.max_weight
        print(f"minimum account for a diversified book: ${need:,.0f} "
              f"(largest minimum ticket ${table['min_notional'].max():,.0f} at a {args.max_weight:.0%} cap)")
        if need > args.capital:
            print(f"  WARNING: planned capital ${args.capital:,.0f} is below that; some symbols could not be held")
    if skipped:
        reasons = pd.Series([r for _, r in skipped]).value_counts()
        print(f"\nskipped {len(skipped)} symbols: {reasons.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
