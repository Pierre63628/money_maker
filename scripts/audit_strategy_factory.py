"""Audit Strategy Factory leaderboard strategies for intrabar fill artifacts.

For each strategy: fetch its public trade list, fetch the venue's OHLC bars
(Bybit linear perps for BYBIT:*.P markets), and measure where exits land inside
the exit bar. Reported metrics are claims; this script tries to falsify them.

Example:
    .venv/bin/python scripts/audit_strategy_factory.py --sort sharpe --min-trades 300 --max-dd 40 --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.bybit import fetch_bybit_klines  # noqa: E402
from quant.evaluation.fill_audit import audit_intrabar_fills  # noqa: E402
from quant.integrations.strategy_factory import StrategyFactoryClient  # noqa: E402

SF_TIMEFRAMES = {"1": "1m", "3": "3m", "5": "5m", "5m": "5m", "15": "15m", "15m": "15m", "30": "30m", "30m": "30m",
                 "60": "1h", "1h": "1h", "120": "2h", "240": "4h", "4h": "4h", "1D": "1d"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sort", default="sharpe")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--max-dd", type=float, default=40.0)
    parser.add_argument("--min-days", type=float, default=180.0, help="skip backtests shorter than this")
    parser.add_argument("--max-audit-trades", type=int, default=400, help="most recent trades audited per strategy")
    parser.add_argument("--out", default=str(ROOT / "var" / "reports" / "strategy_factory_audit.csv"))
    args = parser.parse_args()

    client = StrategyFactoryClient()
    print("platform:", client.stats())
    candidates = client.search(sort=args.sort, limit=min(args.limit * 3, 60), minTrades=args.min_trades,
                               maxDrawdownPct=args.max_dd)
    seen_results: set[tuple] = set()
    rows = []
    for c in candidates:
        r = c.get("result")
        if not r or len(rows) >= args.limit:
            continue
        days = (r["toTs"] - r["fromTs"]) / 86_400_000
        key = (c["symbol"], c["timeframe"], round(r["netProfitPct"], 1), r["totalTrades"])
        if days < args.min_days or key in seen_results:
            continue
        seen_results.add(key)
        summary, trades = client.trades(r["resultId"])
        interval = SF_TIMEFRAMES.get(str(summary["timeframe"]))
        venue = summary.get("displaySymbol", "")
        row = {"name": c["name"][:50], "symbol": summary["symbol"], "tf": interval, "venue": venue,
               "days": round(days), "net_profit_pct": r["netProfitPct"], "pf": r["profitFactor"],
               "max_dd_pct": r["maxDrawdownPct"], "sharpe": r["sharpeRatio"], "trades": r["totalTrades"],
               "avg_bars_in_trade": summary.get("avgBarsInTrade"), "commission_pct": summary.get("parityProfile", {}).get("commissionValue"),
               "result_id": r["resultId"]}
        if not venue.startswith("BYBIT:") or interval is None or trades.empty:
            row["verdict"] = "SKIPPED_UNSUPPORTED_VENUE"
            rows.append(row)
            continue
        sample = trades.tail(args.max_audit_trades)
        bar = pd.Timedelta(interval.replace("m", "min"))
        bars = fetch_bybit_klines(summary["symbol"], interval, sample["entry_time"].min(), sample["exit_time"].max() + bar)
        rep = audit_intrabar_fills(sample, bars)
        row.update({k: rep.as_dict()[k] for k in ("n_matched", "exit_location_median", "share_exit_best_5pct",
                                                   "share_exit_outside_range", "verdict")})
        rows.append(row)
        print(f"{row['verdict']:<26} {row['symbol']:<10} {interval:<4} {days:>5.0f}d pnl={r['netProfitPct']:>9.0f}% "
              f"PF={r['profitFactor'] or 0:5.2f} DD={r['maxDrawdownPct']:5.1f}% best5%={row.get('share_exit_best_5pct', float('nan')):.2f} "
              f"{c['name'][:40]}")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if len(df) and "verdict" in df:
        print("\nverdicts:", df["verdict"].value_counts().to_dict())
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
