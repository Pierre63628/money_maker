"""One-screen view of the testnet account: positions, unrealised PnL, and the last loop step."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.brokers.binance_futures import BinanceFuturesBroker  # noqa: E402

JOURNAL = ROOT / "var" / "paper" / "crowding_top50_4h-c0-shadow-testnet" / "events.jsonl"


def main() -> int:
    broker = BinanceFuturesBroker(ROOT / "var" / "secrets" / "binance_testnet.json",
                                  ROOT / "var" / "paper" / "crowding_top50_4h-c0-shadow-testnet" / "broker")
    rows = broker._request("GET", "/fapi/v2/positionRisk")
    held = [r for r in rows if float(r["positionAmt"]) != 0]
    frame = pd.DataFrame([{
        "symbole": r["symbol"],
        "sens": "LONG" if float(r["positionAmt"]) > 0 else "SHORT",
        "notionnel": abs(float(r["notional"])),
        "entrée": float(r["entryPrice"]),
        "marque": float(r["markPrice"]),
        "pnl": float(r["unRealizedProfit"]),
    } for r in held])
    equity = broker.equity()
    if frame.empty:
        print(f"aucune position | solde {equity:,.2f} USDT")
        return 0
    frame["pnl_bps"] = 1e4 * frame["pnl"] / frame["notionnel"]
    frame = frame.sort_values("pnl", ascending=False)
    print(frame.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    gross = frame["notionnel"].sum()
    net = sum(float(r["notional"]) for r in held)
    print(f"\n{len(frame)} positions | brut {gross:,.0f} USDT ({gross / equity:.0%} de l'équité) | "
          f"net {net:+,.0f} USDT | PnL latent {frame['pnl'].sum():+,.2f} USDT")
    print(f"solde (équité) {equity:,.2f} USDT")
    if JOURNAL.exists():
        steps = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
        steps = [e for e in steps if e.get("event") == "step_summary"]
        if steps:
            last = steps[-1]
            print(f"dernier pas: barre {last['decision_bar']}, {last['n_orders_filled']} ordres, "
                  f"{len(steps)} pas au total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
