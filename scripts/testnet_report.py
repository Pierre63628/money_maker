"""Where the Binance demo (testnet) account stands, from the venue's own records.

    .venv/bin/python scripts/testnet_report.py

Everything comes from the exchange (balances, income ledger, positions), not from our journals: it
is the account as Binance sees it. The loop's journal is only read for the last decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.brokers.binance_futures import BinanceFuturesBroker  # noqa: E402

JOURNAL = ROOT / "var" / "paper" / "crowding_top50_4h-c0-shadow-testnet" / "events.jsonl"


def income_ledger(broker: BinanceFuturesBroker, since: pd.Timestamp) -> pd.DataFrame:
    rows, start = [], int(since.value // 10**6)
    while True:
        page = broker._request("GET", "/fapi/v1/income", {"startTime": start, "limit": 1000})
        if not page:
            break
        rows += page
        if len(page) < 1000:
            break
        start = int(page[-1]["time"]) + 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["income"] = frame["income"].astype(float)
    frame["ts"] = pd.to_datetime(frame["time"].astype("int64"), unit="ms", utc=True)
    return frame.drop_duplicates(subset=["tranId", "incomeType", "symbol", "time"]).sort_values("ts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-10-05", help="account inception (UTC)")
    parser.add_argument("--credentials", default=str(ROOT / "var" / "secrets" / "binance_testnet.json"))
    args = parser.parse_args()

    broker = BinanceFuturesBroker(args.credentials, ROOT / "var" / "testnet" / "state")
    now = pd.Timestamp.now(tz="UTC")
    account = broker._request("GET", "/fapi/v2/account")
    wallet, latent = float(account["totalWalletBalance"]), float(account["totalUnrealizedProfit"])
    equity = float(account["totalMarginBalance"])

    ledger = income_ledger(broker, pd.Timestamp(args.since, tz="UTC"))
    deposits = float(ledger.loc[ledger.incomeType == "TRANSFER", "income"].sum()) if len(ledger) else 0.0
    flows = ledger[ledger.incomeType != "TRANSFER"] if len(ledger) else ledger
    by_type = flows.groupby("incomeType").income.sum() if len(flows) else pd.Series(dtype=float)

    print(f"COMPTE DÉMO BINANCE, {now:%d/%m/%Y %H:%M} UTC")
    print(f"  dotation {deposits:,.2f} | solde encaissé {wallet:,.2f} | latent {latent:+,.2f} | équité {equity:,.2f} USDT")
    if deposits:
        print(f"  résultat depuis l'ouverture : {equity - deposits:+,.2f} USDT ({equity / deposits - 1:+.2%})")

    print("\nD'OÙ VIENT LE RÉSULTAT")
    realised, fees = by_type.get("REALIZED_PNL", 0.0), by_type.get("COMMISSION", 0.0)
    funding = by_type.get("FUNDING_FEE", 0.0)
    for label, value in (("gains et pertes réalisés", realised), ("frais de transaction", fees),
                         ("funding", funding), ("latent des positions ouvertes", latent)):
        print(f"  {label:<32} {value:+10.2f}")
    print(f"  {'total':<32} {realised + fees + funding + latent:+10.2f}")

    if len(flows):
        daily = flows.assign(jour=flows.ts.dt.strftime("%d/%m")).pivot_table(
            index="jour", columns="incomeType", values="income", aggfunc="sum").fillna(0.0)
        daily["encaissé"] = daily.sum(axis=1)
        daily["ordres"] = flows[flows.incomeType == "COMMISSION"].assign(
            jour=lambda d: d.ts.dt.strftime("%d/%m")).groupby("jour").size()
        print("\nPAR JOUR (encaissé : réalisé + frais + funding, hors latent)")
        print(daily.rename(columns={"COMMISSION": "frais", "FUNDING_FEE": "funding", "REALIZED_PNL": "réalisé"})
              .fillna(0).round(2).to_string())

    held = [r for r in broker._request("GET", "/fapi/v2/positionRisk") if float(r["positionAmt"]) != 0]
    if held:
        book = pd.DataFrame([{
            "symbole": r["symbol"], "sens": "LONG" if float(r["positionAmt"]) > 0 else "SHORT",
            "notionnel": abs(float(r["notional"])), "entrée": float(r["entryPrice"]),
            "marque": float(r["markPrice"]), "pnl": float(r["unRealizedProfit"])} for r in held])
        book["pct"] = 100 * book.pnl / book.notionnel
        book = book.sort_values("pnl", ascending=False)
        print(f"\n{len(book)} POSITIONS OUVERTES")
        print(book.to_string(index=False, formatters={
            "notionnel": "{:,.0f}".format, "entrée": "{:,.5g}".format, "marque": "{:,.5g}".format,
            "pnl": "{:+.2f}".format, "pct": "{:+.2f}%".format}))
        longs, shorts = book[book.sens == "LONG"], book[book.sens == "SHORT"]
        net = longs.notionnel.sum() - shorts.notionnel.sum()
        print(f"\n  longs  : {len(longs):>2} positions, {longs.notionnel.sum():>7,.0f} USDT, latent {longs.pnl.sum():+8.2f}")
        print(f"  shorts : {len(shorts):>2} positions, {shorts.notionnel.sum():>7,.0f} USDT, latent {shorts.pnl.sum():+8.2f}")
        print(f"  exposition brute {book.notionnel.sum() / equity:.0%} de l'équité, nette {net:+,.0f} USDT ({net / equity:+.1%})")

    if len(flows) and (flows.incomeType == "REALIZED_PNL").any():
        closed = flows[flows.incomeType == "REALIZED_PNL"].groupby("symbol").income.sum().sort_values()
        print("\nRÉALISÉ PAR SYMBOLE (positions soldées ou réduites)")
        print("  meilleurs : " + ", ".join(f"{s} {v:+.2f}" for s, v in closed.tail(4)[::-1].items()))
        print("  pires     : " + ", ".join(f"{s} {v:+.2f}" for s, v in closed.head(4).items()))

    if JOURNAL.exists():
        steps = [json.loads(line) for line in JOURNAL.read_text().splitlines() if '"step_summary"' in line]
        if steps:
            rebalances = [s for s in steps if s.get("rebalanced")]
            print(f"\nBOUCLE : {len(steps)} pas, {len(rebalances)} rebalancements, "
                  f"{sum(s['n_orders_filled'] for s in steps)} ordres ; dernier pas {steps[-1]['decision_bar'][:16]} UTC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
