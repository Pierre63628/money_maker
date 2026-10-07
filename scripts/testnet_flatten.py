"""Close every open position on the Binance futures testnet (reduce-only market orders).

    .venv/bin/python scripts/testnet_flatten.py

Useful after a manual test, or as a kill-switch rehearsal: the same sequence is what a live account
would need in an emergency, so it is worth having as a script rather than an improvised snippet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.brokers.binance_futures import BinanceFuturesBroker  # noqa: E402
from quant.execution.orders import Order, Side, client_order_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--credentials", default=str(ROOT / "var" / "secrets" / "binance_testnet.json"))
    parser.add_argument("--state-dir", default=str(ROOT / "var" / "testnet" / "flatten"))
    args = parser.parse_args()

    broker = BinanceFuturesBroker(args.credentials, args.state_dir, strategy_id="flatten")
    positions = broker.positions()
    if positions.empty:
        print("nothing to close")
        return 0
    print(f"closing {len(positions)} positions (balance {broker.equity():,.2f} USDT)")
    failures = []
    for symbol, amount in positions.items():
        now = pd.Timestamp.now(tz="UTC")
        mark = float(broker._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol}, signed=False)["markPrice"])
        side = Side.SELL if amount > 0 else Side.BUY
        order = Order(client_order_id("flatten", symbol, now, amount), symbol, side, abs(amount), now, mark,
                      reduce_only=True)
        fill = broker.submit(order, mark, now, bar_quote_volume=1e9)
        if fill:
            print(f"  {symbol:<14} {side.value} {abs(amount):>12,.4f} @ {fill.price:,.6g}  fee {fill.fee:.4f}")
        else:
            failures.append(symbol)
            print(f"  {symbol:<14} NOT CLOSED")
    remaining = broker.positions()
    print(f"\nremaining: {remaining.to_dict() if len(remaining) else 'flat'} | balance {broker.equity():,.2f} USDT")
    return 1 if failures or len(remaining) else 0


if __name__ == "__main__":
    raise SystemExit(main())
