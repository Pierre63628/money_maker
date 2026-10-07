"""Check the Binance futures testnet connection, then optionally place one throwaway order.

    .venv/bin/python scripts/testnet_check.py                      # read-only
    .venv/bin/python scripts/testnet_check.py --place-test-order   # one small round trip, fake money

Credentials live in a file the script never prints:
    mkdir -p var/secrets && chmod 700 var/secrets
    cat > var/secrets/binance_testnet.json  # {"api_key": "...", "api_secret": "..."}
    chmod 600 var/secrets/binance_testnet.json
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

CHECK_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--credentials", default=str(ROOT / "var" / "secrets" / "binance_testnet.json"))
    parser.add_argument("--state-dir", default=str(ROOT / "var" / "testnet" / "state"))
    parser.add_argument("--place-test-order", action="store_true",
                        help="buy then flatten a minimum-size BTCUSDT position (testnet money)")
    args = parser.parse_args()

    broker = BinanceFuturesBroker(args.credentials, args.state_dir, strategy_id="testnet_check")
    print(f"connected to the testnet; margin balance {broker.equity():,.2f} USDT, free cash {broker.cash():,.2f}")
    positions = broker.positions()
    print(f"open positions: {positions.to_dict() if len(positions) else 'none'}")
    print("\nexchange filters (what the venue will actually accept):")
    for symbol in CHECK_SYMBOLS:
        try:
            spec = broker.filters(symbol)
        except Exception as exc:                      # noqa: BLE001 - report and continue
            print(f"  {symbol}: unavailable ({exc})")
            continue
        print(f"  {symbol}: step {spec['step']}, min qty {spec['min_qty']}, "
              f"min notional {spec['min_notional']} USDT, tick {spec['tick']}")

    if not args.place_test_order:
        print("\nread-only check done. Re-run with --place-test-order to exercise the order path.")
        return 0

    symbol = "BTCUSDT"
    spec = broker.filters(symbol)
    mark = float(broker._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol}, signed=False)["markPrice"])
    qty = max(spec["min_qty"], broker.round_quantity(symbol, (spec["min_notional"] * 1.1) / mark))
    now = pd.Timestamp.now(tz="UTC")
    order = Order(client_order_id("testnet_check", symbol, now, qty), symbol, Side.BUY, qty, now, mark)
    print(f"\nbuying {qty} {symbol} at about {mark:,.1f} ({qty * mark:,.0f} USDT)...")
    fill = broker.submit(order, mark, now, bar_quote_volume=1e9)
    print(f"  fill: {fill.quantity} @ {fill.price:,.2f}, fee {fill.fee:.4f} USDT" if fill else "  no fill")
    if fill:
        closing = Order(client_order_id("testnet_check_close", symbol, now, qty), symbol, Side.SELL,
                        fill.quantity, now, fill.price, reduce_only=True)
        closed = broker.submit(closing, fill.price, pd.Timestamp.now(tz="UTC"), bar_quote_volume=1e9)
        print(f"  flattened: {closed.quantity} @ {closed.price:,.2f}" if closed else "  WARNING: still open")
    print(f"\npositions now: {broker.positions().to_dict() or 'flat'}; balance {broker.equity():,.2f} USDT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
