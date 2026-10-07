"""Binance USDT-M futures broker.

Testnet only by design: ``testnet=False`` raises, because sending a real order needs the account
owner's explicit approval and a reviewed implementation (see ``brokers/live.py``).

What this adapter is for: exercising the parts a simulator cannot prove — exchange lot and notional
filters, deterministic client order ids surviving a restart, position and balance reconciliation,
funding settled by the venue rather than by our model, and the error paths of a real API.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from quant.brokers.base import Broker, BrokerError, DuplicateOrderError
from quant.execution.orders import Fill, Order, Side

TESTNET_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"


class CredentialsError(BrokerError):
    pass


def load_credentials(path: str | Path) -> tuple[str, str]:
    """Read {"api_key": ..., "api_secret": ...}; never log the contents of this file."""
    file = Path(path)
    if not file.exists():
        raise CredentialsError(f"no credentials file at {file}")
    if file.stat().st_mode & 0o077:
        raise CredentialsError(f"{file} is readable by others; run: chmod 600 {file}")
    raw = json.loads(file.read_text())
    try:
        return str(raw["api_key"]), str(raw["api_secret"])
    except KeyError as exc:
        raise CredentialsError(f"{file} must contain api_key and api_secret") from exc


class BinanceFuturesBroker(Broker):
    def __init__(self, credentials_path: str | Path, state_dir: str | Path, *, testnet: bool = True,
                 strategy_id: str = "", recv_window: int = 5000, timeout: float = 15.0,
                 fill_timeout: float = 20.0, session: requests.Session | None = None):
        if not testnet:
            raise BrokerError("mainnet trading is disabled in this adapter; use PaperBroker or get approval")
        self._key, self._secret = load_credentials(credentials_path)
        self._base = TESTNET_URL
        self._session = session or requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self._key})
        self.recv_window, self.timeout, self.fill_timeout = recv_window, timeout, fill_timeout
        self.strategy_id = strategy_id
        self._state = Path(state_dir)
        self._state.mkdir(parents=True, exist_ok=True)
        self._ids_file = self._state / "submitted_order_ids.json"
        self._order_ids: set[str] = set(json.loads(self._ids_file.read_text())) if self._ids_file.exists() else set()
        self._filters: dict[str, dict] = {}
        self._funding_cursor_file = self._state / "funding_cursor.json"

    # ---- plumbing -------------------------------------------------------------------------
    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = True) -> dict | list:
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window
            query = urlencode(params)
            params["signature"] = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        for attempt in range(4):
            response = self._session.request(method, self._base + path, params=params, timeout=self.timeout)
            if response.status_code in (418, 429) or response.status_code >= 500:
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise BrokerError(f"{method} {path} failed [{response.status_code}]: {response.text[:300]}")
            return response.json()
        raise BrokerError(f"{method} {path} kept failing (rate limited or server errors)")

    def _persist_ids(self) -> None:
        self._ids_file.write_text(json.dumps(sorted(self._order_ids)))

    def filters(self, symbol: str) -> dict:
        """Exchange lot/notional filters; a strategy quantity that ignores them is simply rejected."""
        if not self._filters:
            info = self._request("GET", "/fapi/v1/exchangeInfo", signed=False)
            for item in info["symbols"]:
                if item.get("status") not in (None, "TRADING"):
                    continue
                by_type = {f["filterType"]: f for f in item["filters"]}
                self._filters[item["symbol"]] = {
                    "step": float(by_type["LOT_SIZE"]["stepSize"]),
                    "min_qty": float(by_type["LOT_SIZE"]["minQty"]),
                    "tick": float(by_type["PRICE_FILTER"]["tickSize"]),
                    "min_notional": float(by_type.get("MIN_NOTIONAL", {}).get("notional", 0.0)),
                    "quantity_precision": int(item["quantityPrecision"]),
                }
        if symbol not in self._filters:
            raise BrokerError(f"{symbol} is not listed on this venue")
        return self._filters[symbol]

    def supported_symbols(self) -> set[str]:
        self.filters("BTCUSDT")          # warms the cache
        return set(self._filters)

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Floor to the exchange step.

        The epsilon is not cosmetic: 0.0006 / 0.0001 is 5.999999999999999 in binary floating point, so a
        plain floor turns an exact multiple into one step less — enough to drop an order below the
        venue's minimum notional and have it silently skipped.
        """
        spec = self.filters(symbol)
        steps = math.floor(abs(quantity) / spec["step"] + 1e-9)
        return round(steps * spec["step"], spec["quantity_precision"])

    # ---- Broker interface -----------------------------------------------------------------
    def positions(self) -> pd.Series:
        rows = self._request("GET", "/fapi/v2/positionRisk")
        held = {r["symbol"]: float(r["positionAmt"]) for r in rows if float(r["positionAmt"]) != 0.0}
        return pd.Series(held, dtype=float).sort_index()

    def cash(self) -> float:
        for row in self._request("GET", "/fapi/v2/balance"):
            if row["asset"] == "USDT":
                return float(row["balance"])
        return 0.0

    def equity(self, prices: pd.Series | None = None) -> float:
        """Margin balance as the venue computes it: wallet plus unrealised PnL."""
        account = self._request("GET", "/fapi/v2/account")
        return float(account["totalMarginBalance"])

    def known_order_ids(self) -> set[str]:
        return set(self._order_ids)

    def submit(self, order: Order, market_price: float, now: pd.Timestamp, bar_quote_volume: float,
               bar_sigma: float = 0.02, bars_per_day: float = 24.0,
               observed_half_spread_bps: float | None = None) -> Fill | None:
        if order.client_order_id in self._order_ids:
            raise DuplicateOrderError(order.client_order_id)
        spec = self.filters(order.symbol)
        qty = self.round_quantity(order.symbol, order.quantity)
        notional = qty * market_price
        # The venue waives the minimum notional for reduce-only orders; applying it anyway would leave
        # dust positions impossible to close (verified on the testnet: a 34 USDT close is accepted).
        too_small = qty < spec["min_qty"] or qty <= 0
        if not order.reduce_only:
            too_small = too_small or notional < spec["min_notional"]
        if too_small:
            self._order_ids.add(order.client_order_id)   # decided and closed: never retried
            self._persist_ids()
            return None
        params = {"symbol": order.symbol, "side": order.side.value, "type": "MARKET",
                  "quantity": f"{qty:.{spec['quantity_precision']}f}",
                  "newClientOrderId": order.client_order_id}
        if order.reduce_only:
            params["reduceOnly"] = "true"
        self._order_ids.add(order.client_order_id)       # recorded BEFORE sending: a crash must not duplicate
        self._persist_ids()
        placed = self._request("POST", "/fapi/v1/order", params)
        return self._await_fill(order, placed, now)

    def _await_fill(self, order: Order, placed: dict, now: pd.Timestamp) -> Fill | None:
        deadline = time.time() + self.fill_timeout
        state = placed
        while state.get("status") not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") and time.time() < deadline:
            time.sleep(0.4)
            state = self._request("GET", "/fapi/v1/order",
                                  {"symbol": order.symbol, "origClientOrderId": order.client_order_id})
        executed = float(state.get("executedQty", 0.0) or 0.0)
        if executed <= 0:
            return None
        price = float(state.get("avgPrice", 0.0) or 0.0) or order.reference_price
        fee = self._commission(order.symbol, int(state["orderId"]))
        return Fill(client_order_id=order.client_order_id, symbol=order.symbol, side=order.side,
                    quantity=executed, price=price, fee=fee,
                    fill_time=pd.Timestamp(int(state.get("updateTime", now.value // 10**6)), unit="ms", tz="UTC"),
                    reference_price=order.reference_price,
                    extra={"venue_order_id": state.get("orderId"), "status": state.get("status"),
                           "maker": False, "requested_qty": order.quantity})

    def _commission(self, symbol: str, order_id: int) -> float:
        trades = self._request("GET", "/fapi/v1/userTrades", {"symbol": symbol, "orderId": order_id})
        return sum(float(t["commission"]) for t in trades)

    settles_funding_itself = True

    def apply_funding(self, symbol: str, rate: float, mark_price: float, settle_time: pd.Timestamp) -> float:
        """Nothing to apply: the venue settles funding on the account itself."""
        return 0.0

    def funding_paid_since_last_check(self, now: pd.Timestamp) -> float:
        """Funding actually charged by the venue since the previous call (positive = paid).

        The cursor is persisted so that a restart neither re-counts nor skips a settlement; the first
        call only arms it (history before the loop existed is not ours to report).
        """
        now_ms = int(now.value // 10**6)
        if not self._funding_cursor_file.exists():
            self._funding_cursor_file.write_text(json.dumps({"since_ms": now_ms}))
            return 0.0
        since_ms = int(json.loads(self._funding_cursor_file.read_text())["since_ms"])
        rows = self._request("GET", "/fapi/v1/income",
                             {"incomeType": "FUNDING_FEE", "startTime": since_ms + 1, "endTime": now_ms, "limit": 1000})
        self._funding_cursor_file.write_text(json.dumps({"since_ms": now_ms}))
        return -sum(float(r["income"]) for r in rows)    # income < 0 means we paid
