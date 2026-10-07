from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant.brokers.base import BrokerError, DuplicateOrderError
from quant.brokers.binance_futures import BinanceFuturesBroker, CredentialsError, load_credentials
from quant.execution.orders import Order, Side

EXCHANGE_INFO = {"symbols": [{"symbol": "BTCUSDT", "quantityPrecision": 3, "filters": [
    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
    {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records every call so the tests can assert on signing and parameters."""

    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def request(self, method, url, params=None, timeout=None):
        path = url.split("binance.com")[-1] if "binance.com" in url else url.split(".com")[-1]
        self.calls.append((method, path, dict(params or {})))
        handler = self.routes[(method, path)]
        payload = handler(params) if callable(handler) else handler
        return FakeResponse(payload)


@pytest.fixture
def creds(tmp_path):
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"api_key": "KEY", "api_secret": "SECRET"}))
    path.chmod(0o600)
    return path


def make_broker(creds, tmp_path, routes):
    session = FakeSession(routes)
    broker = BinanceFuturesBroker(creds, tmp_path / "state", strategy_id="t", session=session, fill_timeout=1.0)
    return broker, session


def test_credentials_must_not_be_world_readable(tmp_path):
    path = tmp_path / "open.json"
    path.write_text(json.dumps({"api_key": "a", "api_secret": "b"}))
    path.chmod(0o644)
    with pytest.raises(CredentialsError, match="readable by others"):
        load_credentials(path)


def test_mainnet_is_refused(creds, tmp_path):
    with pytest.raises(BrokerError, match="mainnet"):
        BinanceFuturesBroker(creds, tmp_path, testnet=False)


def test_requests_are_signed_and_never_expose_the_secret(creds, tmp_path):
    broker, session = make_broker(creds, tmp_path, {("GET", "/fapi/v2/balance"): [{"asset": "USDT", "balance": "500"}]})
    assert broker.cash() == 500.0
    _, _, params = session.calls[-1]
    assert params["signature"] and "SECRET" not in params["signature"]
    assert session.headers["X-MBX-APIKEY"] == "KEY"


def test_order_quantity_is_rounded_to_the_exchange_step(creds, tmp_path):
    broker, _ = make_broker(creds, tmp_path, {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO})
    assert broker.round_quantity("BTCUSDT", 0.0127) == 0.012      # floored, never rounded up
    with pytest.raises(BrokerError, match="not listed"):
        broker.filters("NOPEUSDT")


def test_order_below_the_minimum_notional_is_skipped_once(creds, tmp_path):
    broker, session = make_broker(creds, tmp_path, {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO})
    order = Order("id-small", "BTCUSDT", Side.BUY, 0.001, pd.Timestamp("2026-10-05", tz="UTC"), 50_000.0)
    assert broker.submit(order, 50_000.0, pd.Timestamp.now(tz="UTC"), 1e7) is None   # 50 USDT < 100 minimum
    assert "id-small" in broker.known_order_ids()
    assert not [c for c in session.calls if c[0] == "POST"]
    with pytest.raises(DuplicateOrderError):
        broker.submit(order, 50_000.0, pd.Timestamp.now(tz="UTC"), 1e7)


def test_market_order_fills_and_reports_commission(creds, tmp_path):
    placed = {"orderId": 77, "status": "NEW", "executedQty": "0", "avgPrice": "0"}
    filled = {"orderId": 77, "status": "FILLED", "executedQty": "0.004", "avgPrice": "50100.0",
              "updateTime": 1790000000000}
    routes = {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO,
              ("POST", "/fapi/v1/order"): placed,
              ("GET", "/fapi/v1/order"): filled,
              ("GET", "/fapi/v1/userTrades"): [{"commission": "0.08"}, {"commission": "0.02"}]}
    broker, session = make_broker(creds, tmp_path, routes)
    order = Order("id-ok", "BTCUSDT", Side.BUY, 0.0041, pd.Timestamp("2026-10-05", tz="UTC"), 50_000.0)
    fill = broker.submit(order, 50_000.0, pd.Timestamp.now(tz="UTC"), 1e7)
    assert fill.quantity == 0.004 and fill.price == 50_100.0 and fill.fee == pytest.approx(0.10)
    assert fill.extra["venue_order_id"] == 77
    post = [c for c in session.calls if c[0] == "POST"][0][2]
    assert post["quantity"] == "0.004" and post["newClientOrderId"] == "id-ok" and post["type"] == "MARKET"


def test_submitted_ids_survive_a_restart(creds, tmp_path):
    routes = {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO,
              ("POST", "/fapi/v1/order"): {"orderId": 1, "status": "FILLED", "executedQty": "0.004",
                                           "avgPrice": "50000", "updateTime": 1790000000000},
              ("GET", "/fapi/v1/userTrades"): []}
    broker, _ = make_broker(creds, tmp_path, routes)
    order = Order("id-persist", "BTCUSDT", Side.BUY, 0.004, pd.Timestamp("2026-10-05", tz="UTC"), 50_000.0)
    broker.submit(order, 50_000.0, pd.Timestamp.now(tz="UTC"), 1e7)
    restarted, _ = make_broker(creds, tmp_path, routes)
    assert "id-persist" in restarted.known_order_ids()
    with pytest.raises(DuplicateOrderError):
        restarted.submit(order, 50_000.0, pd.Timestamp.now(tz="UTC"), 1e7)


def test_positions_and_equity_come_from_the_venue(creds, tmp_path):
    routes = {("GET", "/fapi/v2/positionRisk"): [{"symbol": "BTCUSDT", "positionAmt": "0.004"},
                                                 {"symbol": "ETHUSDT", "positionAmt": "0"},
                                                 {"symbol": "SOLUSDT", "positionAmt": "-3.0"}],
              ("GET", "/fapi/v2/account"): {"totalMarginBalance": "10123.45"}}
    broker, _ = make_broker(creds, tmp_path, routes)
    pos = broker.positions()
    assert list(pos.index) == ["BTCUSDT", "SOLUSDT"] and pos["SOLUSDT"] == -3.0
    assert broker.equity() == pytest.approx(10123.45)


def test_exact_multiples_are_not_floored_away(creds, tmp_path):
    """0.0006 / 0.0001 == 5.999999999999999 in binary: a naive floor loses a whole step."""
    broker, _ = make_broker(creds, tmp_path, {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO})
    assert broker.round_quantity("BTCUSDT", 0.006) == 0.006
    assert broker.round_quantity("BTCUSDT", 0.003) == 0.003
    assert broker.round_quantity("BTCUSDT", 0.029) == 0.029
    assert broker.round_quantity("BTCUSDT", 0.0129) == 0.012      # still floors genuine remainders


def test_an_order_at_the_minimum_notional_is_sent(creds, tmp_path):
    """Regression: rounding used to push an order just under the minimum, skipping it in silence."""
    routes = {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO,
              ("POST", "/fapi/v1/order"): {"orderId": 9, "status": "FILLED", "executedQty": "0.003",
                                           "avgPrice": "40000", "updateTime": 1790000000000},
              ("GET", "/fapi/v1/userTrades"): []}
    broker, session = make_broker(creds, tmp_path, routes)
    order = Order("id-edge", "BTCUSDT", Side.BUY, 0.003, pd.Timestamp("2026-10-05", tz="UTC"), 40_000.0)
    fill = broker.submit(order, 40_000.0, pd.Timestamp.now(tz="UTC"), 1e7)   # exactly 120 USDT > 100 minimum
    assert fill is not None and fill.quantity == 0.003
    assert [c for c in session.calls if c[0] == "POST"][0][2]["quantity"] == "0.003"


def test_reduce_only_orders_ignore_the_minimum_notional(creds, tmp_path):
    """Otherwise a dust position can never be closed (the venue itself allows it)."""
    routes = {("GET", "/fapi/v1/exchangeInfo"): EXCHANGE_INFO,
              ("POST", "/fapi/v1/order"): {"orderId": 5, "status": "FILLED", "executedQty": "0.001",
                                           "avgPrice": "40000", "updateTime": 1790000000000},
              ("GET", "/fapi/v1/userTrades"): []}
    broker, session = make_broker(creds, tmp_path, routes)
    dust = Order("id-dust", "BTCUSDT", Side.SELL, 0.001, pd.Timestamp("2026-10-05", tz="UTC"), 40_000.0,
                 reduce_only=True)
    fill = broker.submit(dust, 40_000.0, pd.Timestamp.now(tz="UTC"), 1e7)   # 40 USDT < 100 minimum
    assert fill is not None
    assert [c for c in session.calls if c[0] == "POST"][0][2]["reduceOnly"] == "true"


def test_funding_is_reported_from_the_venue_once_and_survives_a_restart(creds, tmp_path):
    """The venue settles funding itself; the adapter only reports what was charged, without ever
    counting a settlement twice (the earlier version re-summed the whole history on every call)."""
    calls = []

    def income(params):
        calls.append(dict(params))
        return [{"income": "-1.50"}, {"income": "0.25"}]        # paid 1.50, received 0.25

    routes = {("GET", "/fapi/v1/income"): income}
    broker, _ = make_broker(creds, tmp_path, routes)
    t0 = pd.Timestamp("2026-10-06 00:01", tz="UTC")
    assert broker.settles_funding_itself
    assert broker.funding_paid_since_last_check(t0) == 0.0       # first call only arms the cursor
    assert not calls
    t1 = t0 + pd.Timedelta(hours=4)
    assert broker.funding_paid_since_last_check(t1) == pytest.approx(1.25)
    assert calls[-1]["startTime"] == int(t0.value // 10**6) + 1 and calls[-1]["incomeType"] == "FUNDING_FEE"

    restarted, _ = make_broker(creds, tmp_path, routes)          # same state directory
    t2 = t1 + pd.Timedelta(hours=4)
    restarted.funding_paid_since_last_check(t2)
    assert calls[-1]["startTime"] == int(t1.value // 10**6) + 1  # resumes where it stopped
    assert broker.apply_funding("BTCUSDT", 0.0001, 50_000.0, t2) == 0.0
