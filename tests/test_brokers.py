from __future__ import annotations

import json

import pandas as pd
import pytest

from quant.backtesting.costs import CostModel
from quant.brokers.base import DuplicateOrderError
from quant.brokers.live import LiveBroker, LiveTradingDisabled
from quant.brokers.paper import PaperBroker
from quant.execution.orders import Order, Side, client_order_id

NOW = pd.Timestamp("2026-09-14 10:00", tz="UTC")
COSTS = CostModel(fee_bps=5.0, half_spread_bps=1.0, slippage_bps=1.0, impact_coef=0.0)


def order(side=Side.BUY, qty=1.0, sym="BTCUSDT", t=NOW, price=100.0, **kw):
    return Order(client_order_id(kw.pop("sid", "s"), sym, t, qty if side is Side.BUY else -qty), sym, side, qty, t,
                 price, **kw)


def test_fill_costs_and_equity(tmp_path):
    b = PaperBroker(tmp_path, COSTS, initial_capital=1_000.0)
    fill = b.submit(order(qty=2.0), market_price=100.0, now=NOW, bar_quote_volume=1e9)
    assert fill.price == pytest.approx(100.0 * (1 + 2e-4))
    assert fill.fee == pytest.approx(2 * fill.price * 5e-4)
    assert fill.slippage_bps == pytest.approx(2.0)
    assert b.positions()["BTCUSDT"] == 2.0
    assert b.equity(pd.Series({"BTCUSDT": 100.0})) == pytest.approx(1_000 - 2 * (fill.price - 100) - fill.fee)


def test_duplicate_order_rejected_even_after_restart(tmp_path):
    b = PaperBroker(tmp_path, COSTS)
    o = order()
    b.submit(o, 100.0, NOW, 1e9)
    with pytest.raises(DuplicateOrderError):
        b.submit(o, 100.0, NOW, 1e9)
    restarted = PaperBroker(tmp_path, COSTS)
    with pytest.raises(DuplicateOrderError):
        restarted.submit(o, 100.0, NOW, 1e9)
    assert restarted.positions()["BTCUSDT"] == 1.0
    lines = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["type"] == "fill"


def test_reduce_only_and_participation_cap(tmp_path):
    b = PaperBroker(tmp_path, COSTS, max_participation=0.05)
    assert b.submit(order(side=Side.SELL, reduce_only=True), 100.0, NOW, 1e9) is None
    fill = b.submit(order(qty=100.0, sid="big"), 100.0, NOW, bar_quote_volume=10_000.0)
    assert fill.quantity == pytest.approx(5.0)  # 5% of 10,000 notional at 100


def test_funding_sign(tmp_path):
    b = PaperBroker(tmp_path, CostModel.zero(), initial_capital=1_000.0)
    b.submit(order(side=Side.SELL, qty=1.0), 100.0, NOW, 1e9)
    paid = b.apply_funding("BTCUSDT", 0.001, 100.0, NOW)
    assert paid == pytest.approx(-0.1) and b.cash() == pytest.approx(1_000 + 100 + 0.1)


def test_live_broker_is_disabled():
    with pytest.raises(LiveTradingDisabled):
        LiveBroker()
