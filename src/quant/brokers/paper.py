"""Paper broker: simulated execution against real market prices, with persistent state.

- fills market orders at the observed market price +/- (half spread + slippage + impact) and
  charges taker fees, using the same ``CostModel`` as the backtester;
- refuses duplicate client order ids (idempotency across restarts);
- persists balances/positions atomically after every mutation and appends every fill
  and funding event to a journal, so a crash never loses or double-counts state.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtesting.costs import CostModel
from quant.brokers.base import Broker, DuplicateOrderError
from quant.execution.orders import Fill, Order


@dataclass
class PaperBroker(Broker):
    state_dir: Path
    costs: CostModel
    initial_capital: float = 10_000.0
    max_participation: float = 0.05

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / "broker_state.json"
        self._journal_path = self.state_dir / "journal.jsonl"
        if self._state_path.exists():
            state = json.loads(self._state_path.read_text())
            self._cash = float(state["cash"])
            self._positions = {k: float(v) for k, v in state["positions"].items()}
            self._order_ids = set(state["order_ids"])
        else:
            self._cash = float(self.initial_capital)
            self._positions = {}
            self._order_ids = set()
            self._persist()

    # ------------------------------------------------------------- persistence
    def _persist(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cash": self._cash, "positions": self._positions,
                                   "order_ids": sorted(self._order_ids)}))
        tmp.replace(self._state_path)

    def _journal(self, record: dict) -> None:
        with self._journal_path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # --------------------------------------------------------------- interface
    def positions(self) -> pd.Series:
        return pd.Series({k: v for k, v in self._positions.items() if v != 0.0}, dtype=float)

    def cash(self) -> float:
        return self._cash

    def equity(self, prices: pd.Series) -> float:
        value = self._cash
        for sym, qty in self._positions.items():
            if qty == 0.0:
                continue
            px = prices.get(sym, math.nan)
            if not math.isfinite(px):
                raise ValueError(f"no price to mark {sym}")
            value += qty * px
        return value

    def known_order_ids(self) -> set[str]:
        return set(self._order_ids)

    def ledger_positions(self) -> pd.Series:
        """Positions rebuilt from the fill journal (independent of the state file), for reconciliation."""
        pos: dict[str, float] = {}
        if self._journal_path.exists():
            for line in self._journal_path.read_text().splitlines():
                rec = json.loads(line)
                if rec.get("type") == "fill":
                    sign = 1.0 if rec["side"] == "BUY" else -1.0
                    pos[rec["symbol"]] = pos.get(rec["symbol"], 0.0) + sign * float(rec["quantity"])
        return pd.Series({k: v for k, v in pos.items() if abs(v) > 1e-9}, dtype=float)

    def submit(self, order: Order, market_price: float, now: pd.Timestamp, bar_quote_volume: float,
               bar_sigma: float = 0.02, bars_per_day: float = 24.0,
               observed_half_spread_bps: float | None = None) -> Fill | None:
        """``market_price`` is the mid. ``bar_quote_volume``/``bar_sigma``: lagged average bar notional and bar
        volatility. ``observed_half_spread_bps`` (from the live order book) replaces the modelled spread."""
        if order.client_order_id in self._order_ids:
            raise DuplicateOrderError(order.client_order_id)
        if not (math.isfinite(market_price) and market_price > 0):
            raise ValueError(f"invalid market price for {order.symbol}: {market_price}")
        qty = order.quantity
        current = self._positions.get(order.symbol, 0.0)
        if order.reduce_only:
            qty = min(qty, abs(current)) if current * order.signed_quantity < 0 else 0.0
        notional = qty * market_price
        if math.isfinite(bar_quote_volume) and bar_quote_volume > 0:
            cap = self.max_participation * bar_quote_volume
            if notional > cap:
                qty = cap / market_price
                notional = cap
        self._order_ids.add(order.client_order_id)
        if qty <= 0:
            self._persist()
            self._journal({"type": "order_rejected", "order": order.to_dict(), "time": now.isoformat(),
                           "reason": "reduce_only with nothing to reduce"})
            return None
        daily_volume = bar_quote_volume * bars_per_day if math.isfinite(bar_quote_volume) else math.nan
        modelled = float(self.costs.half_spread_matrix([order.symbol], np.array([[daily_volume]]))[0, 0])
        half_spread = observed_half_spread_bps if observed_half_spread_bps is not None else modelled
        impact = 0.0
        if math.isfinite(bar_quote_volume) and bar_quote_volume > 0 and math.isfinite(bar_sigma):
            impact = self.costs.impact_coef * bar_sigma * 1e4 * math.sqrt(notional / bar_quote_volume)
        cost_bps = half_spread + self.costs.slippage_bps + impact
        sign = 1.0 if order.signed_quantity > 0 else -1.0
        price = market_price * (1 + sign * cost_bps / 1e4)
        fee = qty * price * self.costs.fee_bps_for(order.symbol) / 1e4
        self._cash -= sign * qty * price + fee
        self._positions[order.symbol] = current + sign * qty
        if abs(self._positions[order.symbol]) < 1e-12:
            self._positions.pop(order.symbol)
        fill = Fill(order.client_order_id, order.symbol, order.side, qty, price, fee, now, order.reference_price,
                    {"market_price": market_price, "cost_bps": cost_bps, "requested_qty": order.quantity,
                     "half_spread_bps_used": half_spread, "half_spread_bps_modelled": modelled})
        self._persist()
        self._journal({"type": "fill", **fill.to_dict(), "strategy_id": order.strategy_id})
        return fill

    def apply_funding(self, symbol: str, rate: float, mark_price: float, settle_time: pd.Timestamp) -> float:
        qty = self._positions.get(symbol, 0.0)
        if qty == 0.0 or not math.isfinite(rate):
            return 0.0
        paid = qty * mark_price * rate
        self._cash -= paid
        self._persist()
        self._journal({"type": "funding", "symbol": symbol, "rate": rate, "mark_price": mark_price, "paid": paid,
                       "time": settle_time.isoformat()})
        return paid
