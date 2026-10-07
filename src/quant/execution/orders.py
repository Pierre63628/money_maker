"""Order and fill model shared by every broker (paper or live)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum

import pandas as pd


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    NEW = "NEW"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


def client_order_id(strategy_id: str, symbol: str, decision_time: pd.Timestamp, target_qty: float) -> str:
    """Deterministic id: re-running the same decision (e.g. after a crash) yields the same id,
    so the broker can refuse duplicates instead of doubling the position."""
    raw = f"{strategy_id}|{symbol}|{decision_time.isoformat()}|{target_qty:.10g}"
    return "q" + hashlib.sha256(raw.encode()).hexdigest()[:30]


@dataclass
class Order:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float                 # base units, > 0
    decision_time: pd.Timestamp     # close of the bar the decision was based on
    reference_price: float          # price used for sizing (last close); slippage is measured against it
    reduce_only: bool = False
    strategy_id: str = ""
    status: OrderStatus = OrderStatus.NEW
    reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side is Side.BUY else -self.quantity

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["status"] = self.status.value
        d["decision_time"] = self.decision_time.isoformat()
        return d


@dataclass
class Fill:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    fee: float
    fill_time: pd.Timestamp
    reference_price: float
    extra: dict = field(default_factory=dict)

    @property
    def slippage_bps(self) -> float:
        sign = 1.0 if self.side is Side.BUY else -1.0
        return sign * (self.price / self.reference_price - 1.0) * 1e4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["fill_time"] = self.fill_time.isoformat()
        d["slippage_bps"] = self.slippage_bps
        return d
