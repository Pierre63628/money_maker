"""Broker interface. Strategy and risk code only ever talk to this interface, so
``PaperBroker`` can be swapped for a live implementation without touching them."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from quant.execution.orders import Fill, Order


class BrokerError(RuntimeError):
    pass


class DuplicateOrderError(BrokerError):
    pass


class Broker(ABC):
    # True when the venue debits/credits funding on its own (a real exchange): the loop must then
    # neither apply funding itself nor subtract it from the equity the venue reports.
    settles_funding_itself: bool = False

    @abstractmethod
    def positions(self) -> pd.Series:
        """Signed base quantity per symbol (only non-zero positions)."""

    @abstractmethod
    def cash(self) -> float:
        """Quote-currency balance (can be negative with leverage)."""

    @abstractmethod
    def equity(self, prices: pd.Series) -> float:
        """Account value marked at ``prices``."""

    @abstractmethod
    def submit(self, order: Order, market_price: float, now: pd.Timestamp, bar_quote_volume: float,
               bar_sigma: float = 0.02, bars_per_day: float = 24.0) -> Fill | None:
        """Execute (or route) an order. Must reject a ``client_order_id`` that was already processed."""

    @abstractmethod
    def apply_funding(self, symbol: str, rate: float, mark_price: float, settle_time: pd.Timestamp) -> float:
        """Settle funding on the current position; returns the amount paid (negative = received)."""

    @abstractmethod
    def known_order_ids(self) -> set[str]:
        ...

    def supported_symbols(self) -> set[str] | None:
        """Symbols the venue will accept, or None when it accepts whatever the data has.

        A venue rarely lists everything the research universe contains: the Binance testnet carries a
        fraction of mainnet's perpetuals, and a CFD broker carries a handful. The loop intersects its
        universe with this set instead of discovering the gap when an order is rejected."""
        return None
