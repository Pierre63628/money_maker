"""Live broker placeholder.

Deliberately NOT implemented. Connecting a real account or sending a real order requires
the owner's explicit approval (see docs/ARCHITECTURE.md, phase P7). When approved, the
implementation must satisfy the ``Broker`` interface and additionally provide:
signed REST/WebSocket clients, client-order-id idempotency on the exchange side, position
and balance reconciliation, rate-limit handling, retries with backoff, and a hard kill switch.
"""

from __future__ import annotations

from quant.brokers.base import BrokerError


class LiveTradingDisabled(BrokerError):
    pass


class LiveBroker:
    def __init__(self, *_, **__):
        raise LiveTradingDisabled(
            "Live trading is disabled. It requires explicit written approval from the account owner "
            "and a reviewed implementation (phase P7). Use PaperBroker."
        )
