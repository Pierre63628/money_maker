"""MetaTrader 5 adapter (NOT enabled).

Placing real orders requires the owner's explicit approval; this module only documents what a
compliant implementation needs and refuses to run until it exists.

Practical constraints to settle before writing it:
- **Platform**: the official ``MetaTrader5`` Python package is Windows-only. On macOS the options are a
  Windows VPS (also better for uptime: the paper loop showed that a sleeping laptop silently skips bars)
  or an MQL5 Expert Advisor driven by files/sockets.
- **Instruments**: CFDs, not Binance perpetuals. Symbol names, contract size, lot step, minimum volume,
  stop level and trading sessions all come from ``symbol_info`` and must be honoured when sizing.
- **Costs**: quoted spread + commission + overnight swap (charged on both legs, triple on Wednesdays at
  most brokers). Use ``BrokerProfile`` to re-validate a strategy with the broker's real numbers first.
- **Accounting**: MT5 positions are per-symbol netted or hedged depending on the account type; the
  reconciliation logic must match the account mode.
- **Safety**: idempotent client order ids (``magic``/``comment`` fields), retry-safe order sending,
  position/balance reconciliation on every step, kill switch, and rate-limit handling.
"""

from __future__ import annotations

from quant.brokers.base import BrokerError


class MT5NotConfigured(BrokerError):
    pass


class MT5Broker:
    def __init__(self, *_, **__):
        raise MT5NotConfigured(
            "The MetaTrader 5 live adapter is not implemented. Supply the broker's contract "
            "specification (symbols, spreads, swaps, minimum volumes), validate the strategy with "
            "BrokerProfile, then approve the live implementation explicitly."
        )
