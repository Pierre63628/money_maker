"""Broker cost/instrument profiles, so a strategy can be re-tested under a specific venue.

A CFD broker (MetaTrader 5 and similar) differs from Binance perpetuals in three ways that decide
whether a strategy survives:
- **instruments**: only the symbols the broker lists can be traded;
- **spread + commission**: quoted spreads are far wider than an exchange's order book;
- **overnight financing (swap)**: charged on *both* legs of a hedged book, unlike perpetual funding
  which is a transfer between longs and shorts.

A profile is plain JSON so it can be filled from the broker's contract specification:

```json
{
  "name": "example-broker",
  "account_currency": "USD",
  "commission_bps_per_side": 0.0,
  "slippage_bps": 2.0,
  "default": {"half_spread_bps": 10.0, "swap_bps_per_day": 2.0},
  "instruments": {
    "BTCUSD": {"source": "BTCUSDT", "half_spread_bps": 3.0, "swap_long_bps_per_day": 2.0,
               "swap_short_bps_per_day": 2.0, "min_notional": 100.0},
    "ETHUSD": {"source": "ETHUSDT", "half_spread_bps": 4.0}
  },
  "max_leverage": 2.0
}
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from quant.backtesting.costs import CostModel
from quant.backtesting.engine import BacktestConfig, ExecutionModel


@dataclass(frozen=True)
class Instrument:
    broker_symbol: str
    source_symbol: str            # the research symbol this instrument tracks (e.g. BTCUSDT on Binance)
    half_spread_bps: float
    swap_long_bps_per_day: float
    swap_short_bps_per_day: float
    min_notional: float = 0.0

    @property
    def financing_bps_per_day(self) -> float:
        """Cost of carrying either leg: a market-neutral book pays roughly the average of both sides."""
        return (self.swap_long_bps_per_day + self.swap_short_bps_per_day) / 2


@dataclass(frozen=True)
class BrokerProfile:
    name: str
    instruments: dict[str, Instrument]
    commission_bps_per_side: float = 0.0
    slippage_bps: float = 2.0
    impact_coef: float = 0.5
    max_leverage: float = 1.0
    account_currency: str = "USD"
    notes: str = ""
    quotes_are_dealable: bool = False   # CFDs: you take the broker's price, no passive fills

    @property
    def source_symbols(self) -> list[str]:
        return sorted({i.source_symbol for i in self.instruments.values()})

    @classmethod
    def from_json(cls, path: str | Path) -> "BrokerProfile":
        raw = json.loads(Path(path).read_text())
        default = raw.get("default", {})
        instruments = {}
        for broker_symbol, spec in raw["instruments"].items():
            half = float(spec.get("half_spread_bps", default.get("half_spread_bps", 10.0)))
            swap = float(spec.get("swap_bps_per_day", default.get("swap_bps_per_day", 0.0)))
            instruments[broker_symbol] = Instrument(
                broker_symbol=broker_symbol,
                source_symbol=spec.get("source", broker_symbol),
                half_spread_bps=half,
                swap_long_bps_per_day=float(spec.get("swap_long_bps_per_day", swap)),
                swap_short_bps_per_day=float(spec.get("swap_short_bps_per_day", swap)),
                min_notional=float(spec.get("min_notional", 0.0)),
            )
        return cls(
            name=raw["name"],
            instruments=instruments,
            commission_bps_per_side=float(raw.get("commission_bps_per_side", 0.0)),
            slippage_bps=float(raw.get("slippage_bps", 2.0)),
            impact_coef=float(raw.get("impact_coef", 0.5)),
            max_leverage=float(raw.get("max_leverage", 1.0)),
            account_currency=raw.get("account_currency", "USD"),
            notes=raw.get("notes", ""),
            quotes_are_dealable=bool(raw.get("quotes_are_dealable", False)),
        )

    def backtest_config(self, initial_capital: float = 10_000.0) -> BacktestConfig:
        """Costs as the broker charges them, keyed by the research symbols."""
        spreads = {i.source_symbol: i.half_spread_bps for i in self.instruments.values()}
        financing = {i.source_symbol: i.financing_bps_per_day for i in self.instruments.values()}
        min_notional = max((i.min_notional for i in self.instruments.values()), default=5.0)
        costs = CostModel(
            fee_bps=self.commission_bps_per_side,
            half_spread_bps=max(spreads.values()) if spreads else 10.0,
            slippage_bps=self.slippage_bps,
            impact_coef=self.impact_coef,
            per_symbol_half_spread_bps=spreads,
            per_symbol_financing_bps_per_day=financing,
            funding=False,                       # CFDs have no perpetual funding; swaps are the carry
            financing_bps_per_day=0.0,
        )
        return BacktestConfig(
            initial_capital=initial_capital,
            costs=costs,
            max_gross_leverage=self.max_leverage,
            min_trade_notional=max(min_notional, 5.0),
            # a broker quote is dealable on request: no resting limit orders earning the spread
            execution=ExecutionModel(mode="taker"),
        )


TYPICAL_CRYPTO_CFD = BrokerProfile(
    name="typical-crypto-cfd (placeholder until the real specification is supplied)",
    instruments={
        sym: Instrument(sym, source, half_spread_bps=spread, swap_long_bps_per_day=2.0,
                        swap_short_bps_per_day=2.0, min_notional=100.0)
        for sym, source, spread in [
            ("BTCUSD", "BTCUSDT", 3.0), ("ETHUSD", "ETHUSDT", 4.0), ("SOLUSD", "SOLUSDT", 8.0),
            ("XRPUSD", "XRPUSDT", 8.0), ("ADAUSD", "ADAUSDT", 10.0), ("DOGEUSD", "DOGEUSDT", 10.0),
            ("LTCUSD", "LTCUSDT", 10.0), ("BNBUSD", "BNBUSDT", 10.0), ("LINKUSD", "LINKUSDT", 12.0),
            ("AVAXUSD", "AVAXUSDT", 12.0),
        ]
    },
    commission_bps_per_side=0.0,
    slippage_bps=2.0,
    max_leverage=1.0,
    notes="Order of magnitude only: real spreads and swaps must come from the broker's contract specification.",
)
