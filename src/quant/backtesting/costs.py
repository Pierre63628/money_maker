"""Transaction cost model.

Every fill pays, relative to the reference price (bar open):
    half_spread_bps + slippage_bps + impact_coef * sigma_bar * sqrt(notional / adv_bar)   [price]
    + fee_bps on the filled notional                                                      [fee]

The square-root impact term uses per-bar volatility and per-bar average quote
volume, which is scale-invariant (sigma_bar * sqrt(Q / V_bar) == sigma_day * sqrt(Q / V_day)).
Both inputs are lagged by one bar in the engine, so costs never use future data.

Defaults are deliberately conservative for Binance USDT-M VIP0 taker orders
(fee 5 bps). Liquid majors trade with ~0.5-1 bp half spread; small caps far more,
so pass ``per_symbol_half_spread_bps`` when trading alts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

import numpy as np


def _lookup(table: Mapping[str, float], symbol: str, default: float) -> float:
    """Exact symbol key first, then ``"*suffix"`` rules (e.g. ``{"*@spot": 10.0}``)."""
    if symbol in table:
        return float(table[symbol])
    for key, value in table.items():
        if key.startswith("*") and symbol.endswith(key[1:]):
            return float(value)
    return float(default)


@dataclass(frozen=True)
class CostModel:
    fee_bps: float = 5.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 1.0
    impact_coef: float = 0.5
    per_symbol_half_spread_bps: Mapping[str, float] = field(default_factory=dict)
    per_symbol_fee_bps: Mapping[str, float] = field(default_factory=dict)  # exact symbol or "*suffix" rule
    funding: bool = True
    # CFD-style overnight financing charged on the GROSS position (both legs pay), in bps per day.
    # Perpetual funding is a transfer between longs and shorts and is modelled by ``funding`` instead.
    financing_bps_per_day: float = 0.0
    per_symbol_financing_bps_per_day: Mapping[str, float] = field(default_factory=dict)
    # "fixed": half_spread_bps (or the per-symbol override) everywhere.
    # "volume": half spread = spread_coef_bps * sqrt(1e9 / lagged daily quote volume), clipped;
    #           ~1 bp for a $1bn/day market, ~7 bp at $20m/day. Per-symbol overrides still win.
    spread_model: Literal["fixed", "volume"] = "fixed"
    spread_coef_bps: float = 1.0
    min_half_spread_bps: float = 0.5
    max_half_spread_bps: float = 25.0

    def fee_bps_for(self, symbol: str) -> float:
        return _lookup(self.per_symbol_fee_bps, symbol, self.fee_bps)

    def financing_bps_for(self, symbol: str) -> float:
        return _lookup(self.per_symbol_financing_bps_per_day, symbol, self.financing_bps_per_day)

    def half_spread_for(self, symbol: str) -> float:
        return float(self.per_symbol_half_spread_bps.get(symbol, self.half_spread_bps))

    def half_spread_matrix(self, symbols: list[str], daily_quote_volume: np.ndarray) -> np.ndarray:
        """(T, N) half spread in bps. ``daily_quote_volume`` must already be lagged (known before the bar)."""
        T, N = daily_quote_volume.shape
        if self.spread_model == "fixed":
            out = np.broadcast_to(np.array([self.half_spread_for(s) for s in symbols]), (T, N)).copy()
        elif self.spread_model == "volume":
            with np.errstate(divide="ignore", invalid="ignore"):
                est = self.spread_coef_bps * np.sqrt(1e9 / daily_quote_volume)
            out = np.clip(np.nan_to_num(est, nan=self.max_half_spread_bps, posinf=self.max_half_spread_bps),
                          self.min_half_spread_bps, self.max_half_spread_bps)
        else:
            raise ValueError(f"unknown spread_model {self.spread_model}")
        for j, sym in enumerate(symbols):
            if sym in self.per_symbol_half_spread_bps:
                out[:, j] = self.per_symbol_half_spread_bps[sym]
        return np.ascontiguousarray(out, dtype=np.float64)

    def scaled(self, factor: float) -> "CostModel":
        """Multiply every cost component (stress testing). Funding is a market cash flow, not scaled."""
        return replace(
            self,
            fee_bps=self.fee_bps * factor,
            half_spread_bps=self.half_spread_bps * factor,
            slippage_bps=self.slippage_bps * factor,
            impact_coef=self.impact_coef * factor,
            financing_bps_per_day=self.financing_bps_per_day * factor,
            per_symbol_financing_bps_per_day={k: v * factor for k, v in self.per_symbol_financing_bps_per_day.items()},
            spread_coef_bps=self.spread_coef_bps * factor,
            min_half_spread_bps=self.min_half_spread_bps * factor,
            max_half_spread_bps=self.max_half_spread_bps * factor,
            per_symbol_half_spread_bps={k: v * factor for k, v in self.per_symbol_half_spread_bps.items()},
            per_symbol_fee_bps={k: v * factor for k, v in self.per_symbol_fee_bps.items()},
        )

    @classmethod
    def zero(cls) -> "CostModel":
        return cls(fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0, impact_coef=0.0, funding=False)
