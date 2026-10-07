"""Backtest output container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from quant.backtesting.engine import BacktestConfig


@dataclass
class BacktestResult:
    name: str
    interval: str
    config: "BacktestConfig"
    bars: pd.DataFrame            # equity, exposures and per-bar cost flows
    trades: pd.DataFrame          # one row per round trip (open trades marked-to-market)
    target_weights: pd.DataFrame  # decided weights (before execution delay), after leverage cap
    n_partial_fills: int
    liquidated_at: pd.Timestamp | None
    ruined_at: pd.Timestamp | None
    final_positions: pd.Series
    n_maker_fills: int = 0
    n_taker_fills: int = 0
    n_unfilled_order_bars: int = 0

    @property
    def equity(self) -> pd.Series:
        return self.bars["equity"]

    @property
    def returns(self) -> pd.Series:
        prev = self.equity.shift(1).fillna(self.config.initial_capital)
        return self.equity / prev - 1.0

    def metrics(self, periods: str = "1D") -> dict[str, Any]:
        from quant.evaluation.metrics import compute_metrics

        return compute_metrics(self, periods=periods)
