"""Strategy interface.

A strategy maps market data to target portfolio weights. ``weights.loc[t]`` must
be computable from data up to the close of bar ``t``; the backtester executes it
at a later bar. Weights are fractions of equity (1.0 = 100% notional long,
-0.5 = 50% short). Strategies never see equity, fills or risk limits: sizing
constraints belong to the risk engine / portfolio layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import ClassVar

import pandas as pd

from quant.data.market_data import MarketData


@dataclass(frozen=True)
class Strategy(ABC):
    family: ClassVar[str] = "unspecified"
    # Where the idea comes from: "internal", "textbook:<ref>", "strategy_factory:<resultId>".
    provenance: ClassVar[str] = "internal"

    @abstractmethod
    def target_weights(self, data: MarketData) -> pd.DataFrame:
        ...

    @property
    def params(self) -> dict:
        return asdict(self)

    @property
    def name(self) -> str:
        args = ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))
        return f"{type(self).__name__}({args})"


def equal_allocation(signal: pd.DataFrame) -> pd.DataFrame:
    """Scale per-symbol signals in [-1, 1] by 1/N so gross exposure stays <= 1."""
    n = max(signal.shape[1], 1)
    return signal / n
