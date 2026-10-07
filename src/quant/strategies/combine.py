"""Strategy combinations that avoid parameter selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day
from quant.portfolio.sizing import hold_between_rebalances
from quant.strategies.base import Strategy


def _aligned(strategy: Strategy, data: MarketData) -> pd.DataFrame:
    return strategy.target_weights(data).reindex(index=data.index, columns=data.symbols).fillna(0.0)


@dataclass(frozen=True)
class ParameterEnsemble(Strategy):
    """Equal-weight average of the target weights of every member (e.g. a whole parameter grid).

    No member is selected, so there is no selection bias to overfit; netting across members
    also lowers turnover.
    """

    members: tuple[Strategy, ...]
    label: str = "ensemble"
    family = "ensemble"
    provenance = "internal:parameter-ensemble"

    @property
    def name(self) -> str:
        return f"Ensemble[{self.label}]({len(self.members)} members)"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        total = None
        for m in self.members:
            w = _aligned(m, data)
            total = w if total is None else total + w
        return total / len(self.members)


@dataclass(frozen=True)
class RiskParitySleeves(Strategy):
    """Combine sleeves with weights inversely proportional to their trailing volatility.

    Sleeve returns are estimated causally from each sleeve's lagged weights and close-to-close
    asset returns (frictionless proxy, used only for scaling). The combined book is scaled
    to ``target_vol`` (annualised) and capped at ``max_gross``; scales update every
    ``rebalance_hours``.
    """

    sleeves: tuple[Strategy, ...]
    label: str = "risk_parity"
    vol_lookback_days: int = 90
    target_vol: float = 0.30
    max_gross: float = 1.0
    rebalance_hours: int = 24
    drawdown_brake: float = 0.0   # >0: scale exposure by clip(1 - proxy drawdown / brake, floor, 1)
    brake_floor: float = 0.25
    family = "multi_sleeve"
    provenance = "internal:inverse-vol-sleeves"

    @property
    def name(self) -> str:
        return f"RiskParity[{self.label}]({len(self.sleeves)} sleeves, tv={self.target_vol})"

    def target_weights(self, data: MarketData) -> pd.DataFrame:
        bpd = bars_per_day(data.interval)
        window = max(20, int(self.vol_lookback_days * bpd))
        asset_ret = data.close.ffill().pct_change(fill_method=None).fillna(0.0)
        sleeve_w, sleeve_vol = [], []
        for s in self.sleeves:
            w = _aligned(s, data)
            r = (w.shift(1).fillna(0.0) * asset_ret).sum(axis=1)
            vol = r.rolling(window, min_periods=window // 2).std() * math.sqrt(365 * bpd)
            sleeve_w.append(w)
            sleeve_vol.append(vol)
        vols = pd.concat(sleeve_vol, axis=1)
        inv = 1.0 / vols.where(vols > 0)
        alloc = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
        combined_ret = sum(alloc.iloc[:, i].shift(1).fillna(0.0) *
                           (sleeve_w[i].shift(1).fillna(0.0) * asset_ret).sum(axis=1) for i in range(len(self.sleeves)))
        port_vol = combined_ret.rolling(window, min_periods=window // 2).std() * math.sqrt(365 * bpd)
        scale = (self.target_vol / port_vol.where(port_vol > 0)).fillna(0.0)
        if self.drawdown_brake > 0:
            proxy = (1 + scale.shift(1).fillna(0.0).clip(upper=self.max_gross) * combined_ret).cumprod()
            dd = 1 - proxy / proxy.cummax()
            scale = scale * np.clip(1 - dd / self.drawdown_brake, self.brake_floor, 1.0)
        book = sum(sleeve_w[i].mul(alloc.iloc[:, i] * scale, axis=0) for i in range(len(self.sleeves)))
        gross = book.abs().sum(axis=1)
        book = book.mul(np.minimum(1.0, self.max_gross / gross.where(gross > 0, 1.0)), axis=0)
        return hold_between_rebalances(book, self.rebalance_hours, interval=data.interval)
