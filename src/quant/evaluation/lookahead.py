"""Generic look-ahead detector.

A causal strategy must produce, for every bar ``t <= c``, the same weights whether
it sees the full history or only the first ``c + 1`` bars. Any difference means
future data leaked into past decisions (centred windows, bfill, full-sample
normalisation, shift(-k), fitting on the whole sample, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant.data.market_data import MarketData


@dataclass
class LookaheadReport:
    strategy: str
    cuts: list[int]
    violations: list[tuple[int, pd.Timestamp, float]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        if self.passed:
            return f"{self.strategy}: no look-ahead detected over {len(self.cuts)} truncation points"
        first = self.violations[0]
        return (f"{self.strategy}: LOOK-AHEAD at {len(self.violations)} cut(s); first cut={first[0]} "
                f"differs at {first[1]} (max abs diff {first[2]:.3g})")


def check_no_lookahead(strategy, data: MarketData, n_cuts: int = 8, min_bars: int | None = None,
                       seed: int = 0, atol: float = 1e-9) -> LookaheadReport:
    full = strategy.target_weights(data)
    n = len(data)
    lo = min_bars if min_bars is not None else max(10, n // 5)
    if n - 1 <= lo:
        raise ValueError("not enough bars to test truncations")
    rng = np.random.default_rng(seed)
    cuts = sorted(set(int(c) for c in rng.integers(lo, n - 1, size=n_cuts)))
    report = LookaheadReport(strategy=getattr(strategy, "name", type(strategy).__name__), cuts=cuts)
    for c in cuts:
        partial = strategy.target_weights(data.head(c + 1))
        a = full.iloc[: c + 1].reindex(columns=partial.columns).to_numpy(dtype=float)
        b = partial.to_numpy(dtype=float)
        if a.shape != b.shape:
            report.violations.append((c, data.index[c], float("inf")))
            continue
        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a, nan=np.inf) - np.nan_to_num(b, nan=np.inf)))
        diff = np.where(np.isnan(diff), np.inf, diff)
        bad_rows = np.where((diff > atol).any(axis=1))[0]
        if len(bad_rows):
            report.violations.append((c, data.index[bad_rows[0]], float(diff.max())))
    return report
