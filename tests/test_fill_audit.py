from __future__ import annotations

import numpy as np
import pandas as pd

from quant.evaluation.fill_audit import audit_intrabar_fills
from tests.conftest import gbm_frame


def _trades(bars: pd.DataFrame, exit_price_fn, n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entries = np.sort(rng.choice(len(bars) - 2, size=n, replace=False))
    rows = []
    for e in entries:
        direction = 1 if rng.random() < 0.5 else -1
        eb, xb = bars.iloc[e], bars.iloc[e + 1]
        rows.append({"direction": "long" if direction > 0 else "short", "entry_time": bars.index[e],
                     "entry_price": eb["open"], "exit_time": bars.index[e + 1],
                     "exit_price": exit_price_fn(xb, direction, rng)})
    return pd.DataFrame(rows)


def test_exits_at_favourable_extreme_are_flagged():
    bars = gbm_frame(2000, 7)
    trades = _trades(bars, lambda b, d, r: b["high"] * (1 - 1e-5) if d > 0 else b["low"] * (1 + 1e-5))
    report = audit_intrabar_fills(trades, bars)
    assert report.verdict == "SUSPECT_INTRABAR_EXITS"
    assert report.share_exit_best_5pct > 0.9


def test_close_exits_are_not_flagged():
    bars = gbm_frame(2000, 8)
    trades = _trades(bars, lambda b, d, r: b["close"])
    report = audit_intrabar_fills(trades, bars)
    assert report.verdict == "NO_OBVIOUS_ARTIFACT"


def test_fills_outside_bar_range_are_impossible():
    bars = gbm_frame(2000, 9)
    trades = _trades(bars, lambda b, d, r: b["high"] * 1.01)
    assert audit_intrabar_fills(trades, bars).verdict == "IMPOSSIBLE_FILLS"
