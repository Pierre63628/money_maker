"""Audit an external trade list for unrealistic intrabar fills.

Bar backtesters (e.g. TradingView's broker emulator) must guess the path inside a
bar. Tight trailing stops then "exit" near the most favourable extreme of the
bar, which is unachievable live: the high is only known after it prints. For
exits decided without intrabar foresight, the exit location in the bar range is
roughly spread out; a heavy pile-up at the favourable extreme is a red flag.

exit_location = (exit - low) / (high - low) for longs, (high - exit) / (high - low)
for shorts; 1.0 = best possible price inside the exit bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest


@dataclass
class FillAuditReport:
    n_trades: int
    n_matched: int
    exit_location_median: float
    share_exit_best_5pct: float
    share_exit_outside_range: float
    share_entry_outside_range: float
    binom_pvalue: float
    verdict: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _locate(price: np.ndarray, high: np.ndarray, low: np.ndarray, direction: np.ndarray) -> np.ndarray:
    rng = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        loc_long = (price - low) / rng
    loc = np.where(direction > 0, loc_long, 1.0 - loc_long)
    return np.where(rng > 0, loc, np.nan)


def audit_intrabar_fills(trades: pd.DataFrame, bars: pd.DataFrame, tolerance_bps: float = 2.0,
                         suspicious_share: float = 0.35) -> FillAuditReport:
    """``trades`` needs direction (+1/-1 or 'long'/'short'), entry_time, entry_price, exit_time, exit_price.

    ``bars`` is an OHLC frame indexed by bar open time on the same venue as the trades.
    Times must be bar open timestamps (UTC).
    """
    t = trades.copy()
    if not pd.api.types.is_numeric_dtype(t["direction"]):
        t["direction"] = t["direction"].map({"long": 1, "short": -1})
    entry = bars.reindex(pd.DatetimeIndex(t["entry_time"]))
    exit_ = bars.reindex(pd.DatetimeIndex(t["exit_time"]))
    matched = entry["high"].notna().to_numpy() & exit_["high"].notna().to_numpy()
    t, entry, exit_ = t[matched], entry[matched], exit_[matched]
    n = int(matched.sum())
    if n == 0:
        return FillAuditReport(len(trades), 0, np.nan, np.nan, np.nan, np.nan, np.nan, "NO_DATA")

    d = t["direction"].to_numpy(dtype=float)
    tol = tolerance_bps / 1e4
    xh, xl = exit_["high"].to_numpy(), exit_["low"].to_numpy()
    eh, el = entry["high"].to_numpy(), entry["low"].to_numpy()
    xp, ep = t["exit_price"].to_numpy(dtype=float), t["entry_price"].to_numpy(dtype=float)

    loc = _locate(xp, xh, xl, d)
    valid = ~np.isnan(loc)
    best = float(np.mean(loc[valid] >= 0.95)) if valid.any() else np.nan
    outside_exit = float(np.mean((xp > xh * (1 + tol)) | (xp < xl * (1 - tol))))
    outside_entry = float(np.mean((ep > eh * (1 + tol)) | (ep < el * (1 - tol))))
    k = int(np.sum(loc[valid] >= 0.95))
    pvalue = float(binomtest(k, int(valid.sum()), 0.05, alternative="greater").pvalue) if valid.any() else np.nan

    if outside_exit > 0.01 or outside_entry > 0.01:
        verdict = "IMPOSSIBLE_FILLS"
    elif best >= suspicious_share and pvalue < 1e-6:
        verdict = "SUSPECT_INTRABAR_EXITS"
    else:
        verdict = "NO_OBVIOUS_ARTIFACT"
    return FillAuditReport(len(trades), n, float(np.nanmedian(loc)), best, outside_exit, outside_entry, pvalue, verdict)
