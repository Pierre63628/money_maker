"""Stateful entry/exit latch: turns event conditions into a held position in {-1, 0, 1}."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _latch(entry_long, exit_long, entry_short, exit_short, max_hold):
    T, N = entry_long.shape
    out = np.zeros((T, N))
    for j in range(N):
        pos = 0.0
        held = 0
        for t in range(T):
            if pos != 0.0:
                held += 1
            timed_out = max_hold > 0 and held >= max_hold
            if pos > 0 and (exit_long[t, j] or timed_out):
                pos = 0.0
            elif pos < 0 and (exit_short[t, j] or timed_out):
                pos = 0.0
            if pos == 0.0:
                held = 0
                if entry_long[t, j] and not entry_short[t, j]:
                    pos = 1.0
                elif entry_short[t, j] and not entry_long[t, j]:
                    pos = -1.0
            out[t, j] = pos
    return out


def latch_positions(entry_long: pd.DataFrame, exit_long: pd.DataFrame,
                    entry_short: pd.DataFrame | None = None, exit_short: pd.DataFrame | None = None,
                    max_hold: int = 0) -> pd.DataFrame:
    """Position held after processing bar t's conditions (known at the close of t).

    On a bar where a long exits and a short entry fires, the position flips.
    ``max_hold`` > 0 forces an exit after that many bars in a position (time stop);
    a new entry is allowed on the same bar if its condition holds.
    NaN conditions (indicator warm-up) count as False.
    """
    def arr(frame):
        return np.ascontiguousarray(frame.fillna(False).to_numpy(dtype=np.bool_))

    el, xl = arr(entry_long), arr(exit_long)
    if entry_short is None:
        es = np.zeros_like(el)
        xs = np.zeros_like(el)
    else:
        es, xs = arr(entry_short), arr(exit_short)
    return pd.DataFrame(_latch(el, xl, es, xs, int(max_hold)), index=entry_long.index, columns=entry_long.columns)
