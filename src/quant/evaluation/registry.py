"""Append-only experiment registry.

Every backtest that informs a decision is logged. The trial count feeds the
Deflated Sharpe Ratio: hiding failed trials would make surviving results look better
than they are, so entries are never deleted.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class ExperimentRegistry:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def log(self, experiment: str, strategy: str, family: str, provenance: str, params: dict, data: dict,
            metrics: dict, stage: str = "in_sample", notes: str = "") -> None:
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment,
            "stage": stage,
            "strategy": strategy,
            "family": family,
            "provenance": provenance,
            "params": params,
            "data": data,
            "metrics": {k: metrics[k] for k in metrics if not isinstance(metrics[k], (pd.Series, pd.DataFrame))},
            "notes": notes,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(_jsonable(record)) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.DataFrame([json.loads(line) for line in self.path.read_text().splitlines() if line.strip()])

    def n_trials(self, experiment: str | None = None, family: str | None = None) -> int:
        df = self.load()
        if df.empty:
            return 0
        if experiment is not None:
            df = df[df["experiment"] == experiment]
        if family is not None:
            df = df[df["family"] == family]
        return len(df)
