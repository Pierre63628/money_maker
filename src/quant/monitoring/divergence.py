"""Compare a paper-trading run with a backtest replay of the same period.

Divergence has three usual sources, separated here:
- **execution**: realised cost per fill vs the cost the backtest charges;
- **timing**: the paper loop decides at the same bar close but fills seconds to minutes later;
- **state**: missed bars (machine asleep, API outage) leave the paper book behind the model book.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtesting.result import BacktestResult
from quant.evaluation.metrics import period_returns


@dataclass
class DivergenceReport:
    steps: int
    traded_steps: int
    skipped_steps: int
    paper_return: float
    backtest_return: float
    return_gap: float
    tracking_error_daily: float
    mean_cost_bps_paper: float
    mean_cost_bps_model: float
    mean_latency_seconds: float
    notes: list[str]

    def summary(self) -> str:
        lines = [
            f"steps {self.steps} (traded {self.traded_steps}, skipped {self.skipped_steps})",
            f"paper return {self.paper_return:+.2%} vs backtest {self.backtest_return:+.2%} "
            f"(gap {self.return_gap:+.2%})",
            f"daily tracking error {self.tracking_error_daily:.2%}",
            f"execution cost per fill: paper {self.mean_cost_bps_paper:.2f} bps vs model "
            f"{self.mean_cost_bps_model:.2f} bps; mean latency {self.mean_latency_seconds:.0f}s",
        ]
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def load_events(state_dir: Path) -> list[dict]:
    path = Path(state_dir) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paper_equity(events: list[dict]) -> pd.Series:
    rows = [(pd.Timestamp(e["decision_bar"]), float(e["equity"])) for e in events if e["event"] == "step_summary"]
    if not rows:
        return pd.Series(dtype=float)
    idx, values = zip(*rows)
    return pd.Series(values, index=pd.DatetimeIndex(idx, name="ts")).sort_index()


def compare(state_dir: Path, backtest: BacktestResult, model_cost_bps: float | None = None) -> DivergenceReport:
    """``backtest`` must cover the paper period with the same strategy and configuration."""
    events = load_events(state_dir)
    equity = paper_equity(events)
    steps = [e for e in events if e["event"] == "step_summary"]
    skipped = [e for e in events if e.get("message") == "late step skipped"]
    notes: list[str] = []
    if equity.empty:
        return DivergenceReport(0, 0, len(skipped), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                ["no paper step recorded yet"])

    window = backtest.bars.loc[(backtest.bars.index >= equity.index[0]) & (backtest.bars.index <= equity.index[-1])]
    if window.empty:
        notes.append("backtest does not cover the paper window")
        bt_return = np.nan
        bt_daily = pd.Series(dtype=float)
    else:
        bt_return = float(window["equity"].iloc[-1] / window["equity"].iloc[0] - 1)
        bt_daily = period_returns(backtest, "1D")
        bt_daily = bt_daily[(bt_daily.index >= equity.index[0]) & (bt_daily.index <= equity.index[-1])]

    paper_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    paper_daily = equity.resample("1D").last().pct_change().dropna()
    joined = pd.concat([paper_daily.rename("paper"), bt_daily.rename("backtest")], axis=1).dropna()
    tracking = float((joined["paper"] - joined["backtest"]).std()) if len(joined) > 1 else np.nan
    if len(joined) < 5:
        notes.append(f"only {len(joined)} overlapping days: the comparison is not yet meaningful")

    fills = [f for e in steps for f in e.get("fills", [])]
    costs = [f["extra"]["cost_bps"] for f in fills if "extra" in f and "cost_bps" in f["extra"]]
    modelled = [f["extra"].get("half_spread_bps_modelled", np.nan) for f in fills if "extra" in f]
    latency = [f.get("latency_seconds", np.nan) for f in fills]
    if skipped:
        notes.append(f"{len(skipped)} bars skipped as late (machine asleep or API outage): the paper book lags the model")
    return DivergenceReport(
        steps=len(steps) + len(skipped),
        traded_steps=len(steps),
        skipped_steps=len(skipped),
        paper_return=paper_return,
        backtest_return=bt_return,
        return_gap=paper_return - bt_return if np.isfinite(bt_return) else np.nan,
        tracking_error_daily=tracking,
        mean_cost_bps_paper=float(np.nanmean(costs)) if costs else np.nan,
        mean_cost_bps_model=float(model_cost_bps) if model_cost_bps is not None else float(np.nanmean(modelled)) if modelled else np.nan,
        mean_latency_seconds=float(np.nanmean(latency)) if latency else np.nan,
        notes=notes,
    )
