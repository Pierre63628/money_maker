"""Summarise a paper-trading run: equity, fills, slippage, latency, spread-model calibration, risk actions.

Example:
    .venv/bin/python scripts/paper_report.py var/paper/multisleeve_top30_4h-c0-shadow
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_dir")
    args = parser.parse_args()
    state_dir = Path(args.state_dir)
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    steps = pd.DataFrame([e for e in events if e["event"] == "step_summary"])
    decisions = pd.DataFrame([e for e in events if e["event"] == "risk_decision"])
    alerts = [e for e in events if e["event"] in {"alert", "kill_switch", "reconciliation_mismatch", "error"}]
    fills = pd.DataFrame([f for e in events if e["event"] == "step_summary" for f in e.get("fills", [])])

    print(f"== {state_dir.name}")
    if steps.empty:
        print("no steps yet")
        return 0
    print(f"steps: {len(steps)} ({steps['decision_bar'].iloc[0]} -> {steps['decision_bar'].iloc[-1]}); "
          f"note: {steps['note'].iloc[-1] or '-'}")
    print(f"equity: first {steps['equity'].iloc[0]:.2f} last {steps['equity'].iloc[-1]:.2f}; "
          f"gross exposure last {steps['gross_exposure'].iloc[-1]:.2f}; funding paid {steps['funding_paid'].sum():.2f}")
    if len(fills):
        extra = pd.json_normalize(fills["extra"])
        fills = pd.concat([fills.drop(columns=["extra"]), extra], axis=1)
        notional = fills["quantity"] * fills["price"]
        print(f"fills: {len(fills)}, notional {notional.sum():,.0f}, fees {fills['fee'].sum():.2f}")
        print(f"execution cost vs mid (bps): mean {fills['cost_bps'].mean():.2f}, notional-weighted "
              f"{np.average(fills['cost_bps'], weights=notional):.2f}")
        print(f"slippage vs decision close (bps): mean {fills['slippage_vs_decision_close_bps'].mean():.1f}, "
              f"abs mean {fills['slippage_vs_decision_close_bps'].abs().mean():.1f} "
              f"(includes price drift during latency; mean latency {fills['latency_seconds'].mean():.0f}s)")
        ratio = fills["half_spread_bps_used"] / fills["half_spread_bps_modelled"]
        print(f"spread calibration: observed/modelled half spread median {ratio.median():.2f} "
              f"(p90 {ratio.quantile(0.9):.2f}); >1 means the backtest spread model is too optimistic")
    if len(decisions):
        print("risk actions:", decisions["action"].value_counts().to_dict())
        reasons = decisions["reasons"].explode().dropna()
        if len(reasons):
            print("top risk reasons:", reasons.str.split(" ").str[:4].str.join(" ").value_counts().head(5).to_dict())
    print(f"alerts: {len(alerts)}")
    for a in alerts[-5:]:
        print("  ", a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
