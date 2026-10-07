"""Apply one already-chosen configuration to a window that was never used for research.

This is not a walk-forward: the parameters are frozen, the window is simply earlier (or later) than
the research period. It is a cheap independent check that does not consume the final holdout.
The run is logged to the registry with stage ``pre_sample`` so it counts as evidence, not as a new trial.

Example:
    .venv/bin/python scripts/pre_sample_test.py --experiment crowding_top30_4h --candidate 0 \
        --start 2020-09-01 --end 2022-01-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.backtesting.engine import run_backtest  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.evaluation.metrics import period_returns  # noqa: E402
from quant.evaluation.registry import ExperimentRegistry  # noqa: E402
from quant.evaluation.stats import bootstrap_performance, probabilistic_sharpe_ratio  # noqa: E402
from quant.evaluation.validation import calendar_returns, cost_stress  # noqa: E402
from quant.research.experiments import EXPERIMENTS  # noqa: E402
from quant.research.universe_data import load_universe  # noqa: E402

KEYS = ["total_return", "cagr", "volatility", "sharpe", "max_drawdown", "calmar", "n_trades", "turnover_annual",
        "win_rate", "profit_factor", "exposure_time", "maker_fill_share"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--candidate", type=int, default=0, help="index in the experiment's candidate list")
    parser.add_argument("--by-name", default=None, help="pick the candidate by its full name instead of index")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--stage", default="pre_sample")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    parser.add_argument("--registry", default=str(ROOT / "var" / "experiments" / "registry.jsonl"))
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    candidates = exp.candidates()
    strategy = next(c for c in candidates if c.name == args.by_name) if args.by_name else candidates[args.candidate]
    store = ParquetStore(Path(args.data_dir))
    data = load_universe(store, exp.universe, exp.interval, args.start, args.end, exp.source_interval,
                         with_order_flow=exp.with_order_flow, with_metrics=exp.with_metrics)
    print(f"{strategy.name}\n  window {data.index[0]} -> {data.index[-1]} ({len(data)} bars, "
          f"{len(data.symbols)} symbols)")
    result = run_backtest(strategy.target_weights(data), data, exp.config, name=strategy.name)
    metrics = result.metrics()
    daily = period_returns(result, "1D")
    print("  " + "  ".join(f"{k}={metrics[k]:.3f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}"
                           for k in KEYS))
    print(f"  PSR(0)={probabilistic_sharpe_ratio(daily.to_numpy()):.3f}")
    if len(daily) >= 30:
        boot = bootstrap_performance(daily, n_samples=1000)
        print(f"  bootstrap sharpe 5/50/95% = {', '.join(f'{x:.2f}' for x in boot['sharpe_q05_q50_q95'])}, "
              f"P(sharpe<=0)={boot['prob_sharpe_le_0']:.2f}")
    print("  calendar: " + " | ".join(f"{p}: {r:+.1%}" for p, r in calendar_returns(result).items()))
    stress = cost_stress(result.target_weights, data, exp.config, factors=(1.0, 2.0, 3.0), name=strategy.name)
    print("  cost stress sharpe: " + ", ".join(f"{idx}={row['sharpe']:.2f}" for idx, row in stress.iterrows()))

    ExperimentRegistry(Path(args.registry)).log(
        exp.name, strategy.name, strategy.family, strategy.provenance, strategy.params,
        {"interval": exp.interval, "universe": asdict(exp.universe), "start": args.start, "end": args.end},
        {k: metrics[k] for k in KEYS}, stage=args.stage,
        notes="frozen configuration applied to a window outside the research period")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
