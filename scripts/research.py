"""Run pre-registered research experiments (see src/quant/research/experiments.py).

Examples:
    .venv/bin/python scripts/research.py --list
    .venv/bin/python scripts/research.py trend_top10_4h xsmom_top50_4h
    .venv/bin/python scripts/research.py trend_top10_4h --holdout     # one-shot final test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant.data.store import ParquetStore  # noqa: E402
from quant.evaluation.registry import ExperimentRegistry  # noqa: E402
from quant.research.experiments import EXPERIMENTS  # noqa: E402
from quant.research.runner import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="also evaluate the untouched holdout window (once!)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    parser.add_argument("--registry", default=str(ROOT / "var" / "experiments" / "registry.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "var" / "reports" / "research"))
    args = parser.parse_args()

    if args.list or not args.experiments:
        for exp in EXPERIMENTS.values():
            print(f"{exp.name:<18} {exp.interval:<4} {len(exp.candidates()):>3} candidates  {exp.hypothesis}")
        return 0
    store = ParquetStore(Path(args.data_dir))
    registry = ExperimentRegistry(Path(args.registry))
    summary = []
    for name in args.experiments:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name}; use --list")
        out = run_experiment(EXPERIMENTS[name], store, registry, Path(args.out_dir), holdout=args.holdout,
                             bootstrap_samples=args.bootstrap, log=lambda msg: print(msg, flush=True))
        summary.append((name, out["verdict"], out["oos"]["sharpe"], out["robust"]["deflated_sharpe"]["dsr"],
                        out["pbo"]["pbo"]))
    print("\nexperiment          verdict            OOS_sharpe  DSR    PBO")
    for row in summary:
        print(f"{row[0]:<19} {row[1]:<18} {row[2]:>10.2f}  {row[3]:.2f}  {row[4]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
