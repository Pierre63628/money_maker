"""Paper trading on live public Binance data. No account, no API key, no real orders.

Examples:
    .venv/bin/python scripts/paper_trade.py --experiment multisleeve_top30_4h --once --shadow
    .venv/bin/python scripts/paper_trade.py --experiment multisleeve_top30_4h --loop --shadow
    touch var/paper/<strategy_id>/KILL      # emergency stop: flattens paper positions on the next step
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.brokers.paper import PaperBroker  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.timeframes import interval_to_timedelta  # noqa: E402
from quant.live.feed import BinancePublicFeed  # noqa: E402
from quant.live.loop import LoopConfig, TradingLoop  # noqa: E402
from quant.research.experiments import EXPERIMENTS  # noqa: E402
from quant.risk.engine import RiskLimits  # noqa: E402


def _metric_fields(strategy) -> list[str] | None:
    """Positioning columns the strategy reads, so the feed only queries those endpoints (one request
    per symbol instead of five). None = every endpoint, the safe default for anything unknown."""
    ratio = getattr(strategy, "ratio", None)
    return [ratio] if isinstance(ratio, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--candidate", type=int, default=0, help="index of the candidate within the experiment")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--shadow", action="store_true",
                        help="required for strategies that did not pass the promotion gates (infrastructure test)")
    parser.add_argument("--history-days", type=int, default=240)
    parser.add_argument("--rebalance", choices=("on_change", "drift"), default="on_change",
                        help="'on_change' = the validated behaviour; 'drift' = hold every position at its "
                             "target weight (candidate, forward-tested in parallel)")
    parser.add_argument("--broker", choices=("paper", "testnet"), default="paper",
                        help="'testnet' routes orders to the Binance futures testnet (fake money, real API)")
    parser.add_argument("--credentials", default=str(ROOT / "var" / "secrets" / "binance_testnet.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    strategy = exp.candidates()[args.candidate]
    registry = ROOT / "var" / "experiments" / "registry.jsonl"
    promoted = False
    if registry.exists():
        from quant.evaluation.registry import ExperimentRegistry
        reg = ExperimentRegistry(registry).load()
        wf = reg[(reg["experiment"] == exp.name) & (reg["stage"] == "walk_forward")]
        promoted = bool(len(wf)) and wf.iloc[-1]["metrics"].get("verdict") == "PROMOTE_TO_PAPER"
    if not promoted and not args.shadow:
        raise SystemExit(f"{exp.name} was not promoted by the research gates; pass --shadow to run it as an "
                         "infrastructure test (results are not evidence of an edge)")

    store = ParquetStore(Path(args.data_dir))
    classes = pd.read_csv(store.root / "binance" / "um" / "symbol_classification.csv", index_col=0)
    candidates = sorted(classes.index[classes["is_crypto"].astype(bool) & (classes["status"] == "TRADING")])
    strategy_id = (f"{exp.name}-c{args.candidate}" + ("-shadow" if not promoted else "")
                   + ("-drift" if args.rebalance == "drift" else "")
                   + ("-testnet" if args.broker == "testnet" else ""))
    state_dir = ROOT / "var" / "paper" / strategy_id
    cfg = LoopConfig(strategy_id=strategy_id, strategy=strategy, interval=exp.interval, universe=exp.universe,
                     candidate_symbols=candidates, history_days=args.history_days,
                     risk_limits=RiskLimits(max_gross_leverage=exp.config.max_gross_leverage, max_net_leverage=1.0,
                                            max_position_weight=0.25, max_daily_loss=0.06, max_drawdown=0.30),
                     shadow_note="" if promoted else "SHADOW: strategy not promoted; forward test only",
                     pair_mode=exp.data_loader == "carry_pairs", with_metrics=exp.with_metrics,
                     metric_fields=_metric_fields(strategy), rebalance_mode=args.rebalance)
    feed = BinancePublicFeed(store, ROOT / "var" / "live_cache")
    if args.broker == "testnet":   # its own journal, equity history and order ids
        from quant.brokers.binance_futures import BinanceFuturesBroker
        broker = BinanceFuturesBroker(args.credentials, state_dir / "broker", strategy_id=strategy_id)
        print(f"[{strategy_id}] routing orders to the Binance futures TESTNET "
              f"(balance {broker.equity():,.2f} USDT)")
    else:
        broker = PaperBroker(state_dir / "broker", exp.config.costs, initial_capital=args.capital)
    loop = TradingLoop(cfg, feed, broker, state_dir)

    if args.once or not args.loop:
        print(loop.step())
        return 0
    bar = interval_to_timedelta(exp.interval)
    while True:
        try:
            print(loop.step(), flush=True)
        except Exception as exc:  # keep the loop alive; the journal records the failure
            loop.journal.log("error", error=repr(exc))
        now = pd.Timestamp.now(tz="UTC")
        wake = now.floor(bar) + bar + pd.Timedelta(seconds=60)
        time.sleep(max(30.0, (wake - now).total_seconds()))


if __name__ == "__main__":
    raise SystemExit(main())
