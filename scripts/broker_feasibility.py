"""Can a validated strategy still work at a given broker?

Re-runs frozen strategies on the instruments the broker lists, with the broker's spreads, commissions
and overnight swaps, and reports the break-even spread beyond which the edge disappears.

Examples:
    .venv/bin/python scripts/broker_feasibility.py                         # placeholder CFD profile
    .venv/bin/python scripts/broker_feasibility.py --profile my_broker.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.backtesting.costs import CostModel  # noqa: E402
from quant.backtesting.engine import run_backtest  # noqa: E402
from quant.brokers.profiles import TYPICAL_CRYPTO_CFD, BrokerProfile  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.universe import UniverseRule  # noqa: E402
from quant.research.experiments import MAKER_CONFIG  # noqa: E402
from quant.research.universe_data import load_universe  # noqa: E402
from quant.strategies.combine import ParameterEnsemble  # noqa: E402
from quant.strategies.positioning import CrowdingContrarianRobust  # noqa: E402
from quant.strategies.trend import TrendFollowing  # noqa: E402

FROZEN = {
    "crowding_robust": CrowdingContrarianRobust(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=4,
                                                vol_window_days=30, max_weight=0.05, min_age_days=0),
    "trend_ensemble": ParameterEnsemble(tuple(TrendFollowing(lookback_days=lb, band=b, rebalance_hours=24)
                                              for lb in (7, 14, 30, 60) for b in (0.0, 0.5, 1.0)), "trend"),
}


def metrics_row(result) -> dict:
    m = result.metrics()
    return {k: m[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown", "turnover_annual", "n_trades")}


def breakeven_half_spread(weights, data, config, lo: float = 0.0, hi: float = 200.0) -> float:
    """Half spread (bps) at which the annualised Sharpe reaches zero, by bisection."""
    def sharpe(bps: float) -> float:
        costs = replace(config.costs, per_symbol_half_spread_bps={}, half_spread_bps=bps)
        return run_backtest(weights, data, replace(config, costs=costs)).metrics()["sharpe"]

    if sharpe(lo) <= 0:
        return 0.0
    if sharpe(hi) > 0:
        return hi
    for _ in range(18):
        mid = (lo + hi) / 2
        if sharpe(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default=None, help="broker profile JSON (default: the placeholder CFD profile)")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-09-01")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    args = parser.parse_args()

    profile = BrokerProfile.from_json(args.profile) if args.profile else TYPICAL_CRYPTO_CFD
    print(f"broker profile: {profile.name}")
    print(f"  {len(profile.instruments)} instruments, commission {profile.commission_bps_per_side} bps/side, "
          f"slippage {profile.slippage_bps} bps, max leverage {profile.max_leverage}")
    if profile.notes:
        print(f"  note: {profile.notes}")

    store = ParquetStore(Path(args.data_dir))
    universe = load_universe(store, UniverseRule(top_n=50, min_median_quote_volume=5e6), args.interval,
                             args.start, args.end, with_metrics=True)
    available = [s for s in profile.source_symbols if s in universe.symbols]
    missing = [s for s in profile.source_symbols if s not in universe.symbols]
    if missing:
        print(f"  no research data for: {missing}")
    if len(available) < 4:
        raise SystemExit("too few instruments with research data to evaluate a cross-sectional strategy")
    data = universe.select(available)
    print(f"  evaluating on {len(available)} instruments: {', '.join(available)}\n")

    broker_config = profile.backtest_config(args.capital)
    binance_config = replace(MAKER_CONFIG, initial_capital=args.capital)
    rows = {}
    for name, strategy in FROZEN.items():
        weights = strategy.target_weights(data)
        rows[(name, "binance (maker, as researched)")] = metrics_row(run_backtest(weights, data, binance_config))
        rows[(name, f"{profile.name} (spreads+swaps)")] = metrics_row(run_backtest(weights, data, broker_config))
        no_swap = replace(broker_config, costs=replace(broker_config.costs, per_symbol_financing_bps_per_day={},
                                                       financing_bps_per_day=0.0))
        rows[(name, "broker spreads, no swap")] = metrics_row(run_backtest(weights, data, no_swap))
        be = breakeven_half_spread(weights, data, no_swap)
        worst = max((i.half_spread_bps for i in profile.instruments.values()), default=float("nan"))
        print(f"{name}: break-even half spread {be:.1f} bps per side (broker's widest listed: {worst:.1f} bps)")
    table = pd.DataFrame(rows).T
    table.index.names = ["strategy", "scenario"]
    pd.set_option("display.width", 200)
    print("\n" + table.round(3).to_string())
    out = ROOT / "var" / "reports" / f"broker_feasibility_{profile.name.split()[0]}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
