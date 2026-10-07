"""Run baseline strategies and benchmarks on stored data, with cost stress.

These are textbook parameters, NOT optimised, and the whole period is in-sample.
The output is a reference point for later research, not evidence of an edge.

Example:
    .venv/bin/python scripts/run_baselines.py --interval 1h --start 2020-01-01 --end 2026-09-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.backtesting.costs import CostModel  # noqa: E402
from quant.backtesting.engine import BacktestConfig, run_backtest  # noqa: E402
from quant.data.market_data import load_market_data  # noqa: E402
from quant.data.store import ParquetStore  # noqa: E402
from quant.data.timeframes import bars_per_day  # noqa: E402
from quant.evaluation.lookahead import check_no_lookahead  # noqa: E402
from quant.evaluation.metrics import compare  # noqa: E402
from quant.strategies.baselines import (BuyAndHold, DonchianBreakout, RsiReversion, SmaCross,  # noqa: E402
                                        TimeSeriesMomentum)

HALF_SPREAD_BPS = {"BTCUSDT": 0.5, "ETHUSDT": 0.5, "SOLUSDT": 1.0}


def strategies_for(interval: str):
    d = int(bars_per_day(interval))
    return [
        BuyAndHold(),
        SmaCross(fast=d, slow=7 * d),
        SmaCross(fast=d, slow=7 * d, long_only=True),
        SmaCross(fast=50 * d // 24 * 24 if d >= 24 else 50, slow=200 * d // 24 * 24 if d >= 24 else 200),
        TimeSeriesMomentum(lookback=7 * d),
        TimeSeriesMomentum(lookback=30 * d),
        DonchianBreakout(entry_window=55, exit_window=20),
        DonchianBreakout(entry_window=10 * d, exit_window=5 * d),
        RsiReversion(window=14, lower=30, upper=70, exit_level=50),
    ]


def yearly_returns(result) -> pd.Series:
    eq = result.equity.copy()
    yearly = eq.resample("YE").last()
    start = pd.Series([result.config.initial_capital], index=[eq.index[0] - pd.Timedelta(seconds=1)])
    return pd.concat([start, yearly]).pct_change().dropna().rename(lambda ts: ts.year)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-09-01", help="exclusive; keep within funding-rate coverage")
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "var" / "reports"))
    parser.add_argument("--stress", nargs="+", type=float, default=[0.0, 1.0, 2.0])
    args = parser.parse_args()

    store = ParquetStore(Path(args.data_dir))
    data = load_market_data(store, args.symbols, args.interval, args.start, args.end)
    print(f"data: {data.symbols} {args.interval} {data.index[0]} -> {data.index[-1]} ({len(data)} bars)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    strategies = strategies_for(args.interval)
    for s in strategies:
        report = check_no_lookahead(s, data.slice(end=data.index[min(len(data) - 1, 20_000)]), n_cuts=4)
        if not report.passed:
            raise SystemExit(f"refusing to backtest: {report}")
    print("look-ahead check: all strategies causal")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    base_costs = CostModel(per_symbol_half_spread_bps=HALF_SPREAD_BPS)
    tables = []
    for factor in args.stress:
        costs = base_costs.scaled(factor) if factor != 1.0 else base_costs
        if factor == 0.0:
            costs = CostModel.zero()
        config = BacktestConfig(costs=costs, max_gross_leverage=1.0)
        results = [run_backtest(s.target_weights(data), data, config, name=s.name) for s in strategies]
        table = compare(results)
        table.insert(0, "cost_x", factor)
        tables.append(table)
        cols = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades",
                "win_rate", "profit_factor", "avg_bars_held", "turnover_annual", "fees", "spread_slippage",
                "funding", "exposure_time"]
        print(f"\n=== costs x{factor} ===")
        print(table[cols].astype(float).round(3).to_string())
        if factor == 1.0:
            yearly = pd.DataFrame({r.name: yearly_returns(r) for r in results}).T
            print("\n--- calendar-year returns (costs x1) ---")
            print(yearly.round(3).to_string())
            yearly.to_csv(out / f"baselines_{args.interval}_yearly.csv")
    pd.concat(tables).to_csv(out / f"baselines_{args.interval}.csv")
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
