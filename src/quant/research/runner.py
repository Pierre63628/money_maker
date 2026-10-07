"""Run a pre-registered experiment end to end and write an auditable report.

Pipeline: point-in-time universe -> look-ahead check -> every candidate in-sample
(logged) -> PBO across candidates -> walk-forward selection (OOS) -> deflated Sharpe,
bootstrap, cost stress, regimes, benchmarks -> promotion gates -> optional one-shot holdout.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant.backtesting.engine import run_backtest
from quant.data.store import ParquetStore
from quant.evaluation.lookahead import check_no_lookahead
from quant.evaluation.metrics import period_returns
from quant.evaluation.registry import ExperimentRegistry
from quant.evaluation.stats import deflated_sharpe_ratio, pbo_cscv
from quant.evaluation.validation import calendar_returns, robustness_summary, slice_result, walk_forward
from quant.portfolio.sizing import hold_between_rebalances, normalise_by_count
from quant.research.experiments import Experiment
from quant.research.universe_data import load_universe

KEY_METRICS = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "n_trades",
               "win_rate", "profit_factor", "turnover_annual", "fees", "spread_slippage", "funding", "exposure_time",
               "maker_fill_share"]


def _fmt(v, pct=False, digits=2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v * 100:.{digits - 1}f}%" if pct else f"{v:.{digits}f}"


def _metrics_row(m: dict) -> dict:
    return {k: m.get(k) for k in KEY_METRICS}


INTEGER_COLS = {"n_trades"}


def _markdown_table(df: pd.DataFrame, pct_cols: set[str] = frozenset()) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [row[c] if isinstance(row[c], str)
                              else _fmt(int(row[c]) if c in INTEGER_COLS and pd.notna(row[c]) else row[c],
                                        pct=c in pct_cols)
                              for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_experiment(exp: Experiment, store: ParquetStore, registry: ExperimentRegistry, out_dir: Path,
                   holdout: bool = False, bootstrap_samples: int = 2000, log: Callable[[str], None] = print) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir) / exp.name
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_start = pd.Timestamp(exp.holdout_start, tz="UTC")
    end = exp.holdout_end if holdout else exp.holdout_start
    if holdout and not registry.load().empty:
        prior = registry.load()
        if ((prior["experiment"] == exp.name) & (prior["stage"] == "holdout")).any():
            log("WARNING: holdout already evaluated for this experiment; this re-run is NOT an independent test")

    if exp.data_loader == "carry_pairs":
        from quant.research.carry_data import load_carry_pairs
        data = load_carry_pairs(store, exp.universe, exp.interval, exp.start, end, exp.source_interval)
    else:
        data = load_universe(store, exp.universe, exp.interval, exp.start, end, exp.source_interval,
                             with_order_flow=exp.with_order_flow, with_metrics=exp.with_metrics)
    if exp.with_metrics:
        oi = data.extras.get("open_interest") if data.extras else None
        if oi is None:
            raise RuntimeError(f"{exp.name} needs metrics but none were loaded (run scripts/download_metrics.py)")
        covered = oi.notna().any(axis=0)
        missing = [s for s in data.symbols if not covered.get(s, False)]
        share = len(missing) / max(len(data.symbols), 1)
        log(f"[{exp.name}] open interest available for {len(data.symbols) - len(missing)}/{len(data.symbols)} symbols")
        if share > 0.2:
            raise RuntimeError(f"{exp.name}: open interest missing for {share:.0%} of the universe; "
                               f"download it before drawing conclusions (missing e.g. {missing[:5]})")
    n_eligible = data.universe.sum(axis=1)
    log(f"[{exp.name}] {len(data.symbols)} symbols ever eligible, {len(data)} {exp.interval} bars, "
        f"eligible per bar median {int(n_eligible.median())} (min {int(n_eligible.min())}) [{time.time() - t0:.0f}s]")

    candidates = exp.candidates()
    sample = data.head(min(len(data), 2500))
    for cls in {type(c) for c in candidates}:
        probe = next(c for c in candidates if type(c) is cls)
        report = check_no_lookahead(probe, sample, n_cuts=3, min_bars=len(sample) // 2)
        if not report.passed:
            raise RuntimeError(f"look-ahead detected, aborting: {report}")
    log(f"[{exp.name}] look-ahead check passed for {len({type(c) for c in candidates})} strategy class(es)")

    cfg = exp.config
    precomputed, rows, research_daily = {}, [], {}
    for i, c in enumerate(candidates, 1):
        w = c.target_weights(data).reindex(index=data.index, columns=data.symbols).fillna(0.0)
        res = run_backtest(w, data, cfg, name=c.name)
        research = slice_result(res, end=holdout_start) if holdout else res
        m = research.metrics()
        precomputed[c.name] = (w, res)
        research_daily[c.name] = period_returns(research, "1D")
        rows.append({"strategy": c.name, **_metrics_row(m)})
        if not holdout:
            registry.log(exp.name, c.name, c.family, c.provenance, c.params,
                         {"interval": exp.interval, "universe": asdict(exp.universe), "start": exp.start, "end": end},
                         _metrics_row(m), stage="in_sample")
        if i % 8 == 0:
            log(f"[{exp.name}] {i}/{len(candidates)} candidates [{time.time() - t0:.0f}s]")
    in_sample = pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)
    in_sample.to_csv(out_dir / "in_sample.csv")
    daily_matrix = pd.DataFrame(research_daily)
    if len(candidates) > 1:
        pbo = pbo_cscv(daily_matrix, n_blocks=16)
    else:  # no selection among candidates: overfitting-by-selection does not apply
        pbo = {"pbo": float("nan"), "n_splits": 0, "median_logit": float("nan"), "prob_oos_sharpe_le_0": float("nan"),
               "is_to_oos_slope": float("nan"), "mean_is_sharpe_of_best": float("nan"),
               "mean_oos_sharpe_of_best": float("nan")}
    log(f"[{exp.name}] PBO={pbo['pbo']:.2f} [{time.time() - t0:.0f}s]")

    wf = walk_forward(candidates, data, cfg, exp.walk_forward, name=f"WF[{exp.name}]", precomputed=precomputed)
    oos = slice_result(wf.oos, start=wf.oos_start, end=holdout_start)
    wf.folds.to_csv(out_dir / "walk_forward_folds.csv", index=False)

    # Deflated Sharpe: N and the cross-trial Sharpe variance must come from the SAME set of trials
    # (every in-sample trial ever logged), per Bailey & Lopez de Prado (2014).
    all_trials = registry.load()
    pooled = pd.Series(dtype=float)
    if len(all_trials):
        ins = all_trials[all_trials["stage"] == "in_sample"].drop_duplicates("strategy", keep="last")
        pooled = pd.Series([m.get("sharpe") for m in ins["metrics"]], dtype=float).dropna() / math.sqrt(365.0)
    n_trials_total = max(len(pooled), len(candidates))
    pooled_trial_sharpes = pooled.to_numpy() if len(pooled) > 1 else None
    btc_close = None
    try:
        btc_close = store.read_klines("um", "BTCUSDT", "1h", exp.start, end, columns=["close"])["close"]
    except FileNotFoundError:
        pass
    robust = robustness_summary(oos, n_trials=n_trials_total, trial_daily_returns=daily_matrix,
                                reference_close=btc_close, bootstrap_samples=bootstrap_samples)
    # Gate (pre-committed spec): N = all logged trials, Sharpe variance across this experiment's candidates
    # (sampling variance when there is a single candidate). Two alternative specs are reported for information
    # only; they must not be used to overturn a verdict after seeing results.
    oos_daily = period_returns(oos, "1D").to_numpy()
    robust["deflated_sharpe_pooled_info"] = deflated_sharpe_ratio(oos_daily, n_trials_total, pooled_trial_sharpes)
    robust["deflated_sharpe_sampling_null_info"] = deflated_sharpe_ratio(oos_daily, n_trials_total, None)
    oos_m = robust["metrics"]

    stress = {}
    for k in (2.0, 3.0):
        stressed = run_backtest(wf.oos.target_weights, data, cfg.stressed(k), name=f"x{k:g}")
        stress[f"x{k:g}"] = _metrics_row(slice_result(stressed, start=wf.oos_start, end=holdout_start).metrics())

    tradable = data.tradable
    ew = hold_between_rebalances(normalise_by_count(tradable.astype(float), tradable), 24, interval=data.interval)
    bench = {
        "equal_weight_universe_long": slice_result(run_backtest(ew, data, cfg, name="EW"), start=wf.oos_start,
                                                   end=holdout_start).metrics(),
    }
    if "BTCUSDT" in data.symbols:
        btc_w = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
        btc_w["BTCUSDT"] = 1.0
        bench["btc_buy_and_hold"] = slice_result(run_backtest(btc_w, data, cfg, name="BTC"), start=wf.oos_start,
                                                 end=holdout_start).metrics()

    g = exp.gates
    gates = {
        f"OOS Sharpe >= {g.min_oos_sharpe}": oos_m["sharpe"] >= g.min_oos_sharpe,
        f"Deflated Sharpe >= {g.min_deflated_sharpe} (N={n_trials_total})": robust["deflated_sharpe"]["dsr"] >= g.min_deflated_sharpe,
        f"PBO <= {g.max_pbo}" + (" (n/a: single candidate)" if len(candidates) < 2 else ""):
            True if len(candidates) < 2 else pbo["pbo"] <= g.max_pbo,
        f"Bootstrap P(Sharpe<=0) <= {g.max_bootstrap_prob_sharpe_le_0}": robust["bootstrap"]["prob_sharpe_le_0"] <= g.max_bootstrap_prob_sharpe_le_0,
        f"OOS Sharpe at 2x costs >= {g.min_oos_sharpe_costs_x2}": (stress["x2"]["sharpe"] or -9) >= g.min_oos_sharpe_costs_x2,
        f"OOS max drawdown >= {g.min_oos_max_drawdown:.0%}": oos_m["max_drawdown"] >= g.min_oos_max_drawdown,
    }
    verdict = "PROMOTE_TO_PAPER" if all(gates.values()) else "REJECT"
    registry.log(exp.name, wf.oos.name, candidates[0].family, "walk_forward_selection",
                 {"walk_forward": asdict(exp.walk_forward)}, {"interval": exp.interval, "start": exp.start, "end": end},
                 {**_metrics_row(oos_m), "dsr": robust["deflated_sharpe"]["dsr"], "pbo": pbo["pbo"], "verdict": verdict},
                 stage="walk_forward")

    holdout_metrics = None
    if holdout:
        hold = slice_result(wf.oos, start=holdout_start, end=pd.Timestamp(exp.holdout_end, tz="UTC"))
        holdout_metrics = hold.metrics()
        registry.log(exp.name, wf.oos.name, candidates[0].family, "walk_forward_selection", {},
                     {"start": exp.holdout_start, "end": exp.holdout_end}, _metrics_row(holdout_metrics), stage="holdout")

    report = _render_report(exp, data, in_sample, pbo, wf, oos_m, robust, stress, bench, gates, verdict,
                            n_trials_total, holdout_metrics, time.time() - t0)
    (out_dir / ("report_holdout.md" if holdout else "report.md")).write_text(report)
    log(f"[{exp.name}] verdict={verdict} OOS sharpe={oos_m['sharpe']:.2f} DSR={robust['deflated_sharpe']['dsr']:.2f} "
        f"PBO={pbo['pbo']:.2f} [{time.time() - t0:.0f}s] -> {out_dir}")
    return {"verdict": verdict, "gates": gates, "oos": oos_m, "pbo": pbo, "robust": robust, "stress": stress,
            "benchmarks": bench, "in_sample": in_sample, "folds": wf.folds, "holdout": holdout_metrics}


def _render_report(exp, data, in_sample, pbo, wf, oos_m, robust, stress, bench, gates, verdict, n_trials,
                   holdout_metrics, elapsed) -> str:
    pct = {"total_return", "cagr", "volatility", "max_drawdown", "win_rate", "exposure_time"}
    lines = [
        f"# {exp.name}",
        "",
        f"**Hypothesis:** {exp.hypothesis}",
        "",
        f"- **Data:** {len(data.symbols)} symbols ever eligible, {exp.interval} bars (from {exp.source_interval}), "
        f"{data.index[0]:%Y-%m-%d} → {exp.holdout_start} (holdout {exp.holdout_start} → {exp.holdout_end}).",
        f"- **Universe:** {asdict(exp.universe)}",
        f"- **Costs:** {asdict(exp.config.costs)}",
        f"- **Execution:** {asdict(exp.config.execution)} (stress x2/x3 also scales maker fee and trade-through)",
        f"- **Walk-forward:** {asdict(exp.walk_forward)}; OOS starts {wf.oos_start:%Y-%m-%d}.",
        f"- **Trials:** {len(in_sample)} in this experiment, {n_trials} logged in total (used for the deflated Sharpe).",
        f"- Runtime {elapsed:.0f}s.",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in gates.items()]
    lines += ["", "## Walk-forward out-of-sample vs benchmarks (same window, same costs)", ""]
    table = pd.DataFrame({"walk_forward_oos": _metrics_row(oos_m), "oos_costs_x2": stress["x2"],
                          "oos_costs_x3": stress["x3"], **{k: _metrics_row(v) for k, v in bench.items()}}).T
    table.index.name = "series"
    lines += [_markdown_table(table[["total_return", "cagr", "volatility", "sharpe", "max_drawdown", "calmar",
                                     "n_trades", "turnover_annual", "fees", "spread_slippage", "funding",
                                     "maker_fill_share"]], pct | {"maker_fill_share"}), ""]
    dsr = robust["deflated_sharpe"]
    b = robust["bootstrap"]
    lines += [
        "## Statistical robustness (OOS daily returns)",
        "",
        f"- **PSR(0):** {robust['psr_vs_0']:.3f}; **minimum track record:** {robust['min_track_record_days']:.0f} days.",
        f"- **Deflated Sharpe (gate, pre-committed spec):** {dsr['dsr']:.3f}. Daily SR {dsr['sr']:.4f} vs threshold "
        f"{dsr['sr_threshold']:.4f} for the best of N={n_trials} trials (variance across this experiment's candidates).",
        f"- *Info only:* DSR with variance pooled over all logged trials "
        f"{robust['deflated_sharpe_pooled_info']['dsr']:.3f} (threshold {robust['deflated_sharpe_pooled_info']['sr_threshold']:.4f}); "
        f"DSR with sampling-noise null variance {robust['deflated_sharpe_sampling_null_info']['dsr']:.3f} "
        f"(threshold {robust['deflated_sharpe_sampling_null_info']['sr_threshold']:.4f}).",
        f"- **Bootstrap (stationary, 10-day blocks):** Sharpe 5/50/95% = "
        f"{', '.join(f'{x:.2f}' for x in b['sharpe_q05_q50_q95'])}; "
        f"CAGR = {', '.join(f'{x:.1%}' for x in b['cagr_q05_q50_q95'])}; "
        f"MDD = {', '.join(f'{x:.1%}' for x in b['max_drawdown_q05_q50_q95'])}; "
        f"P(Sharpe≤0) = {b['prob_sharpe_le_0']:.2f}.",
        f"- **PBO (CSCV, 16 blocks, in-sample candidates):** {pbo['pbo']:.2f}. Median logit {pbo['median_logit']:.2f}; "
        f"P(OOS Sharpe≤0 for the IS winner) = {pbo['prob_oos_sharpe_le_0']:.2f}; IS→OOS slope {pbo['is_to_oos_slope']:.2f}.",
        f"- **Walk-forward selection stability:** {wf.selection_stability:.2f} (share of folds keeping the previous choice).",
        "",
        "## Calendar returns (OOS)",
        "",
        " | ".join(f"{p}: {r:+.1%}" for p, r in robust["calendar_returns"].items()),
        "",
    ]
    if "regimes" in robust:
        reg = robust["regimes"].copy()
        reg.index.name = "regime (BTC)"
        lines += ["## Regimes (BTC trend: 200-day SMA; vol: 30-day vs expanding median)", "",
                  _markdown_table(reg, {"share_days", "ann_return", "ann_vol"}), ""]
    lines += ["## Walk-forward folds", "",
              _markdown_table(wf.folds.assign(test_start=wf.folds["test_start"].dt.strftime("%Y-%m-%d"))
                              .set_index("test_start")[["chosen", "train_score", "test_return_standalone"]],
                              {"test_return_standalone"}), ""]
    top = pd.concat([in_sample.head(8), in_sample.tail(3)])
    top.index.name = "candidate (in-sample, research window)"
    lines += [f"## In-sample candidates (top 8 / bottom 3 of {len(in_sample)}, sorted by Sharpe)", "",
              f"{int((in_sample['sharpe'] > 0).sum())}/{len(in_sample)} candidates have in-sample Sharpe > 0.", "",
              _markdown_table(top[["total_return", "sharpe", "max_drawdown", "n_trades", "turnover_annual"]], pct), ""]
    if holdout_metrics is not None:
        hm = pd.DataFrame({"holdout": _metrics_row(holdout_metrics)}).T
        hm.index.name = "series"
        lines += ["## Holdout (one-shot)", "", _markdown_table(hm, pct), ""]
    return "\n".join(lines)
