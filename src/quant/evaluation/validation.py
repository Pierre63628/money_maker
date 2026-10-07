"""Out-of-sample validation and robustness tooling.

- ``walk_forward``: choose among candidate strategies on a trailing train window, trade
  the winner on the following test window, repeat. The stitched OOS weights are run
  through the engine once, so switching costs between folds are paid.
- ``parameter_sensitivity``: one-at-a-time perturbation of numeric parameters.
- ``regime_labels`` / ``performance_by_regime``: bull/bear x high/low volatility, causal labels.
- ``cost_stress``: same weights under scaled costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from quant.backtesting.engine import BacktestConfig, run_backtest
from quant.backtesting.result import BacktestResult
from quant.data.market_data import MarketData
from quant.evaluation.metrics import period_returns, sharpe_ratio, sortino_ratio
from quant.evaluation.stats import bootstrap_performance, deflated_sharpe_ratio, minimum_track_record_length, \
    probabilistic_sharpe_ratio
from quant.strategies.base import Strategy


# --------------------------------------------------------------------------- helpers
def slice_result(result: BacktestResult, start=None, end=None) -> BacktestResult:
    """Restrict a result to [start, end); equity just before ``start`` becomes the initial capital."""
    bars = result.bars
    mask = np.ones(len(bars), dtype=bool)
    if start is not None:
        mask &= bars.index >= start
    if end is not None:
        mask &= bars.index < end
    sub = bars.loc[mask]
    if sub.empty:
        raise ValueError("empty slice")
    before = bars.loc[bars.index < sub.index[0], "equity"]
    initial = float(before.iloc[-1]) if len(before) else result.config.initial_capital
    trades = result.trades
    if len(trades):
        trades = trades[(trades["entry_time"] >= sub.index[0]) & (trades["entry_time"] <= sub.index[-1])]
    return replace(result, bars=sub, trades=trades, config=replace(result.config, initial_capital=initial))


def calendar_returns(result: BacktestResult, freq: str = "YE") -> pd.Series:
    daily = period_returns(result, "1D")
    covered_day = (daily.index - pd.Timedelta(days=1)).tz_localize(None)  # label D covers the day before D
    return (1 + daily).groupby(covered_day.to_period(freq[0])).prod() - 1


def _score(daily: pd.Series, metric: str) -> float:
    if len(daily) < 20 or daily.std() == 0:
        return -math.inf
    if metric == "sharpe":
        return sharpe_ratio(daily, 365.0)
    if metric == "sortino":
        return sortino_ratio(daily, 365.0)
    growth = float((1 + daily).prod())
    years = len(daily) / 365.0
    cagr = growth ** (1 / years) - 1 if growth > 0 else -1.0
    if metric == "cagr":
        return cagr
    if metric == "calmar":
        curve = (1 + daily).cumprod()
        mdd = float((curve / curve.cummax() - 1).min())
        return cagr / abs(mdd) if mdd < 0 else math.inf
    raise ValueError(f"unknown metric {metric}")


# ----------------------------------------------------------------------- walk forward
@dataclass(frozen=True)
class WalkForwardSpec:
    train_days: int = 365
    test_days: int = 90
    anchored: bool = False
    selection_metric: str = "sharpe"


@dataclass
class WalkForwardResult:
    oos: BacktestResult
    oos_start: pd.Timestamp
    folds: pd.DataFrame
    candidate_daily_returns: pd.DataFrame
    candidate_names: list[str]

    def oos_only(self) -> BacktestResult:
        return slice_result(self.oos, start=self.oos_start)

    @property
    def selection_stability(self) -> float:
        """Share of folds that kept the previous fold's choice."""
        chosen = self.folds["chosen"].tolist()
        if len(chosen) < 2:
            return float("nan")
        return float(np.mean([a == b for a, b in zip(chosen[:-1], chosen[1:])]))


def walk_forward(candidates: Sequence[Strategy], data: MarketData, config: BacktestConfig,
                 spec: WalkForwardSpec = WalkForwardSpec(), name: str = "walk_forward",
                 precomputed: dict[str, tuple[pd.DataFrame, BacktestResult]] | None = None) -> WalkForwardResult:
    """``precomputed`` maps candidate name -> (weights, full-period result) to avoid recomputation."""
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    precomputed = precomputed or {}
    weights, results = {}, {}
    for c in candidates:
        if c.name in precomputed:
            weights[c.name], results[c.name] = precomputed[c.name]
        else:
            weights[c.name] = c.target_weights(data).reindex(index=data.index, columns=data.symbols).fillna(0.0)
            results[c.name] = run_backtest(weights[c.name], data, config, name=c.name)
    daily = pd.DataFrame({n: period_returns(r, "1D") for n, r in results.items()})

    first_day = data.index[0].normalize()
    last = data.index[-1]
    test_start = first_day + pd.Timedelta(days=spec.train_days)
    stitched = pd.DataFrame(0.0, index=data.index, columns=data.symbols)
    rows = []
    while test_start < last:
        test_end = min(test_start + pd.Timedelta(days=spec.test_days), last + pd.Timedelta(seconds=1))
        train_start = first_day if spec.anchored else test_start - pd.Timedelta(days=spec.train_days)
        # daily return labelled D covers the day ending at D: train uses labels in (train_start, test_start]
        train = daily.loc[(daily.index > train_start) & (daily.index <= test_start)]
        scores = {n: _score(train[n].dropna(), spec.selection_metric) for n in names}
        chosen = max(names, key=lambda n: scores[n])
        mask = (data.index >= test_start) & (data.index < test_end)
        stitched.loc[mask] = weights[chosen].loc[mask].to_numpy()
        test = daily.loc[(daily.index > test_start) & (daily.index <= test_end), chosen]
        rows.append({"test_start": test_start, "test_end": test_end, "chosen": chosen,
                     "train_score": scores[chosen], "test_return_standalone": float((1 + test).prod() - 1),
                     "test_sharpe_standalone": _score(test, "sharpe") if len(test) >= 20 else float("nan")})
        test_start = test_end
    oos = run_backtest(stitched, data, config, name=name)
    first_test = rows[0]["test_start"] if rows else data.index[-1]
    return WalkForwardResult(oos=oos, oos_start=first_test, folds=pd.DataFrame(rows),
                             candidate_daily_returns=daily, candidate_names=names)


# ------------------------------------------------------------------ sensitivity
def parameter_sensitivity(make: Callable[..., Strategy], base: Strategy, data: MarketData, config: BacktestConfig,
                          rel_steps: Sequence[float] = (-0.5, -0.25, 0.25, 0.5)) -> pd.DataFrame:
    """Sharpe/CAGR/drawdown when each numeric parameter is moved by ``rel_steps`` (others fixed)."""
    base_params = {f.name: getattr(base, f.name) for f in fields(base)}

    def evaluate(params: dict, label: str, value) -> dict:
        try:
            strat = make(**params)
            res = run_backtest(strat.target_weights(data), data, config, name=strat.name)
        except ValueError as exc:
            return {"param": label, "value": value, "error": str(exc)}
        m = res.metrics()
        return {"param": label, "value": value, "sharpe": m["sharpe"], "cagr": m["cagr"],
                "max_drawdown": m["max_drawdown"], "n_trades": m["n_trades"], "turnover_annual": m["turnover_annual"]}

    rows = [evaluate(base_params, "base", None)]
    for key, val in base_params.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        for step in rel_steps:
            new = val * (1 + step)
            new = max(1, int(round(new))) if isinstance(val, int) else new
            if new == val:
                continue
            rows.append(evaluate({**base_params, key: new}, key, new))
    return pd.DataFrame(rows)


def sensitivity_summary(table: pd.DataFrame) -> dict:
    base = table.loc[table["param"] == "base", "sharpe"].iloc[0]
    neighbours = table.loc[table["param"] != "base", "sharpe"].dropna()
    if neighbours.empty:
        return {"base_sharpe": base, "n_neighbours": 0}
    return {"base_sharpe": float(base), "n_neighbours": int(len(neighbours)),
            "share_neighbours_sharpe_gt_0": float((neighbours > 0).mean()),
            "median_neighbour_sharpe": float(neighbours.median()),
            "worst_neighbour_sharpe": float(neighbours.min())}


# ---------------------------------------------------------------------- regimes
def regime_labels(reference_close: pd.Series, trend_days: int = 200, vol_days: int = 30) -> pd.DataFrame:
    """Daily labels known at the start of each UTC day (computed from closes up to the previous day)."""
    daily_close = reference_close.resample("1D").last().ffill()
    sma = daily_close.rolling(trend_days, min_periods=trend_days).mean()
    trend = np.where(daily_close > sma, "bull", "bear")
    vol = np.log(daily_close).diff().rolling(vol_days, min_periods=vol_days).std()
    vol_median = vol.expanding(min_periods=vol_days).median()
    vol_label = np.where(vol > vol_median, "high_vol", "low_vol")
    labels = pd.DataFrame({"trend": trend, "vol": vol_label}, index=daily_close.index)
    labels[sma.isna().to_numpy()] = np.nan
    labels.loc[vol_median.isna().to_numpy(), "vol"] = np.nan
    return labels.shift(1)


def performance_by_regime(daily_returns: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    day = (daily_returns.index - pd.Timedelta(days=1)).floor("1D")
    aligned = labels.reindex(day)
    aligned.index = daily_returns.index
    frame = pd.DataFrame({"r": daily_returns, "trend": aligned["trend"], "vol": aligned["vol"]}).dropna()
    frame["regime"] = frame["trend"] + "/" + frame["vol"]
    rows = []
    for key, group in [*frame.groupby("trend"), *frame.groupby("vol"), *frame.groupby("regime")]:
        r = group["r"]
        rows.append({"regime": key, "share_days": len(r) / len(frame), "ann_return": r.mean() * 365,
                     "ann_vol": r.std() * math.sqrt(365), "sharpe": sharpe_ratio(r, 365.0)})
    return pd.DataFrame(rows).set_index("regime")


# ------------------------------------------------------------------ cost stress
def cost_stress(weights: pd.DataFrame, data: MarketData, config: BacktestConfig,
                factors: Sequence[float] = (1.0, 2.0, 3.0), name: str = "strategy") -> pd.DataFrame:
    rows = {}
    for k in factors:
        cfg = config.stressed(k)
        m = run_backtest(weights, data, cfg, name=name).metrics()
        rows[f"x{k:g}"] = {key: m[key] for key in ("total_return", "cagr", "sharpe", "max_drawdown", "fees",
                                                  "spread_slippage", "funding")}
    return pd.DataFrame(rows).T


# -------------------------------------------------------------- full report
def robustness_summary(result: BacktestResult, n_trials: int, trial_daily_returns: pd.DataFrame | None = None,
                       reference_close: pd.Series | None = None, bootstrap_samples: int = 2000) -> dict:
    daily = period_returns(result, "1D")
    trial_sharpes = None
    if trial_daily_returns is not None and trial_daily_returns.shape[1] > 1:
        trial_sharpes = (trial_daily_returns.mean() / trial_daily_returns.std()).to_numpy()
    out = {
        "metrics": result.metrics(),
        "psr_vs_0": probabilistic_sharpe_ratio(daily.to_numpy()),
        "deflated_sharpe": deflated_sharpe_ratio(daily.to_numpy(), n_trials, trial_sharpes),
        "min_track_record_days": minimum_track_record_length(daily.to_numpy()),
        "bootstrap": bootstrap_performance(daily, n_samples=bootstrap_samples),
        "calendar_returns": calendar_returns(result),
    }
    if reference_close is not None:
        out["regimes"] = performance_by_regime(daily, regime_labels(reference_close))
    return out
