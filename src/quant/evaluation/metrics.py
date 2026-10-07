"""Performance metrics.

Conventions:
- risk-free rate = 0; annualisation uses 365 days (crypto trades every day);
- volatility, Sharpe and Sortino use returns resampled to ``periods`` (daily by
  default) because intraday bar returns are autocorrelated and noisy;
- drawdown uses every bar close (intrabar drawdown is not observed and is larger);
- trade statistics use closed round trips only; open trades are reported separately.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quant.backtesting.result import BacktestResult

DAYS_PER_YEAR = 365.0


def _periods_per_year(periods: str) -> float:
    return DAYS_PER_YEAR * 86_400 / pd.Timedelta(periods).total_seconds()


def equity_with_origin(equity: pd.Series, initial_capital: float, bar: pd.Timedelta) -> pd.Series:
    """Re-stamp marks at bar close time (open + bar) and prepend the initial capital at the start."""
    marks = pd.Series(equity.to_numpy(), index=equity.index + bar)
    origin = pd.Series([initial_capital], index=[equity.index[0]])
    return pd.concat([origin, marks])


def period_returns(result: "BacktestResult", periods: str = "1D") -> pd.Series:
    """Returns resampled to ``periods`` (label = period end), first period measured from initial capital."""
    from quant.data.timeframes import interval_to_timedelta

    equity = result.equity.dropna()
    eq0 = equity_with_origin(equity, result.config.initial_capital, interval_to_timedelta(result.interval))
    sampled = eq0.resample(periods, label="right", closed="right").last().dropna()
    return sampled.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    if dd.empty:
        return 0.0, None, None
    trough = dd.idxmin()
    peak_time = equity.loc[:trough].idxmax()
    return float(dd.min()), peak_time, trough


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd > 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = math.sqrt(float((np.minimum(r, 0.0) ** 2).mean()))
    if downside == 0:
        return float("nan")
    return float(r.mean() / downside * math.sqrt(periods_per_year))


def trade_statistics(trades: pd.DataFrame) -> dict[str, Any]:
    closed = trades[~trades["is_open"]] if len(trades) else trades
    n = len(closed)
    stats: dict[str, Any] = {
        "n_trades": n,
        "n_open_trades": int(trades["is_open"].sum()) if len(trades) else 0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "expectancy": float("nan"),
        "expectancy_pct_equity": float("nan"),
        "avg_trade_pct_equity": float("nan"),
        "avg_win_pct_equity": float("nan"),
        "avg_loss_pct_equity": float("nan"),
        "best_trade_pct_equity": float("nan"),
        "worst_trade_pct_equity": float("nan"),
        "avg_bars_held": float("nan"),
        "n_liquidations": int((trades["exit_reason"] == "liquidation").sum()) if len(trades) else 0,
    }
    if n == 0:
        return stats
    pnl = closed["pnl"]
    roe = closed["return_on_equity"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = -losses.sum()
    stats.update({
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy": float(pnl.mean()),
        "expectancy_pct_equity": float(roe.mean()),
        "avg_trade_pct_equity": float(roe.mean()),
        "avg_win_pct_equity": float(roe[pnl > 0].mean()) if len(wins) else float("nan"),
        "avg_loss_pct_equity": float(roe[pnl < 0].mean()) if len(losses) else float("nan"),
        "best_trade_pct_equity": float(roe.max()),
        "worst_trade_pct_equity": float(roe.min()),
        "avg_bars_held": float(closed["bars_held"].mean()),
    })
    return stats


def compute_metrics(result: "BacktestResult", periods: str = "1D") -> dict[str, Any]:
    from quant.data.timeframes import interval_to_timedelta

    cfg = result.config
    bar = interval_to_timedelta(result.interval)
    equity = result.equity.dropna()
    if equity.empty:
        raise ValueError("empty equity curve")
    initial = cfg.initial_capital
    final = float(equity.iloc[-1])
    eq0 = equity_with_origin(equity, initial, bar)

    span_days = (equity.index[-1] + bar - equity.index[0]).total_seconds() / 86_400
    years = span_days / DAYS_PER_YEAR
    total_return = final / initial - 1.0
    cagr = (final / initial) ** (1.0 / years) - 1.0 if final > 0 and years > 0 else -1.0

    period_rets = period_returns(result, periods)
    ppy = _periods_per_year(periods)
    vol = float(period_rets.std(ddof=1) * math.sqrt(ppy)) if len(period_rets) > 1 else float("nan")
    mdd, dd_peak, dd_trough = max_drawdown(eq0)

    bars = result.bars
    avg_equity = float(equity.mean())
    traded = float(bars["traded_notional"].sum())
    fees = float(bars["fees"].sum())
    slippage = float(bars["spread_slippage"].sum())
    funding = float(bars["funding"].sum())

    metrics: dict[str, Any] = {
        "name": result.name,
        "start": equity.index[0],
        "end": equity.index[-1] + bar,
        "years": years,
        "initial_capital": initial,
        "final_equity": final,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe_ratio(period_rets, ppy),
        "sortino": sortino_ratio(period_rets, ppy),
        "max_drawdown": mdd,
        "max_drawdown_peak": dd_peak,
        "max_drawdown_trough": dd_trough,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "skew": float(period_rets.skew()) if len(period_rets) > 2 else float("nan"),
        "kurtosis": float(period_rets.kurt()) if len(period_rets) > 3 else float("nan"),
        "turnover_annual": traded / avg_equity / years if years > 0 and avg_equity > 0 else float("nan"),
        "fees": fees,
        "spread_slippage": slippage,
        "funding": funding,
        "total_costs": fees + slippage + funding,
        "costs_pct_initial": (fees + slippage + funding) / initial,
        "exposure_time": float((bars["gross_exposure"] > 1e-9).mean()),
        "avg_gross_exposure": float(bars["gross_exposure"].mean()),
        "avg_net_exposure": float(bars["net_exposure"].mean()),
        "n_partial_fills": result.n_partial_fills,
        "maker_fill_share": (result.n_maker_fills / (result.n_maker_fills + result.n_taker_fills)
                             if (result.n_maker_fills + result.n_taker_fills) else float("nan")),
        "n_unfilled_order_bars": result.n_unfilled_order_bars,
        "liquidated": result.liquidated_at is not None,
        "ruined": result.ruined_at is not None,
    }
    metrics.update(trade_statistics(result.trades))
    return metrics


def compare(results: list["BacktestResult"], columns: list[str] | None = None, periods: str = "1D") -> pd.DataFrame:
    """Side-by-side metrics table (one row per result)."""
    default = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar",
               "n_trades", "win_rate", "profit_factor", "expectancy_pct_equity", "worst_trade_pct_equity",
               "avg_bars_held", "turnover_annual", "fees", "spread_slippage", "funding",
               "exposure_time", "avg_gross_exposure", "liquidated"]
    rows = {r.name: r.metrics(periods) for r in results}
    table = pd.DataFrame(rows).T
    return table[columns or default]
