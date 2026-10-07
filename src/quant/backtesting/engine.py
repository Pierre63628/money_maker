"""Bar-based portfolio backtester for target-weight strategies.

Timing contract (enforced, not optional):
- ``weights.loc[t]`` is a decision taken with information up to the close of bar ``t``;
- it is executed at the open of bar ``t + execution_delay`` (delay >= 1).

Rebalancing (``rebalance``):
- ``"on_change"`` (default): trade a symbol only when its target weight changes
  (or when leverage drifts above the cap). Holdings otherwise drift with prices,
  which is how a real position behaves (buy & hold stays buy & hold);
- ``"every_bar"``: re-target weights each bar when the drift exceeds ``rebalance_threshold``.

Intrabar stops/limits are NOT modelled here (see the event-driven engine);
strategies that need them must not be approximated with this engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

from quant.backtesting import _core
from quant.backtesting.costs import CostModel
from quant.backtesting.result import BacktestResult
from quant.data.market_data import MarketData
from quant.data.timeframes import bars_per_day


@dataclass(frozen=True)
class ExecutionModel:
    """How orders reach the market.

    ``taker``: market orders at the next open, paying spread, slippage, impact and taker fee.
    ``maker``: limit orders at the decision close -/+ ``limit_offset_bps``. A fill requires the
    next bar to trade through the limit by ``trade_through_bps`` (a haircut for queue position);
    the fill is at the limit price with ``maker_fee_bps``. Adverse selection is inherent: buys fill
    only when price falls through the limit. Unfilled orders keep working (re-quoted every bar);
    risk-reducing orders cross the spread after ``maker_fallback_bars`` unfilled bars.
    """

    mode: Literal["taker", "maker"] = "taker"
    maker_fee_bps: float = 2.0
    limit_offset_bps: float = 1.0
    trade_through_bps: float = 2.0
    maker_fallback_bars: int = 3
    per_symbol_maker_fee_bps: dict = field(default_factory=dict)

    def maker_fee_bps_for(self, symbol: str) -> float:
        from quant.backtesting.costs import _lookup
        return _lookup(self.per_symbol_maker_fee_bps, symbol, self.maker_fee_bps)

    def stressed(self, factor: float) -> "ExecutionModel":
        return replace(self, maker_fee_bps=self.maker_fee_bps * factor, trade_through_bps=self.trade_through_bps * factor,
                       per_symbol_maker_fee_bps={k: v * factor for k, v in self.per_symbol_maker_fee_bps.items()})


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    execution_delay: int = 1
    costs: CostModel = field(default_factory=CostModel)
    rebalance: Literal["on_change", "every_bar"] = "on_change"
    rebalance_threshold: float = 0.05
    allow_short: bool = True
    max_gross_leverage: float = 1.0
    leverage_tolerance: float = 0.10
    max_participation: float = 0.05      # share of lagged average bar quote volume per fill
    volume_lookback: int = 24            # bars for the average volume used by the liquidity cap
    vol_lookback: int = 96               # bars for the volatility used by the impact model
    sigma_fallback: float = 0.02         # bar volatility assumed before enough history exists
    maintenance_margin: float = 0.005    # fraction of notional
    liquidation_fee: float = 0.0125      # fraction of notional charged at liquidation
    min_trade_notional: float = 5.0      # exchange minimum order notional (quote currency)
    delist_penalty_bps: float = 100.0    # exit haircut when a symbol's data ends
    max_gap_bars: int = 72               # a longer data gap is treated as a delisting (forced exit before it)
    execution: ExecutionModel = field(default_factory=ExecutionModel)

    def stressed(self, factor: float) -> "BacktestConfig":
        """Every friction multiplied by ``factor`` (costs, maker fee, queue haircut)."""
        return replace(self, costs=self.costs.scaled(factor), execution=self.execution.stressed(factor))

    def __post_init__(self) -> None:
        if self.execution_delay < 1:
            raise ValueError("execution_delay must be >= 1 bar: same-bar execution is look-ahead")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if not 0 < self.max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")


def segment_end_mask(close: np.ndarray, max_gap_bars: int) -> np.ndarray:
    """True at the last valid bar before the data ends or before a gap longer than ``max_gap_bars``.

    Uses the future only to know that a symbol stops trading; the exit itself happens at
    the last observed close with a penalty, which approximates a delisting settlement.
    """
    T, N = close.shape
    valid = ~np.isnan(close)
    out = np.zeros((T, N), dtype=np.bool_)
    rows = np.arange(T)
    for j in range(N):
        idx = rows[valid[:, j]]
        if len(idx) == 0:
            continue
        nxt = np.append(idx[1:], T + max_gap_bars + 1)
        ends = idx[(nxt - idx - 1) > max_gap_bars]
        out[ends[ends < T - 1], j] = True
    return out


def _prepare_weights(weights: pd.DataFrame, data: MarketData, config: BacktestConfig) -> pd.DataFrame:
    unknown = set(weights.columns) - set(data.symbols)
    if unknown:
        raise ValueError(f"weights reference symbols absent from data: {sorted(unknown)}")
    if not weights.index.isin(data.index).all():
        raise ValueError("weights index contains timestamps absent from data")
    w = weights.reindex(index=data.index, columns=data.symbols).astype(float).fillna(0.0)
    values = w.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("weights contain inf")
    if not config.allow_short and (values < 0).any():
        raise ValueError("negative weights while allow_short=False")
    gross = np.abs(values).sum(axis=1)
    scale = np.where(gross > config.max_gross_leverage, config.max_gross_leverage / np.where(gross > 0, gross, 1), 1.0)
    return pd.DataFrame(values * scale[:, None], index=w.index, columns=w.columns)


def run_backtest(weights: pd.DataFrame, data: MarketData, config: BacktestConfig | None = None,
                 name: str = "strategy") -> BacktestResult:
    config = config or BacktestConfig()
    decided = _prepare_weights(weights, data, config)
    executed = decided.shift(config.execution_delay).fillna(0.0)
    change = executed.ne(executed.shift(1).fillna(0.0)).to_numpy()

    close = data.close.to_numpy(dtype=np.float64)
    adv = data.quote_volume.rolling(config.volume_lookback, min_periods=1).mean().shift(1).to_numpy(dtype=np.float64)
    min_periods = max(10, config.vol_lookback // 4)
    sigma = (np.log(data.close).diff().rolling(config.vol_lookback, min_periods=min_periods).std()
             .shift(1).to_numpy(dtype=np.float64))
    funding = (data.funding.to_numpy(dtype=np.float64) if data.funding is not None
               else np.full(close.shape, np.nan))
    force_exit = segment_end_mask(close, config.max_gap_bars)
    costs = config.costs
    daily_volume = adv * bars_per_day(data.interval)
    half_spread = costs.half_spread_matrix(data.symbols, daily_volume)
    financing = np.array([costs.financing_bps_for(s) for s in data.symbols], dtype=np.float64) / 1e4 \
        / bars_per_day(data.interval)

    (equity, gross, net, fees, spread_slip, funding_paid, traded, trades, n_partial,
     liquidated_at, ruined_at, final_qty, final_cash, n_maker, n_taker, n_waits) = _core.simulate(
        np.ascontiguousarray(data.open.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(data.high.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(data.low.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(close),
        np.ascontiguousarray(executed.to_numpy(dtype=np.float64)),
        np.ascontiguousarray(change),
        np.ascontiguousarray(funding),
        np.ascontiguousarray(adv),
        np.ascontiguousarray(sigma),
        half_spread,
        np.ascontiguousarray(force_exit),
        float(config.initial_capital),
        np.array([costs.fee_bps_for(sym) / 1e4 for sym in data.symbols], dtype=np.float64),
        float(costs.slippage_bps),
        float(costs.impact_coef),
        float(config.max_participation),
        float(config.maintenance_margin),
        float(config.liquidation_fee),
        config.rebalance == "every_bar",
        float(config.rebalance_threshold),
        float(config.max_gross_leverage),
        float(config.leverage_tolerance),
        float(config.min_trade_notional),
        float(config.delist_penalty_bps),
        float(config.sigma_fallback),
        bool(costs.funding),
        config.execution.mode == "maker",
        np.array([config.execution.maker_fee_bps_for(sym) / 1e4 for sym in data.symbols], dtype=np.float64),
        float(config.execution.limit_offset_bps),
        float(config.execution.trade_through_bps),
        int(config.execution.maker_fallback_bars),
        financing,
        bool((financing > 0).any()),
    )

    index = data.index
    trade_df = pd.DataFrame(trades, columns=_core.TRADE_COLUMNS)
    if len(trade_df):
        sym_idx = trade_df["symbol_idx"].astype(int).to_numpy()
        entry_bar = trade_df["entry_bar"].astype(int).to_numpy()
        exit_bar = trade_df["exit_bar"].astype(int).to_numpy()
        trade_df.insert(0, "symbol", np.array(data.symbols, dtype=object)[sym_idx])
        trade_df.insert(1, "entry_time", index[entry_bar])
        trade_df.insert(2, "exit_time", index[exit_bar])
        trade_df["bars_held"] = exit_bar - entry_bar + 1
        trade_df["return_on_equity"] = trade_df["pnl"] / trade_df["entry_equity"]
        trade_df["is_open"] = trade_df["is_open"].astype(bool)
        reasons = {0: "signal", 1: "liquidation", 2: "delisting", 3: "end_of_data"}
        trade_df["exit_reason"] = trade_df["exit_reason"].astype(int).map(reasons)
        trade_df = trade_df.drop(columns=["symbol_idx"])

    bars = pd.DataFrame({
        "equity": equity,
        "gross_exposure": gross,
        "net_exposure": net,
        "fees": fees,
        "spread_slippage": spread_slip,
        "funding": funding_paid,
        "traded_notional": traded,
    }, index=index)

    return BacktestResult(
        name=name,
        interval=data.interval,
        config=config,
        bars=bars,
        trades=trade_df,
        target_weights=decided,
        n_partial_fills=int(n_partial),
        liquidated_at=index[liquidated_at] if liquidated_at >= 0 else None,
        ruined_at=index[ruined_at] if ruined_at >= 0 else None,
        final_positions=pd.Series(final_qty, index=data.symbols),
        n_maker_fills=int(n_maker),
        n_taker_fills=int(n_taker),
        n_unfilled_order_bars=int(n_waits),
    )
