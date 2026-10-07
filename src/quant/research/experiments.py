"""Pre-registered research experiments.

Each experiment fixes its hypothesis, universe, bar size, parameter grid and
evaluation windows BEFORE results are seen. Changing a grid after looking at results
creates a new experiment (new name) so the trial count stays honest.

Research window: 2020-01-01 -> 2026-01-01. Holdout: 2026-01-01 -> 2026-09-01, evaluated
once per experiment, after the walk-forward verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Callable

from quant.backtesting.costs import CostModel
from quant.backtesting.engine import BacktestConfig, ExecutionModel
from quant.data.universe import UniverseRule
from quant.evaluation.validation import WalkForwardSpec
from quant.strategies.base import Strategy
from quant.strategies.combine import ParameterEnsemble, RiskParitySleeves
from quant.strategies.cross_sectional import (BtcLeadLag, CarryTrendFiltered, CashAndCarry, CrossSectionalMomentum,
                                             FundingCarry)
from quant.strategies.events import IdiosyncraticCascadeReversal, LiquidationCascadeReversal
from quant.strategies.positioning import (CascadeReversalOI, CrowdingContrarian, CrowdingContrarianRobust,
                                          OpenInterestTrend)
from quant.strategies.seasonality import HourOfDaySeasonality
from quant.strategies.mean_reversion import FastRsiReversion
from quant.strategies.trend import TrendFollowing

RESEARCH_START = "2020-01-01"
HOLDOUT_START = "2026-01-01"
HOLDOUT_END = "2026-09-01"

REALISTIC_COSTS = CostModel(fee_bps=5.0, slippage_bps=1.0, impact_coef=0.5, spread_model="volume",
                            spread_coef_bps=1.0, min_half_spread_bps=0.5, max_half_spread_bps=25.0)
# Binance USDT-M VIP0 maker fee 2 bps; limit 1 bp inside the last close; a fill needs 2 bps trade-through.
MAKER_EXECUTION = ExecutionModel(mode="maker", maker_fee_bps=2.0, limit_offset_bps=1.0, trade_through_bps=2.0,
                                 maker_fallback_bars=3)
MAKER_CONFIG = BacktestConfig(costs=REALISTIC_COSTS, max_gross_leverage=1.0, execution=MAKER_EXECUTION)


@dataclass(frozen=True)
class PromotionGates:
    """Minimum evidence to promote a strategy to paper trading."""

    min_oos_sharpe: float = 0.8
    min_deflated_sharpe: float = 0.90
    max_pbo: float = 0.30
    max_bootstrap_prob_sharpe_le_0: float = 0.10
    min_oos_sharpe_costs_x2: float = 0.4
    min_oos_max_drawdown: float = -0.40


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str
    universe: UniverseRule
    interval: str
    candidates: Callable[[], list[Strategy]]
    source_interval: str = "1h"
    start: str = RESEARCH_START
    holdout_start: str = HOLDOUT_START
    holdout_end: str = HOLDOUT_END
    walk_forward: WalkForwardSpec = WalkForwardSpec(train_days=365, test_days=90)
    config: BacktestConfig = field(default_factory=lambda: BacktestConfig(costs=REALISTIC_COSTS, max_gross_leverage=1.0))
    gates: PromotionGates = PromotionGates()
    with_order_flow: bool = False
    with_metrics: bool = False
    data_loader: str = "universe"   # "universe" | "carry_pairs" (perp + @spot columns)


def _trend_grid() -> list[Strategy]:
    return [TrendFollowing(lookback_days=lb, band=band, rebalance_hours=24, vol_target=vt, max_asset_weight=1.0)
            for lb, band, vt in product((7, 14, 30, 60), (0.0, 0.5, 1.0), (0.0, 0.6))]


def _xsmom_grid() -> list[Strategy]:
    return [CrossSectionalMomentum(lookback_hours=24 * d, skip_hours=skip, rebalance_hours=rb, quantile=0.2,
                                   min_symbols=10)
            for d, skip, rb in product((3, 7, 14, 30), (0, 24), (24, 72))]


def _xsrev_grid() -> list[Strategy]:
    return [CrossSectionalMomentum(lookback_hours=lb, rebalance_hours=rb, quantile=0.2, min_symbols=10, reverse=True)
            for lb, rb in product((1, 4, 12, 24), (1, 4, 12))]


def _carry_grid() -> list[Strategy]:
    return [FundingCarry(lookback_days=d, rebalance_hours=rb, quantile=q, min_symbols=10)
            for d, rb, q in product((1, 3, 7, 14), (8, 24), (0.1, 0.2))]


def _rsi2_grid() -> list[Strategy]:
    return [FastRsiReversion(rsi_window=2, lower=lo, upper=100 - lo, exit_level=50.0, max_hold_bars=hold,
                             trend_filter_days=tf, max_positions=10)
            for lo, hold, tf in product((5.0, 10.0), (6, 12, 24), (0, 10))]


def _multi_sleeve() -> list[Strategy]:
    sleeves = (ParameterEnsemble(tuple(_trend_grid()), "trend"), ParameterEnsemble(tuple(_xsmom_grid()), "xsmom"),
               ParameterEnsemble(tuple(_carry_grid()), "carry"))
    return [RiskParitySleeves(sleeves, "trend+xsmom+carry", target_vol=0.30, max_gross=1.5)]


def _carry_filtered_grid() -> list[Strategy]:
    return [CarryTrendFiltered(lookback_days=d, momentum_days=m, rebalance_hours=24, quantile=0.2, min_symbols=10)
            for d, m in product((3, 7, 14), (7, 30))]


def _multi_sleeve_v2() -> list[Strategy]:
    sleeves = (ParameterEnsemble(tuple(_trend_grid()), "trend"), ParameterEnsemble(tuple(_xsmom_grid()), "xsmom"),
               ParameterEnsemble(tuple(_carry_filtered_grid()), "carry_filtered"))
    return [RiskParitySleeves(sleeves, "trend+xsmom+carry_filtered+ddbrake", target_vol=0.30, max_gross=1.5,
                              drawdown_brake=0.30, brake_floor=0.25)]


def _seasonality_grid() -> list[Strategy]:
    return [HourOfDaySeasonality(lookback_days=lb, top_hours=k, long_short=ls)
            for lb, k, ls in product((60, 180), (2, 4), (False, True))]


def _leadlag_grid() -> list[Strategy]:
    return [BtcLeadLag(lookback_hours=lb, beta_window_days=30, rebalance_hours=rb, quantile=0.2, min_symbols=10)
            for lb, rb in product((1, 3), (1, 4))]


def _cascade_grid() -> list[Strategy]:
    return [LiquidationCascadeReversal(sigma_threshold=k, volume_multiple=3.0, max_taker_buy_share=0.45, hold_bars=h,
                                       vol_window_days=30, long_short=ls, max_positions=10)
            for k, h, ls in product((3.0, 5.0), (4, 12, 24), (False, True))]


def _idio_cascade_grid() -> list[Strategy]:
    return [IdiosyncraticCascadeReversal(sigma_threshold=k, volume_multiple=3.0, max_taker_buy_share=0.45, hold_bars=h,
                                         vol_window_days=30, max_breadth=b, breadth_sigma=3.0, max_positions=10)
            for k, h, b in product((4.0, 5.0), (12, 24), (0.10, 0.30))]


def _multi_sleeve_v3() -> list[Strategy]:
    sleeves = (ParameterEnsemble(tuple(_trend_grid()), "trend"), ParameterEnsemble(tuple(_xsmom_grid()), "xsmom"),
               ParameterEnsemble(tuple(_carry_grid()), "carry"), ParameterEnsemble(tuple(_cascade_grid()), "cascade"))
    return [RiskParitySleeves(sleeves, "trend+xsmom+carry+cascade", target_vol=0.30, max_gross=1.5)]


# spot legs: Binance spot VIP0 fee 10 bps for both maker and taker
CARRY_CONFIG = BacktestConfig(
    costs=CostModel(fee_bps=5.0, slippage_bps=1.0, impact_coef=0.5, spread_model="volume", spread_coef_bps=1.0,
                    min_half_spread_bps=0.5, max_half_spread_bps=25.0, per_symbol_fee_bps={"*@spot": 10.0}),
    max_gross_leverage=1.6,
    execution=ExecutionModel(mode="maker", maker_fee_bps=2.0, limit_offset_bps=1.0, trade_through_bps=2.0,
                             maker_fallback_bars=3, per_symbol_maker_fee_bps={"*@spot": 10.0}),
)


def _cash_carry_grid() -> list[Strategy]:
    return [CashAndCarry(entry_apr=e, exit_apr=e / 3, lookback_days=d, rebalance_hours=24, max_pairs=10,
                         gross_per_pair=0.16)
            for e, d in product((0.10, 0.20), (3, 7))]


METRICS_START = "2022-01-01"   # Binance publishes open interest / positioning from 2020-09, downloaded from 2022


def _cascade_oi_grid() -> list[Strategy]:
    return [CascadeReversalOI(sigma_threshold=k, volume_multiple=3.0, max_taker_buy_share=0.45, min_oi_drop=d,
                              hold_bars=h, vol_window_days=30, max_positions=10)
            for k, d, h in product((3.0, 5.0), (0.005, 0.02), (12, 24))]


def _oi_trend_grid() -> list[Strategy]:
    return [OpenInterestTrend(lookback_hours=lb, rebalance_hours=rb, quantile=0.2, min_symbols=8, mode=m)
            for lb, m, rb in product((24, 168), ("confirm", "fade"), (24, 72))]


def _crowding_grid() -> list[Strategy]:
    return [CrowdingContrarian(z_window_days=w, rebalance_hours=rb, quantile=0.2, min_symbols=8, ratio=r)
            for w, rb, r in product((30, 90), (8, 24), ("global_account_ls", "top_trader_position_ls"))]


def _crowding_robust_grid() -> list[Strategy]:
    return [CrowdingContrarianRobust(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=8,
                                     ratio="global_account_ls", vol_window_days=30, max_weight=mw, min_age_days=age)
            for mw, age in product((0.05, 0.10), (0, 60))]


def _multi_sleeve_v4() -> list[Strategy]:
    sleeves = (ParameterEnsemble(tuple(_trend_grid()), "trend"), ParameterEnsemble(tuple(_xsmom_grid()), "xsmom"),
               CrowdingContrarianRobust(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=8,
                                        vol_window_days=30, max_weight=0.10, min_age_days=60))
    return [RiskParitySleeves(sleeves, "trend+xsmom+crowding", target_vol=0.30, max_gross=1.5)]


EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in [
    Experiment(
        name="trend_top10_4h",
        hypothesis="Vol-normalised time-series momentum with hysteresis earns a premium on the most liquid perps "
                   "after costs, and hysteresis/daily rebalancing fixes the turnover problem seen in baselines.",
        universe=UniverseRule(top_n=10, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_trend_grid,
    ),
    Experiment(
        name="xsmom_top50_4h",
        hypothesis="Cross-sectional momentum (winners minus losers over days to weeks) is priced in crypto perps.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_xsmom_grid,
    ),
    Experiment(
        name="xsrev_top30_1h",
        hypothesis="Short-horizon cross-sectional reversal (liquidity provision) survives realistic spreads.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_xsrev_grid,
    ),
    Experiment(
        name="carry_top50_4h",
        hypothesis="Shorting high-funding / buying low-funding perps collects carry that is not fully offset by "
                   "adverse price moves.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_carry_grid,
    ),
    # ---- batch 2 (registered 2026-09-14 AFTER batch 1 results: walk-forward parameter selection was unstable
    # while most grid members were profitable in-sample; ensembles remove selection instead of choosing)
    Experiment(
        name="trend_ensemble_top10_4h",
        hypothesis="Averaging the whole trend grid is more robust out-of-sample than selecting parameters.",
        universe=UniverseRule(top_n=10, min_median_quote_volume=5e6),
        interval="4h",
        candidates=lambda: [ParameterEnsemble(tuple(_trend_grid()), "trend_grid24")],
    ),
    Experiment(
        name="xsmom_ensemble_top50_4h",
        hypothesis="Averaging the whole cross-sectional momentum grid is more robust than selecting parameters.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=lambda: [ParameterEnsemble(tuple(_xsmom_grid()), "xsmom_grid16")],
    ),
    Experiment(
        name="carry_ensemble_top50_4h",
        hypothesis="Averaging the funding-carry grid is more robust than selecting parameters.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=lambda: [ParameterEnsemble(tuple(_carry_grid()), "carry_grid16")],
    ),
    Experiment(
        name="multisleeve_top30_4h",
        hypothesis="Inverse-vol combination of trend, cross-sectional momentum and carry ensembles diversifies "
                   "their drawdowns enough to pass the gates.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_multi_sleeve,
    ),
    # ---- batch 3 (registered 2026-09-14 after the cost decomposition showed a gross short-term reversal edge
    # destroyed by taker costs; same grids, only the execution model changes, parameters fixed a priori)
    Experiment(
        name="rsi2_maker_top30_1h",
        hypothesis="The RSI(2) reversion edge (gross Sharpe ~2.9) survives when entries/exits are passive limit "
                   "orders with a queue haircut, maker fees and adverse selection.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=lambda: _rsi2_grid(),
        config=MAKER_CONFIG,
    ),
    Experiment(
        name="xsrev_maker_top30_1h",
        hypothesis="Short-horizon cross-sectional reversal survives with passive limit-order execution.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=lambda: _xsrev_grid(),
        config=MAKER_CONFIG,
    ),
    Experiment(
        name="multisleeve_maker_top30_4h",
        hypothesis="Passive execution lowers the multi-sleeve's costs more than adverse selection hurts it.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_multi_sleeve,
        config=MAKER_CONFIG,
    ),
    # ---- batch 4 (registered 2026-09-14 after batch 3; parameters fixed a priori)
    Experiment(
        name="carry_filtered_maker_top50_4h",
        hypothesis="Filtering carry legs by momentum (no shorting into rallies, no buying falling perps) removes "
                   "the squeeze drawdowns while keeping the funding income.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_carry_filtered_grid,
        config=MAKER_CONFIG,
    ),
    Experiment(
        name="multisleeve_v2_maker_top30_4h",
        hypothesis="Trend + XS momentum + momentum-filtered carry, inverse-vol weighted with a drawdown brake, "
                   "keeps the multi-sleeve return while bringing the drawdown under 40%.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_multi_sleeve_v2,
        config=MAKER_CONFIG,
    ),
    Experiment(
        name="seasonality_top5_maker_1h",
        hypothesis="Hour-of-day return seasonality in the most liquid perps is persistent enough to trade daily.",
        universe=UniverseRule(top_n=5, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_seasonality_grid,
        config=MAKER_CONFIG,
    ),
    Experiment(
        name="leadlag_top30_maker_1h",
        hypothesis="Alts lag BTC at the hourly horizon; buying laggards / selling leaders earns more than costs.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_leadlag_grid,
        config=MAKER_CONFIG,
    ),
    # ---- batch 5 (registered 2026-09-14 after batch 4: hourly signals lose to costs; look for rare events with
    # large expected bounces and an economic reason: liquidity provision after forced liquidations)
    Experiment(
        name="cascade_maker_top50_1h",
        hypothesis="After liquidation cascades (>= k sigma move, 3x volume, aggressive sellers dominating) prices "
                   "rebound enough over hours to pay passive execution costs and adverse selection.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_cascade_grid,
        config=MAKER_CONFIG,
        with_order_flow=True,
    ),
    # ---- batch 6 (registered 2026-09-14 after batch 5 showed losses concentrated in market-wide crashes)
    Experiment(
        name="idio_cascade_maker_top50_1h",
        hypothesis="Restricting cascade reversals to idiosyncratic (BTC-residual) shocks in calm-breadth markets "
                   "keeps the liquidity premium and removes the systemic-crash losses of 2022 and 2025.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_idio_cascade_grid,
        config=MAKER_CONFIG,
        with_order_flow=True,
    ),
    # ---- batch 7 (registered 2026-09-14: the cascade premium behaves like selling crash insurance; do not select
    # a variant, average the grid, and pair it with trend following, which tends to profit in crashes)
    Experiment(
        name="cascade_ensemble_maker_top50_1h",
        hypothesis="The whole cascade-reversal grid, averaged, harvests the liquidity premium without selection risk.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="1h",
        candidates=lambda: [ParameterEnsemble(tuple(_cascade_grid()), "cascade_grid12")],
        config=MAKER_CONFIG,
        with_order_flow=True,
    ),
    Experiment(
        name="multisleeve_v3_maker_top30_1h",
        hypothesis="Adding the (short-convexity) cascade premium to (long-convexity) trend, XS momentum and carry "
                   "improves risk-adjusted returns and drawdowns of the multi-sleeve.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_multi_sleeve_v3,
        config=MAKER_CONFIG,
        with_order_flow=True,
    ),
    # ---- batch 8 (registered 2026-09-14: structural funding capture without directional exposure)
    Experiment(
        name="cash_carry_top50_4h",
        hypothesis="Long spot / short perpetual on coins with rich funding captures the funding premium net of "
                   "four-leg fees, basis moves and legging risk, with low drawdowns.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_cash_carry_grid,
        config=CARRY_CONFIG,
        data_loader="carry_pairs",
    ),
    Experiment(
        name="cash_carry_taker_top50_4h",
        hypothesis="Same as cash_carry_top50_4h with both legs executed as market orders in the same bar (pair "
                   "trades need atomic execution; independent maker legs created unhedged exposure).",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_cash_carry_grid,
        config=replace(CARRY_CONFIG, execution=ExecutionModel(mode="taker")),
        data_loader="carry_pairs",
    ),
    # ---- batch 9 (registered 2026-09-16: open interest and positioning data, research window starts with the
    # downloaded metrics coverage; parameters fixed a priori)
    Experiment(
        name="cascade_oi_top30_1h",
        hypothesis="Cascades that come with falling open interest are forced unwinds and revert; requiring the "
                   "open-interest drop removes the crashes driven by new short positions.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_cascade_oi_grid,
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_order_flow=True,
        with_metrics=True,
    ),
    Experiment(
        name="oi_trend_top30_4h",
        hypothesis="Price moves backed by rising open interest (new money) continue, while moves on falling open "
                   "interest (position unwinding) revert.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_oi_trend_grid,
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    Experiment(
        name="crowding_top30_4h",
        hypothesis="Perps where retail accounts are unusually long underperform those where they are unusually "
                   "short (crowding is a contrarian signal).",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_crowding_grid,
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    # ---- batch 10 (registered 2026-09-17): confirmation of the crowding signal on a different cross-section and a
    # different bar size, with the configuration frozen (single candidate, no selection).
    Experiment(
        name="crowding_top50_4h",
        hypothesis="The retail-crowding contrarian edge found on the top-30 universe also holds on a wider, less "
                   "liquid cross-section (top 50).",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=lambda: [CrowdingContrarian(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=8,
                                               ratio="global_account_ls")],
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    Experiment(
        name="crowding_top20_1h",
        hypothesis="The same crowding signal holds at hourly decision frequency on the most liquid perps, where "
                   "costs bite harder.",
        universe=UniverseRule(top_n=20, min_median_quote_volume=5e6),
        interval="1h",
        candidates=lambda: [CrowdingContrarian(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=8,
                                               ratio="global_account_ls")],
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    # ---- batch 12 (registered 2026-10-06, during the pipeline review). Origin is POST-HOC and must be read
    # as such: a diagnostic run while fixing the live loop showed that holding positions at their target
    # weight beats letting them drift, on 8 configurations and 3 thresholds. There is a structural reason
    # (a short kept at constant weight earns the log-return of a collapse instead of being capped at its
    # initial notional, and is trimmed during a squeeze instead of growing), but the idea was not formed
    # before looking. Research window only; the evidence that counts is the parallel forward paper loop.
    Experiment(
        name="crowding_drift_top50_4h",
        hypothesis="Re-aligning a position once it has drifted 0.5% of equity from its target weight improves "
                   "the crowding strategy, mainly through the short leg: constant-weight shorts keep their "
                   "exposure through a collapse and shed it through a squeeze.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=lambda: [CrowdingContrarian(z_window_days=90, rebalance_hours=24, quantile=0.2, min_symbols=8,
                                               ratio="global_account_ls")],
        config=replace(MAKER_CONFIG, rebalance="every_bar", rebalance_threshold=0.005),
        start=METRICS_START,
        with_metrics=True,
    ),
    # ---- batch 11 (registered 2026-09-17, after the holdout): the 2026 profit came from three collapsing tokens,
    # so add concentration controls, and check whether the signal diversifies the slower sleeves.
    # The holdout is spent: these are evaluated on the research window only, then forward in paper.
    Experiment(
        name="crowding_robust_top50_4h",
        hypothesis="Inverse-volatility sizing, a per-symbol cap and a minimum listing age keep most of the crowding "
                   "premium while removing its dependence on a handful of collapsing tokens.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_crowding_robust_grid,
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    Experiment(
        name="multisleeve_v4_maker_top50_4h",
        hypothesis="Trend + cross-sectional momentum + the (robust) crowding signal, inverse-vol weighted, produce "
                   "a smoother book than any sleeve alone.",
        universe=UniverseRule(top_n=50, min_median_quote_volume=5e6),
        interval="4h",
        candidates=_multi_sleeve_v4,
        config=MAKER_CONFIG,
        start=METRICS_START,
        with_metrics=True,
    ),
    Experiment(
        name="rsi2_top30_1h",
        hypothesis="Strategy Factory's RSI(2) 1h reversion (01M2AC3J2MJ966Q8EC18EX9ZZD) generalises across liquid "
                   "alts once executed at the next open with realistic costs.",
        universe=UniverseRule(top_n=30, min_median_quote_volume=5e6),
        interval="1h",
        candidates=_rsi2_grid,
    ),
]}
