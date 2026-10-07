from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant.backtesting.costs import CostModel
from quant.brokers.paper import PaperBroker
from quant.data.market_data import MarketData, resample_bars
from quant.data.universe import UniverseRule, expand_to_bars, point_in_time_eligibility
from quant.live.feed import MarketFeed, Quote
from quant.live.loop import LoopConfig, TradingLoop
from quant.risk.engine import RiskLimits
from quant.strategies.cross_sectional import FundingCarry
from quant.strategies.trend import TrendFollowing
from tests.conftest import gbm_frame

SYMBOLS = [f"S{i:02d}USDT" for i in range(8)]


class FakeFeed(MarketFeed):
    def __init__(self, frames):
        self.frames = frames
        self.funding = {}
        for i, (s, f) in enumerate(frames.items()):
            idx = f.index[f.index.hour % 8 == 0]
            self.funding[s] = pd.DataFrame({"funding_rate": np.sin(np.arange(len(idx)) / 30 + i) * 3e-4}, index=idx)

    def bars(self, symbols, interval, start, now):
        last_open = now.floor("1h") - pd.Timedelta(hours=1)
        return {s: self.frames[s].loc[start:last_open, ["open", "high", "low", "close", "volume", "quote_volume"]]
                for s in symbols if s in self.frames}

    def quotes(self, symbols):
        return {s: Quote(float(self.frames[s]["close"].iloc[-1]) * 0.9999, float(self.frames[s]["close"].iloc[-1]) * 1.0001,
                         pd.Timestamp.now(tz="UTC")) for s in symbols if s in self.frames}

    def funding_since(self, symbol, start, now):
        return pd.DataFrame(columns=["funding_rate"])

    def funding_history(self, symbols, start, now):
        return {s: self.funding[s].loc[start:now] for s in symbols if s in self.funding}

    def metrics_history(self, symbols, start, now, interval="1h", fields=None):
        rng = np.random.default_rng(11)
        idx = next(iter(self.frames.values())).loc[start:now].index
        cols = [s for s in symbols if s in self.frames]
        ls = pd.DataFrame(1.5 + np.cumsum(rng.normal(0, 0.02, (len(idx), len(cols))), axis=0) / 20,
                          index=idx, columns=cols).clip(lower=0.2)
        oi = pd.DataFrame(1e7 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), len(cols))), axis=0)),
                          index=idx, columns=cols)
        return {"global_account_ls": ls, "open_interest": oi}

    def quote_volume_24h(self):
        return pd.Series({s: float(f["quote_volume"].iloc[-24:].sum()) for s, f in self.frames.items()})


@pytest.fixture
def frames():
    rng = np.random.default_rng(1)
    out = {}
    for i, s in enumerate(SYMBOLS):
        f = gbm_frame(24 * 200, 500 + i, start="2026-01-01", sigma=0.008, drift=rng.normal(0, 1e-4),
                      quote_volume=50e6 / 24)
        out[s] = f
    return out


def make_loop(frames, tmp_path, now):
    feed = FakeFeed({s: f.loc[: now - pd.Timedelta(hours=1)] for s, f in frames.items()})
    cfg = LoopConfig(strategy_id="test", strategy=TrendFollowing(lookback_days=10, band=0.0, rebalance_hours=4),
                     interval="4h", universe=UniverseRule(top_n=5, min_median_quote_volume=1e6, min_history_days=10),
                     candidate_symbols=SYMBOLS, history_days=120,
                     risk_limits=RiskLimits(max_position_weight=0.5, max_cluster_weight=1.0))
    broker = PaperBroker(tmp_path / "broker", CostModel(), initial_capital=100_000.0)
    return TradingLoop(cfg, feed, broker, tmp_path / "loop"), broker


NOW = pd.Timestamp("2026-07-01 08:00:30", tz="UTC")


def test_step_trades_and_is_idempotent(frames, tmp_path):
    loop, broker = make_loop(frames, tmp_path, NOW)
    out = loop.step(NOW)
    assert out["status"] == "traded" and out["n_orders_filled"] > 0
    assert not broker.positions().empty
    assert loop.step(NOW + pd.Timedelta(minutes=5))["status"] == "already_processed"
    events = loop.journal.read()
    decisions = [e for e in events if e["event"] == "risk_decision"]
    assert decisions and all("max_loss" in e and "liquidity_risk" in e for e in decisions)
    restarted, _ = make_loop(frames, tmp_path, NOW)
    assert restarted.step(NOW + pd.Timedelta(minutes=10))["status"] == "already_processed"
    nxt = NOW + pd.Timedelta(hours=4)
    later, _ = make_loop(frames, tmp_path, nxt)
    assert later.step(nxt)["status"] == "traded"


@pytest.mark.parametrize("strategy", [TrendFollowing(lookback_days=10, band=0.0, rebalance_hours=4),
                                      FundingCarry(lookback_days=3, rebalance_hours=8, min_symbols=4, quantile=0.25)])
def test_live_targets_match_research_computation(frames, tmp_path, strategy):
    loop, _ = make_loop(frames, tmp_path, NOW)
    loop.cfg.strategy = strategy
    decision_bar = NOW.floor("4h") - pd.Timedelta(hours=4)
    data, _ = loop._load_data(decision_bar, NOW)
    live = loop.cfg.strategy.target_weights(data).iloc[-1]
    assert (live != 0).any()
    start = decision_bar - pd.Timedelta(days=loop.cfg.history_days)
    bars = {s: resample_bars(f.loc[start: NOW.floor("1h") - pd.Timedelta(hours=1)], "1h", "4h") for s, f in frames.items()}
    research = MarketData.from_frames(bars, "4h", loop.feed.funding_history(SYMBOLS, start, NOW))
    qv = pd.DataFrame({s: f.loc[start: NOW.floor("1h") - pd.Timedelta(hours=1), "quote_volume"].resample("1D").sum()
                       for s, f in frames.items()})
    research = research.with_universe(expand_to_bars(point_in_time_eligibility(qv, loop.cfg.universe), research.index))
    expected = loop.cfg.strategy.target_weights(research).loc[decision_bar]
    pd.testing.assert_series_equal(live.sort_index(), expected.reindex(live.index).sort_index(), check_names=False)


def test_kill_file_flattens(frames, tmp_path):
    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.step(NOW)
    assert not broker.positions().empty
    (tmp_path / "loop" / "KILL").write_text("manual stop")
    out = loop.step(NOW + pd.Timedelta(hours=4))
    assert out["status"] == "killed" and broker.positions().empty
    assert loop.step(NOW + pd.Timedelta(hours=8))["status"] == "killed"


def test_reconciliation_mismatch_halts(frames, tmp_path):
    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.step(NOW)
    state_path = tmp_path / "broker" / "broker_state.json"
    state = json.loads(state_path.read_text())
    sym = next(iter(state["positions"]))
    state["positions"][sym] *= 2  # corrupted state
    state_path.write_text(json.dumps(state))
    loop2, _ = make_loop(frames, tmp_path, NOW + pd.Timedelta(hours=4))
    out = loop2.step(NOW + pd.Timedelta(hours=4))
    assert out["status"] == "halted" and loop2.state.kill_switch


def test_late_step_is_skipped(frames, tmp_path):
    loop, broker = make_loop(frames, tmp_path, NOW)
    out = loop.step(NOW + pd.Timedelta(hours=2))  # 2h after the 4h bar close
    assert out["status"] == "late_skipped" and broker.positions().empty


def test_pair_mode_cash_and_carry_is_hedged(frames, tmp_path):
    from quant.strategies.cross_sectional import CashAndCarry
    pair_frames = dict(frames)
    rng = np.random.default_rng(3)
    for s, f in frames.items():
        spot = f.copy()
        spot[["open", "high", "low", "close"]] *= 1 + rng.normal(0, 1e-4, (len(f), 1))
        pair_frames[s + "@spot"] = spot
    feed = FakeFeed({s: f.loc[: NOW - pd.Timedelta(hours=1)] for s, f in pair_frames.items()})
    for i, s in enumerate(SYMBOLS):  # rich funding on half the coins
        idx = feed.frames[s].index[feed.frames[s].index.hour % 8 == 0]
        feed.funding[s] = pd.DataFrame({"funding_rate": 4e-4 if i % 2 == 0 else 1e-5}, index=idx)
    feed.funding = {k: v for k, v in feed.funding.items() if not k.endswith("@spot")}
    cfg = LoopConfig(strategy_id="carry", strategy=CashAndCarry(entry_apr=0.2, exit_apr=0.05, lookback_days=3,
                                                                 rebalance_hours=4, max_pairs=4, gross_per_pair=0.4),
                     interval="4h", universe=UniverseRule(top_n=8, min_median_quote_volume=1e6, min_history_days=10),
                     candidate_symbols=SYMBOLS, history_days=60, pair_mode=True,
                     risk_limits=RiskLimits(max_position_weight=0.5, max_cluster_weight=2.0, max_gross_leverage=2.0,
                                            max_net_leverage=1.0, max_position_loss_at_gap=0.5))
    broker = PaperBroker(tmp_path / "broker", CostModel(per_symbol_fee_bps={"*@spot": 10.0}), initial_capital=100_000.0)
    loop = TradingLoop(cfg, feed, broker, tmp_path / "loop")
    out = loop.step(NOW)
    assert out["status"] == "traded" and out["n_orders_filled"] >= 2
    pos = broker.positions()
    perps = [s for s in pos.index if not s.endswith("@spot")]
    assert perps and all(pos[s] < 0 for s in perps)
    for s in perps:
        assert pos[s + "@spot"] == pytest.approx(-pos[s], rel=0.02)
    fills = [f for e in loop.journal.read() if e["event"] == "step_summary" for f in e["fills"]]
    spot_fill = next(f for f in fills if f["symbol"].endswith("@spot"))
    assert spot_fill["fee"] == pytest.approx(spot_fill["quantity"] * spot_fill["price"] * 10e-4)


def test_public_feed_skips_symbols_without_any_data(tmp_path):
    from quant.data.store import ParquetStore
    from quant.live.feed import BinancePublicFeed

    class NoRest:
        def klines(self, *args, **kwargs):
            import requests
            raise requests.HTTPError("400 invalid symbol")

    feed = BinancePublicFeed(ParquetStore(tmp_path / "store"), tmp_path / "cache", rest=NoRest(), spot_rest=NoRest())
    now = pd.Timestamp("2026-09-14 12:00", tz="UTC")
    assert feed.bars(["NOPEUSDT", "NOPEUSDT@spot"], "1h", now - pd.Timedelta(days=2), now) == {}


def test_step_with_no_targets_and_no_positions(frames, tmp_path):
    from dataclasses import replace as dc_replace

    class Flat(TrendFollowing):
        def target_weights(self, data):
            return super().target_weights(data) * 0.0

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.cfg = dc_replace(loop.cfg, strategy=Flat())
    out = loop.step(NOW)
    assert out["status"] == "traded" and out["n_orders_filled"] == 0 and broker.positions().empty
    assert out["equity"] == pytest.approx(100_000.0)


def test_divergence_report_compares_paper_and_backtest(frames, tmp_path):
    from quant.backtesting.engine import BacktestConfig, run_backtest
    from quant.data.market_data import MarketData, resample_bars
    from quant.monitoring.divergence import compare, paper_equity

    loop, _ = make_loop(frames, tmp_path, NOW)
    loop.step(NOW)
    later = NOW + pd.Timedelta(hours=4)
    loop2, _ = make_loop(frames, tmp_path, later)
    loop2.step(later)
    bars = {s: resample_bars(f, "1h", "4h") for s, f in frames.items()}
    data = MarketData.from_frames(bars, "4h")
    result = run_backtest(loop.cfg.strategy.target_weights(data), data, BacktestConfig(), name="replay")
    report = compare(tmp_path / "loop", result)
    assert report.traded_steps == 2 and report.skipped_steps == 0
    assert np.isfinite(report.mean_cost_bps_paper) and report.mean_latency_seconds >= 0
    assert "not yet meaningful" in " ".join(report.notes)
    assert len(paper_equity(loop.journal.read())) == 2
    print(report.summary())


def test_metrics_strategy_runs_in_the_paper_loop(frames, tmp_path):
    from dataclasses import replace as dc_replace
    from quant.strategies.positioning import CrowdingContrarian

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.cfg = dc_replace(loop.cfg, with_metrics=True, interval="4h",
                          strategy=CrowdingContrarian(z_window_days=10, rebalance_hours=4, quantile=0.25,
                                                      min_symbols=4))
    out = loop.step(NOW)
    assert out["status"] == "traded" and out["n_orders_filled"] > 0
    pos = broker.positions()
    assert (pos > 0).any() and (pos < 0).any()   # long/short book


def test_changing_venue_resets_the_equity_history(frames, tmp_path):
    """A 10,000 paper peak carried into a 5,000 testnet account would read as a -50% drawdown."""
    loop, _ = make_loop(frames, tmp_path, NOW)
    loop.state.equity_peak = 10_000.0
    loop.state.kill_switch = True
    loop.state.kill_reason = "stale"
    loop._persist()
    assert loop.state.venue == "PaperBroker"

    class OtherBroker(PaperBroker):
        pass

    other = OtherBroker(tmp_path / "broker2", CostModel(), initial_capital=5_000.0)
    reloaded = TradingLoop(loop.cfg, loop.feed, other, tmp_path / "loop")
    assert reloaded.state.equity_peak is None and reloaded.state.kill_switch is False
    assert reloaded.state.venue == "OtherBroker"
    assert "venue changed" in (tmp_path / "loop" / "events.jsonl").read_text()


def test_state_file_from_an_older_version_still_loads(tmp_path):
    from quant.live.loop import LoopState

    path = tmp_path / "loop_state.json"
    path.write_text('{"last_decision_bar": "2026-01-01T00:00:00+00:00", "equity_peak": 42.0, "legacy_field": 1}')
    state = LoopState.load(path)
    assert state.equity_peak == 42.0 and state.venue is None


def test_universe_is_restricted_to_what_the_venue_lists(frames, tmp_path):
    """A venue listing fewer symbols than the research universe must narrow the book, not crash."""
    loop, broker = make_loop(frames, tmp_path, NOW)
    listed = set(SYMBOLS[:3])
    broker.supported_symbols = lambda: listed
    out = loop.step(NOW)
    assert out["status"] == "traded"
    assert set(broker.positions().index) <= listed
    events = (tmp_path / "loop" / "events.jsonl").read_text()
    assert "symbols not listed at the venue" in events


# ---------------------------------------------------------------- the loop trades like the backtest

def test_loop_leaves_the_book_alone_between_strategy_decisions(frames, tmp_path):
    """The backtester trades a symbol only when its target changes (``on_change``). Re-aligning drift
    on every bar is turnover the research never paid for."""
    from quant.strategies.cross_sectional import CrossSectionalMomentum

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.feed = FakeFeed(frames)                               # the whole history: the test walks forward in time
    loop.cfg.strategy = CrossSectionalMomentum(lookback_hours=24 * 3, rebalance_hours=24, quantile=0.25,
                                               min_symbols=4)
    start = pd.Timestamp("2026-07-01 04:00:30", tz="UTC")      # decision bar 00:00: a daily rebalance
    steps = [loop.step(start + pd.Timedelta(hours=4 * k)) for k in range(6 * 10)]   # ten days, every bar
    assert all(s["status"] == "traded" for s in steps)
    rebalances = [s for s in steps if s["rebalanced"]]
    assert steps[0]["rebalanced"] and steps[0]["n_orders_filled"] > 0
    assert len(rebalances) >= 3                                 # the ranking does change over ten days
    for s in steps:
        if s["rebalanced"]:
            assert pd.Timestamp(s["decision_bar"]).hour == 0    # only on the strategy's own decision bars
        else:
            assert s["n_orders_filled"] == 0                    # five hold bars a day: not a single order
    assert sum(s["n_orders_filled"] > 0 for s in steps) <= len(rebalances)


def test_loop_catches_up_a_decision_it_missed(frames, tmp_path):
    """Down during the rebalance bar: the next processed bar must still act on that decision."""
    from quant.strategies.cross_sectional import CrossSectionalMomentum

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.feed = FakeFeed(frames)
    loop.cfg.strategy = CrossSectionalMomentum(lookback_hours=24 * 5, rebalance_hours=24, quantile=0.25,
                                               min_symbols=4)
    late = pd.Timestamp("2026-07-01 12:00:30", tz="UTC")       # first bar seen is a hold bar (08:00)
    out = loop.step(late)
    assert out["rebalanced"] and out["n_orders_filled"] > 0
    assert pd.Timestamp(out["strategy_change_bar"]) == pd.Timestamp("2026-07-01 00:00", tz="UTC")
    assert loop.step(late + pd.Timedelta(hours=4))["n_orders_filled"] == 0


def test_last_change_bar_is_the_latest_row_that_differs():
    idx = pd.date_range("2026-01-01", periods=6, freq="4h", tz="UTC")
    w = pd.DataFrame({"A": [0, .5, .5, .5, .2, .2], "B": [0, -.5, -.5, -.5, -.2, -.2]}, index=idx)
    assert TradingLoop._last_change_bar(w) == idx[4]
    assert TradingLoop._last_change_bar(w.iloc[:4]) == idx[1]
    assert TradingLoop._last_change_bar(w.iloc[:1]) is None
    assert TradingLoop._last_change_bar(w * 0.0) is None


def test_venue_settled_funding_is_reported_once_and_not_subtracted_from_equity(frames, tmp_path):
    """A venue has already debited funding from the balance it reports."""
    loop, broker = make_loop(frames, tmp_path, NOW)
    calls = []

    class VenueBroker(PaperBroker):
        settles_funding_itself = True

        def funding_paid_since_last_check(self, now):
            calls.append(now)
            return 12.5

        def apply_funding(self, *args, **kwargs):
            raise AssertionError("the loop must not apply funding a venue settles itself")

    venue = VenueBroker(tmp_path / "venue", CostModel(), initial_capital=100_000.0)
    loop = TradingLoop(loop.cfg, loop.feed, venue, tmp_path / "venue_loop")
    out = loop.step(NOW)
    assert out["funding_paid"] == 12.5 and len(calls) == 1
    sized = [json.loads(line) for line in (tmp_path / "venue_loop" / "events.jsonl").read_text().splitlines()]
    notional = [abs(d["position_notional"]) for d in sized if d.get("event") == "risk_decision"]
    assert max(notional) <= 100_000.0 * 0.5 + 1e-6           # sized on the venue's equity, not equity - funding


def test_held_symbols_stay_in_the_candidate_list(frames, tmp_path):
    """A position must remain priced even on a day its volume falls out of the shortlist."""
    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.step(NOW)
    held = list(broker.positions().index)
    assert held
    loop.cfg.max_candidates = 2
    loop.feed.quote_volume_24h = lambda: pd.Series({s: (0.0 if s in held else 1e9) for s in SYMBOLS})
    data, _ = loop._load_data(pd.Timestamp("2026-07-01 08:00", tz="UTC"), NOW + pd.Timedelta(hours=4))
    assert set(held) <= set(data.symbols)


def test_orphan_positions_are_closed_at_the_next_decision(frames, tmp_path):
    """After a state reset the loop remembers no target: a position without one must still be closed."""
    from quant.execution.orders import Order, Side
    from quant.strategies.cross_sectional import CrossSectionalMomentum

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.feed = FakeFeed(frames)
    loop.cfg.strategy = CrossSectionalMomentum(lookback_hours=24 * 3, rebalance_hours=24, quantile=0.25,
                                               min_symbols=4)
    when = pd.Timestamp("2026-07-01 04:00:30", tz="UTC")
    weights = loop.cfg.strategy.target_weights(loop._load_data(pd.Timestamp("2026-07-01 00:00", tz="UTC"), when)[0])
    unwanted = [s for s in SYMBOLS if weights[s].iloc[-1] == 0.0][0]
    price = float(frames[unwanted]["close"].iloc[-1])
    broker.submit(Order("manual-1", unwanted, Side.BUY, 10.0, when, price), price, when, 1e9)
    assert unwanted in broker.positions().index
    out = loop.step(when)
    assert out["rebalanced"] and unwanted not in broker.positions().index


def test_drift_mode_pulls_positions_back_to_their_target_weight(frames, tmp_path):
    """Candidate behaviour (backtester's ``every_bar``): a position that drifted is re-sized even on a
    bar where the strategy decided nothing new."""
    from dataclasses import replace as dc_replace
    from quant.strategies.cross_sectional import CrossSectionalMomentum

    loop, broker = make_loop(frames, tmp_path, NOW)
    loop.feed = FakeFeed(frames)
    loop.cfg = dc_replace(loop.cfg, rebalance_mode="drift", rebalance_tolerance=0.0005,
                          strategy=CrossSectionalMomentum(lookback_hours=24 * 3, rebalance_hours=24, quantile=0.25,
                                                          min_symbols=4))
    start = pd.Timestamp("2026-07-01 04:00:30", tz="UTC")
    steps = [loop.step(start + pd.Timedelta(hours=4 * k)) for k in range(12)]
    hold_bar_orders = sum(s["n_orders_filled"] for s in steps if not s["rebalanced"])
    assert hold_bar_orders > 0 and all(s["rebalance_mode"] == "drift" for s in steps)
    with pytest.raises(ValueError, match="rebalance_mode"):
        TradingLoop(dc_replace(loop.cfg, rebalance_mode="sometimes"), loop.feed, broker, tmp_path / "x")
