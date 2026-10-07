from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant.data.market_data import FIELDS, resample_bars
from quant.data.timeframes import interval_to_timedelta
from quant.discretionary.book import BookConfig, BookError, DiscretionaryBook, Exit, ExitRule, find_exit
from quant.discretionary.brief import build_brief, render_brief
from quant.discretionary.calls import BookLimits, DecisionError, parse_decision
from quant.discretionary.scoring import call_frame, render_scorecard, scorecard, sign_flip_pvalue
from quant.live.feed import MarketFeed, Quote
from tests.conftest import gbm_frame

T0 = pd.Timestamp("2026-03-01 10:02", tz="UTC")      # deliberately off the 5-minute grid
DAY, STEP = pd.Timedelta(days=1), pd.Timedelta(minutes=5)


class FakeFeed(MarketFeed):
    """5-minute frames per symbol; ``clock`` is the instant the venue is asked about."""

    def __init__(self, frames):
        self.frames, self.funding, self.clock = frames, {}, T0

    def bars(self, symbols, interval, start, now):
        step = interval_to_timedelta(interval)
        out = {}
        for s in symbols:
            if s not in self.frames:
                continue
            closed = self.frames[s][self.frames[s].index < now.floor(STEP)]
            closed = resample_bars(closed, "5m", interval)
            out[s] = closed.loc[start: now.floor(step) - step, list(FIELDS)]
        return out

    def quotes(self, symbols):
        out = {}
        for s in symbols:
            if s in self.frames:
                px = float(self.frames[s]["open"].asof(self.clock))
                out[s] = Quote(px * (1 - 1e-4), px * (1 + 1e-4), self.clock)
        return out

    def funding_since(self, symbol, start, end):
        if symbol not in self.funding:
            return pd.DataFrame(columns=["funding_rate"])
        events = self.funding[symbol]
        return events[(events.index >= start) & (events.index < end)]

    def quote_volume_24h(self):
        return pd.Series({s: float(f["quote_volume"][f.index < self.clock].iloc[-288:].sum())
                          for s, f in self.frames.items()})

    def funding_now(self):
        return pd.Series({s: 1e-4 for s in self.frames})


def flat_from(frame: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Freeze the price from ``ts`` on, so a test decides every move that follows the entry."""
    price = float(frame.loc[ts, "open"])
    frame.loc[frame.index >= ts, ["open", "high", "low", "close"]] = price
    return price


@pytest.fixture
def feed():
    specs = {"BTCUSDT": (1, 0.0004, 500e6), "ETHUSDT": (2, 0.0004, 500e6), "SOLUSDT": (3, 0.0004, 500e6),
             "WILDUSDT": (4, 0.006, 500e6), "TINYUSDT": (5, 0.0004, 20e6)}
    frames = {}
    for symbol, (seed, sigma, volume) in specs.items():
        frames[symbol] = gbm_frame(288 * 50, seed, start="2026-01-20", interval="5m", sigma=sigma,
                                   quote_volume=volume / 288)
        flat_from(frames[symbol], T0.floor(STEP))
    return FakeFeed(frames)


@pytest.fixture
def book(feed, tmp_path):
    return DiscretionaryBook(BookConfig(), feed, tmp_path / "book")


def call(symbol="ETHUSDT", **overrides) -> dict:
    return {"symbol": symbol, "direction": "long", "weight": 0.05, "stop_pct": 0.04, "target_pct": 0.08,
            "horizon_days": 5, "confidence": 0.6,
            "thesis": "A catalyst that the market has not priced, described in a full sentence.",
            "invalidation": "The catalyst is cancelled.", "sources": ["https://example.org/a"]} | overrides


def decision(now: pd.Timestamp, calls=(), close=(), **extra) -> dict:
    return {"date": f"{now:%Y-%m-%d}", "as_of": (now - pd.Timedelta(minutes=10)).isoformat(),
            "market_view": "Range-bound market, one idiosyncratic catalyst.", "calls": list(calls),
            "close": list(close)} | extra


def run(book, feed, now, raw, **kwargs):
    feed.clock = now
    return book.execute(parse_decision(raw), now, **kwargs)


def refresh(book, feed, now):
    feed.clock = now
    return book.refresh(now)


def poke(feed, symbol, when, **ohlc):
    feed.frames[symbol].loc[when, list(ohlc)] = list(ohlc.values())


# ------------------------------------------------------------------ decisions
def test_a_valid_decision_parses():
    parsed = parse_decision(decision(T0, [call(), call("SOLUSDT", direction="short", target_pct=None)]))
    assert [c.symbol for c in parsed.calls] == ["ETHUSDT", "SOLUSDT"]
    assert parsed.calls[1].direction.sign == -1 and parsed.calls[1].target_pct is None
    assert len(parsed.sha256) == 64


def test_every_problem_is_reported_at_once():
    bad = call(weight=0.5, direction="up", sources=[], leverage=20)
    del bad["stop_pct"]
    with pytest.raises(DecisionError) as exc:
        parse_decision(decision(T0, [bad]))
    text = "\n".join(exc.value.problems)
    for expected in ("weight", "direction", "stop_pct is required", "sources", "unknown keys ['leverage']"):
        assert expected in text


@pytest.mark.parametrize("overrides, expected", [
    ({"weight": 0.10, "stop_pct": 0.15}, "risks 1.50% of equity"),
    ({"stop_pct": 0.5}, "stop_pct 0.5 outside"),
    ({"horizon_days": 30}, "horizon_days 30 outside"),
    ({"horizon_days": 2.5}, "must be an integer"),
    ({"confidence": 1.0}, "confidence 1 outside"),
    ({"weight": True}, "weight must be a number"),
    ({"symbol": "BTCUSDT_260327"}, "USDT perpetual"),
    ({"thesis": "moon"}, "thesis must be text"),
    ({"sources": ["see twitter"]}, "http(s) URLs"),
])
def test_call_limits(overrides, expected):
    with pytest.raises(DecisionError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        parse_decision(decision(T0, [call(**overrides)]))


def test_decision_level_rules():
    with pytest.raises(DecisionError, match="5 new calls, limit 4"):
        parse_decision(decision(T0, [call(f"A{i}USDT") for i in range(5)]))
    with pytest.raises(DecisionError, match="several calls on the same symbol"):
        parse_decision(decision(T0, [call(), call()]))
    with pytest.raises(DecisionError, match="0 new calls, at least 1 required every day"):
        parse_decision(decision(T0, close=[{"call_id": "2026-02-27:ETHUSDT", "reason": "x"}]))
    with pytest.raises(DecisionError, match="unknown keys \\['no_trade_reason'\\]"):
        parse_decision(decision(T0, [call()], no_trade_reason="Nothing clears the bar today."))
    with pytest.raises(DecisionError, match="as_of does not fall on the decision date"):
        parse_decision(decision(T0, [call()]) | {"date": "2026-02-28"})
    with pytest.raises(DecisionError, match="unknown keys"):
        parse_decision(decision(T0, [call()]) | {"override_limits": True})


# ---------------------------------------------------------------------- exits
def bars(rows) -> pd.DataFrame:
    index = pd.date_range("2026-03-01 10:05", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index, dtype=float)


EXPIRY = pd.Timestamp("2026-03-02", tz="UTC")


@pytest.mark.parametrize("sign, rows, expected", [
    (+1, [(100, 101, 99, 100), (100, 100, 95.5, 97)], ("stop", 96.0, 2)),            # touched: filled at the stop
    (+1, [(100, 101, 99, 100), (93, 94, 92, 93)], ("stop", 93.0, 2)),                # gapped: filled at the open
    (+1, [(100, 109, 95, 100)], ("stop", 96.0, 1)),                                  # both touched: the stop wins
    (+1, [(100, 108.5, 99, 108)], ("target", 108.0, 1)),
    (+1, [(110, 111, 109, 110)], ("target", 108.0, 1)),                              # a limit never fills better
    (-1, [(100, 104.5, 99, 100)], ("stop", 104.0, 1)),
    (-1, [(107, 108, 106, 107)], ("stop", 107.0, 1)),
    (-1, [(100, 101, 91, 92)], ("target", 92.0, 1)),
    (+1, [(100, 101, 99, 100)] * 3, None),
])
def test_find_exit(sign, rows, expected):
    found = find_exit(bars(rows), ExitRule(sign, 100.0, 0.04, 0.08, EXPIRY), STEP)
    if expected is None:
        assert found is None
        return
    reason, price, bars_elapsed = expected
    assert (found.reason, found.price) == (reason, price)
    assert found.time == pd.Timestamp("2026-03-01 10:05", tz="UTC") + bars_elapsed * STEP


def test_time_exit_is_at_the_open_of_the_first_bar_after_expiry():
    frame = bars([(100, 101, 99, 100), (101, 120, 80, 100)])
    found = find_exit(frame, ExitRule(1, 100.0, 0.04, 0.08, frame.index[1]), STEP)
    assert found == Exit(frame.index[1], 101.0, "time")


# ----------------------------------------------------------------------- book
def test_execute_opens_the_call_at_the_live_quote(book, feed, tmp_path):
    price = float(feed.frames["ETHUSDT"].loc[T0.floor(STEP), "open"])
    result = run(book, feed, T0, decision(T0, [call()]))
    assert result["status"] == "executed" and result["open_calls"] == 1
    opened = book.open_calls()[0]
    assert opened.call_id == "2026-03-01:ETHUSDT" and opened.entry_mid == pytest.approx(price)
    assert opened.weight == pytest.approx(0.05) and opened.quantity == pytest.approx(500 / price)
    assert opened.entry_price > price                                   # a buyer pays the spread and slippage
    assert 7 < opened.entry_cost_bps < 10                               # 1 spread + 2 slippage + 5 fee
    assert book.broker.positions()["ETHUSDT"] == pytest.approx(opened.quantity)
    assert 9_999 < result["equity"] < 10_000                            # only the entry cost has been paid
    assert json.loads((tmp_path / "book" / "decisions" / "2026-03-01.json").read_text())["calls"][0]["symbol"] == "ETHUSDT"
    reloaded = DiscretionaryBook(BookConfig(), feed, tmp_path / "book")
    assert reloaded.open_calls()[0].to_dict() == opened.to_dict()


def test_dry_run_commits_nothing(book, feed):
    result = run(book, feed, T0, decision(T0, [call()]), dry_run=True)
    assert result["status"] == "dry_run" and result["plan"][0]["approved_weight"] == pytest.approx(0.05)
    assert not book.state.calls and not book.state.decisions and book.broker.positions().empty


def test_a_decision_is_resumed_but_never_replaced(book, feed):
    raw = decision(T0, [call(), call("SOLUSDT")])
    run(book, feed, T0, raw)
    again = run(book, feed, T0 + pd.Timedelta(minutes=20), raw)
    assert again["opened"] == [] and len(book.open_calls()) == 2
    with pytest.raises(DecisionError, match="immutable"):
        run(book, feed, T0 + pd.Timedelta(minutes=20), decision(T0, [call("BTCUSDT")]))


def test_a_day_that_took_no_position_can_still_take_its_call(book, feed):
    """Decisions from before the one-call-a-day rule could be empty; they locked the day without trading."""
    legacy = {"date": "2026-03-01", "as_of": T0.isoformat(), "market_view": "Nothing to do today.", "calls": [],
              "close": [], "no_trade_reason": "Nothing clears the bar."}
    (book.dir / "decisions").mkdir()
    (book.dir / "decisions" / "2026-03-01.json").write_text(json.dumps(legacy))
    book.state.decisions["2026-03-01"] = {"sha256": "0" * 64, "as_of": T0.isoformat(), "executed_at": T0.isoformat()}
    result = run(book, feed, T0, decision(T0, [call()]))
    assert [c["symbol"] for c in result["opened"]] == ["ETHUSDT"]
    assert json.loads((book.dir / "decisions" / "2026-03-01.no-trade.json").read_text()) == legacy
    assert json.loads((book.dir / "decisions" / "2026-03-01.json").read_text())["calls"][0]["symbol"] == "ETHUSDT"
    assert any(e.get("message") == "decision without calls superseded" for e in book.journal.read())
    with pytest.raises(DecisionError, match="immutable"):                # once a call is taken the day is locked
        run(book, feed, T0 + pd.Timedelta(minutes=20), decision(T0, [call("SOLUSDT")]))


def test_context_rejections(book, feed):
    with pytest.raises(DecisionError, match="today is 2026-03-02"):
        run(book, feed, T0 + DAY, decision(T0, [call()]))
    stale = decision(T0, [call()]) | {"as_of": (T0 - pd.Timedelta(hours=5)).isoformat()}
    with pytest.raises(DecisionError, match="research again"):
        run(book, feed, T0, stale)
    with pytest.raises(DecisionError, match="not in today's whitelist"):
        run(book, feed, T0, decision(T0, [call("TINYUSDT")]))
    with pytest.raises(DecisionError, match="unknown call"):
        run(book, feed, T0, decision(T0, [call()], close=[{"call_id": "2026-02-27:ETHUSDT", "reason": "x"}]))
    assert not book.state.decisions                                     # a rejected decision is not committed
    run(book, feed, T0, decision(T0, [call()]))
    with pytest.raises(DecisionError, match="already open on this symbol"):
        run(book, feed, T0 + DAY, decision(T0 + DAY, [call()]))


def test_open_calls_are_capped(book, feed, tmp_path):
    capped = DiscretionaryBook(BookConfig(limits=BookLimits(max_open_calls=2)), feed, tmp_path / "capped")
    run(capped, feed, T0, decision(T0, [call(), call("SOLUSDT")]))
    with pytest.raises(DecisionError, match="2 open calls \\+ 1 new, limit 2"):
        run(capped, feed, T0 + DAY, decision(T0 + DAY, [call("BTCUSDT")]))


def test_a_stop_hit_between_sessions_is_replayed(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    opened = book.open_calls()[0]
    hit = T0.floor(STEP) + pd.Timedelta(hours=2)
    poke(feed, "ETHUSDT", hit, low=opened.entry_mid * 0.95)
    exits = refresh(book, feed, T0 + DAY)["exits"]
    assert [e["exit_reason"] for e in exits] == ["stop"]
    closed = book.state.calls[opened.call_id]
    assert closed.exit_time == (hit + STEP).isoformat()
    assert closed.exit_price < opened.entry_price * 0.96                # the stop price, minus the cost of hitting it
    assert -0.045 < closed.net_return < -0.04
    assert closed.pnl == pytest.approx(book.broker.cash() - 10_000)     # the call's PnL is the account's PnL
    assert book.broker.positions().empty and closed.shadow_open         # the mirror trade is still running
    refresh(book, feed, T0 + 6 * DAY)
    assert closed.shadow_exit_reason == "time"
    # flat market: the shadow only pays its way in and out (modelled spread 1.4, slippage 2, fee 5)
    assert closed.shadow_return == pytest.approx(-(closed.entry_cost_bps + 8.4) / 1e4, abs=2e-5)
    frame = call_frame(book.state.calls.values())
    assert frame["edge"].iloc[0] == pytest.approx(closed.weight * (closed.net_return - closed.shadow_return) / 2)


def test_a_gap_through_the_stop_fills_at_the_open(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    opened = book.open_calls()[0]
    gap = T0.floor(STEP) + pd.Timedelta(hours=3)
    feed.frames["ETHUSDT"].loc[feed.frames["ETHUSDT"].index >= gap, ["open", "high", "low", "close"]] = \
        opened.entry_mid * 0.90
    refresh(book, feed, T0 + DAY)
    closed = book.state.calls[opened.call_id]
    assert closed.exit_reason == "stop" and closed.net_return < -0.10   # worse than the 4% the stop promised
    assert closed.shadow_exit_reason == "target"                        # the mirror short took its 8%
    assert closed.shadow_return == pytest.approx(0.08 - (closed.entry_cost_bps + 8.4) / 1e4, abs=2e-5)


def test_the_bar_of_the_entry_is_not_replayed(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    poke(feed, "ETHUSDT", T0.floor(STEP), low=1.0)                      # includes prices from before the entry
    assert refresh(book, feed, T0 + DAY)["exits"] == []


def test_target_and_time_exits(book, feed):
    run(book, feed, T0, decision(T0, [call(), call("SOLUSDT", direction="short", target_pct=None, horizon_days=2)]))
    eth = book.state.calls["2026-03-01:ETHUSDT"]
    poke(feed, "ETHUSDT", T0.floor(STEP) + pd.Timedelta(hours=5), high=eth.entry_mid * 1.09)
    exits = {e["call_id"]: e for e in refresh(book, feed, T0 + 3 * DAY)["exits"]}
    assert exits["2026-03-01:ETHUSDT"]["exit_reason"] == "target"
    assert 0.075 < eth.net_return < 0.08
    sol = book.state.calls["2026-03-01:SOLUSDT"]
    assert sol.exit_reason == "time" and sol.exit_time == (T0 + 2 * DAY).ceil(STEP).isoformat()
    assert sol.net_return == pytest.approx(-(sol.entry_cost_bps + 5 + 2 + 1.41) / 1e4, abs=2e-5)


def test_funding_is_paid_once_by_the_call_and_received_by_its_shadow(book, feed):
    settle = pd.Timestamp("2026-03-01 16:00", tz="UTC")
    feed.funding["ETHUSDT"] = pd.DataFrame({"funding_rate": [0.001, 0.001]},
                                           index=[settle - pd.Timedelta(hours=8), settle])
    run(book, feed, T0, decision(T0, [call(horizon_days=1)]))
    opened = book.open_calls()[0]
    refresh(book, feed, T0 + pd.Timedelta(hours=7))
    refresh(book, feed, T0 + pd.Timedelta(hours=9))
    assert opened.funding_paid == pytest.approx(0.001 * opened.quantity * opened.entry_mid)   # only the 16:00 event
    assert opened.shadow_funding_return == pytest.approx(0.001)
    refresh(book, feed, T0 + 2 * DAY)
    assert opened.exit_reason == "time" and opened.shadow_exit_reason == "time"
    assert opened.shadow_return - opened.net_return == pytest.approx(0.002, abs=2e-4)


def test_a_manual_close_ends_the_call_and_its_shadow(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    second = decision(T0 + DAY, [call("SOLUSDT")], close=[{"call_id": "2026-03-01:ETHUSDT",
                                                          "reason": "thesis invalidated"}])
    result = run(book, feed, T0 + DAY, second)
    closed = book.state.calls["2026-03-01:ETHUSDT"]
    assert result["closed"][0]["exit_reason"] == "manual" and closed.exit_note == "thesis invalidated"
    assert closed.shadow_exit_reason == "manual" and list(book.broker.positions().index) == ["SOLUSDT"]
    # same instant, same costs, flat market: call and shadow are indistinguishable, so the edge is zero
    assert closed.net_return == pytest.approx(closed.shadow_return, abs=1e-6)
    # asking again for a call that is no longer open is a no-op, not an error
    rerun = run(book, feed, T0 + DAY + pd.Timedelta(minutes=5), second)
    assert rerun["closed"] == [] and rerun["opened"] == []


def test_the_risk_engine_shrinks_a_volatile_call_or_drops_it(book, feed, tmp_path):
    wild = decision(T0, [call("WILDUSDT", weight=0.10, stop_pct=0.10)])
    opened = run(book, feed, T0, wild)["opened"][0]
    assert 0.01 < opened["weight"] < 0.09 and "gap-risk cap" in opened["risk_reasons"][0]
    strict = DiscretionaryBook(BookConfig(limits=BookLimits(min_approved_weight=0.09)), feed, tmp_path / "strict")
    result = run(strict, feed, T0, wild)
    assert result["opened"] == [] and "gap-risk cap" in result["blocked"][0]["reasons"][0]
    assert strict.broker.positions().empty and "2026-03-01" in strict.state.decisions


def test_daily_loss_blocks_new_risk(book, feed):
    book.state.day, book.state.day_start_equity = T0.floor("1D").isoformat(), 10_500.0   # the day is down 4.8%
    result = run(book, feed, T0, decision(T0, [call(), call("SOLUSDT")]))
    assert result["opened"] == [] and len(result["blocked"]) == 2
    assert "daily loss" in result["blocked"][0]["reasons"][0] and not book.state.calls
    tomorrow = run(book, feed, T0 + DAY, decision(T0 + DAY, [call()]))                    # a new day starts afresh
    assert len(tomorrow["opened"]) == 1


def test_kill_file_flattens_and_blocks_until_a_human_resets(book, feed):
    run(book, feed, T0, decision(T0, [call(), call("SOLUSDT")]))
    book.kill_file.touch()
    state = refresh(book, feed, T0 + pd.Timedelta(hours=2))
    assert state["killed"] and {e["exit_reason"] for e in state["exits"]} == {"kill"}
    assert book.broker.positions().empty
    assert all(not c.shadow_open for c in book.state.calls.values())
    assert run(book, feed, T0 + DAY, decision(T0 + DAY, [call("BTCUSDT")]))["status"] == "killed"
    feed.clock = T0 + DAY
    book.reset_kill("owner", T0 + DAY)
    assert not book.kill_file.exists()
    assert run(book, feed, T0 + DAY, decision(T0 + DAY, [call("BTCUSDT")]))["status"] == "executed"


def test_drawdown_trips_the_kill_switch(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    book.state.equity_peak = 12_000.0                                   # the book once stood 20% higher
    state = refresh(book, feed, T0 + pd.Timedelta(hours=1))
    assert state["killed"] and "drawdown" in state["kill_reason"]
    assert book.state.calls["2026-03-01:ETHUSDT"].exit_reason == "kill" and state["open_calls"] == 0


def test_a_position_the_book_does_not_know_halts_everything(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    book.state.calls["2026-03-01:ETHUSDT"].quantity *= 2
    with pytest.raises(BookError, match="reconciliation mismatch"):
        refresh(book, feed, T0 + DAY)
    assert book.state.kill_switch


# -------------------------------------------------------------------- scoring
def test_sign_flip_pvalue():
    assert sign_flip_pvalue(np.array([1.0, 1.0, 1.0])) == pytest.approx(1 / 8)
    assert sign_flip_pvalue(np.array([1.0, -1.0])) == pytest.approx(3 / 4)
    assert np.isnan(sign_flip_pvalue(np.array([])))
    rng = np.random.default_rng(3)
    assert sign_flip_pvalue(rng.normal(0.0, 1.0, 400)) > 0.05           # noise is not skill
    assert sign_flip_pvalue(rng.normal(0.3, 1.0, 400)) < 0.001


def test_scorecard_separates_a_right_call_from_a_market_move(book, feed):
    run(book, feed, T0, decision(T0, [call(confidence=0.7), call("SOLUSDT", direction="short", confidence=0.7)]))
    for symbol in ("ETHUSDT", "SOLUSDT"):                               # both rally 10%: the long was right
        frame = feed.frames[symbol]
        frame.loc[frame.index >= T0.floor(STEP) + pd.Timedelta(hours=4), ["open", "high", "low", "close"]] *= 1.10
    refresh(book, feed, T0 + 6 * DAY)
    frame = call_frame(book.state.calls.values()).set_index("symbol")
    assert frame.loc["ETHUSDT", "exit_reason"] == "target" and frame.loc["SOLUSDT", "exit_reason"] == "stop"
    assert frame.loc["ETHUSDT", "edge"] > 0 > frame.loc["SOLUSDT", "edge"]
    score = scorecard(book.state.calls.values(), book.equity_points())
    assert score["closed_calls"] == score["scored_calls"] == 2 and score["hit_rate"] == 0.5
    assert score["brier"] == pytest.approx((0.3 ** 2 + 0.7 ** 2) / 2)
    assert score["long_share"] == 0.5 and 0.25 <= score["p_value"] <= 1.0
    assert not score["eligible"] and not score["gate"][0]["passed"]
    assert "NON ÉLIGIBLE" in render_scorecard(score, call_frame(book.state.calls.values()))


def test_an_empty_book_scores_without_crashing(book, feed):
    refresh(book, feed, T0)
    score = scorecard(book.state.calls.values(), book.equity_points())
    assert score["closed_calls"] == 0 and not score["eligible"] and score["total_return"] == 0.0
    assert "NON ÉLIGIBLE" in render_scorecard(score, call_frame(book.state.calls.values()))


def test_brief_shows_the_book_and_the_market(book, feed):
    run(book, feed, T0, decision(T0, [call()]))
    now = T0 + pd.Timedelta(hours=6)
    text = render_brief(build_brief(book, now, refresh(book, feed, now)))
    assert "2026-03-01:ETHUSDT LONG 5.0%" in text and "invalidation : The catalyst is cancelled." in text
    assert "TINYUSDT" not in text and "WILDUSDT" in text                # the whitelist is by liquidity
    assert "Une décision a déjà été exécutée aujourd'hui" in text
