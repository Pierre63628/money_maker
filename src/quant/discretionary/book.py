"""Discretionary book: daily directional calls on a paper account, inside limits the author cannot move.

One session per UTC day:
  refresh  replay the closed 5-minute bars since the last session (stops, targets, time exits,
           funding), mark the book, let the risk engine see the equity;
  execute  check the day's decision against the book and the market, pass it through the risk
           engine, close and open at live quotes.

Nothing runs between sessions, so exits are reconstructed from bars instead of being watched live. A
stop fills at the stop price, or at the bar open when the market gapped through it; when one bar
touches both the stop and the target, the stop wins. The bar in which a call was opened is not
replayed, because its range includes prices from before the entry: up to five minutes are unseen.

Every call carries a mirror trade (the "shadow"): opposite direction from the same mid, same stop,
target and expiry. It is what a coin flip would have earned on the same symbol at the same moment,
and ``scoring.py`` judges the calls against it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from quant.backtesting.costs import CostModel
from quant.brokers.base import DuplicateOrderError
from quant.brokers.paper import PaperBroker
from quant.data.timeframes import interval_to_timedelta
from quant.discretionary.calls import BookLimits, Call, CallRequest, Decision, DecisionError, SYMBOL_RE
from quant.execution.orders import Order, Side, client_order_id
from quant.live.feed import MarketFeed, Quote
from quant.monitoring.journal import EventJournal
from quant.risk.engine import MarketSnapshot, RiskEngine, RiskLimits, RiskState

REPLAY_INTERVAL = "5m"
BENCHMARK = "BTCUSDT"


class BookError(RuntimeError):
    pass


@dataclass(frozen=True)
class BookConfig:
    book_id: str = "directional-v1"
    initial_capital: float = 10_000.0
    limits: BookLimits = field(default_factory=BookLimits)
    risk_limits: RiskLimits = field(default_factory=lambda: RiskLimits(
        max_gross_leverage=0.6, max_net_leverage=0.4, max_position_weight=0.10, max_daily_loss=0.03,
        max_drawdown=0.15, max_cluster_weight=0.30, min_daily_quote_volume=50e6))
    costs: CostModel = field(default_factory=lambda: CostModel(slippage_bps=2.0, spread_model="volume"))
    stop_slippage_bps: float = 5.0     # a stop is a market order into a move that is already running
    history_days: int = 70


@dataclass(frozen=True)
class ExitRule:
    sign: int                  # +1 long, -1 short
    entry: float
    stop_pct: float
    target_pct: float | None
    expiry: pd.Timestamp

    @property
    def stop(self) -> float:
        return self.entry * (1 - self.sign * self.stop_pct)

    @property
    def target(self) -> float | None:
        return None if self.target_pct is None else self.entry * (1 + self.sign * self.target_pct)


@dataclass(frozen=True)
class Exit:
    time: pd.Timestamp
    price: float
    reason: str                # stop | target | time | manual | kill


def find_exit(bars: pd.DataFrame, rule: ExitRule, step: pd.Timedelta) -> Exit | None:
    """First exit of ``rule`` over ``bars`` (indexed by open time), or None while the trade is alive.

    Stop and target exits are stamped at the bar close because the moment inside the bar is unknown;
    a time exit happens at the open of the first bar at or after the expiry.
    """
    stop, target = rule.stop, rule.target
    for ts, open_, high, low in zip(bars.index, bars["open"], bars["high"], bars["low"]):
        if ts >= rule.expiry:
            return Exit(ts, float(open_), "time")
        adverse, favourable = (low, high) if rule.sign > 0 else (high, low)
        if rule.sign * (adverse - stop) <= 0:
            gapped = rule.sign * (open_ - stop) <= 0
            return Exit(ts + step, float(open_ if gapped else stop), "stop")
        if target is not None and rule.sign * (favourable - target) >= 0:
            return Exit(ts + step, float(target), "target")   # a resting limit: never better than its price
    return None


@dataclass
class BookState:
    calls: dict[str, Call] = field(default_factory=dict)
    decisions: dict[str, dict] = field(default_factory=dict)   # date -> hash of the executed decision
    equity_peak: float | None = None
    day_start_equity: float | None = None
    day: str | None = None
    kill_switch: bool = False
    kill_reason: str = ""

    @classmethod
    def load(cls, path: Path) -> "BookState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        state = cls(**{k: v for k, v in raw.items() if k in known})
        state.calls = {k: Call.from_dict(v) for k, v in state.calls.items()}
        return state

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))
        tmp.replace(path)


class DiscretionaryBook:
    def __init__(self, config: BookConfig, feed: MarketFeed, state_dir: str | Path):
        self.cfg, self.feed = config, feed
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.broker = PaperBroker(self.dir / "broker", config.costs, initial_capital=config.initial_capital)
        self.journal = EventJournal(self.dir / "events.jsonl")
        self.state_path = self.dir / "book.json"
        self.equity_path = self.dir / "equity.jsonl"
        self.kill_file = self.dir / "KILL"
        self.state = BookState.load(self.state_path)

    # ------------------------------------------------------------------ views
    def open_calls(self) -> list[Call]:
        return sorted((c for c in self.state.calls.values() if c.is_open), key=lambda c: c.entry_time)

    def equity_points(self) -> list[dict]:
        if not self.equity_path.exists():
            return []
        return [json.loads(line) for line in self.equity_path.read_text().splitlines() if line.strip()]

    def whitelist(self) -> list[str]:
        """What may be traded today: the most liquid USDT perpetuals of the venue, by 24h volume."""
        lim = self.cfg.limits
        volume = self.feed.quote_volume_24h()
        volume = volume[[s for s in volume.index if SYMBOL_RE.match(s)]]
        volume = volume[volume >= lim.min_quote_volume_24h].sort_values(ascending=False)
        return list(volume.index[: lim.whitelist_top_n])

    def hourly_bars(self, symbols: list[str], now: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return self.feed.bars(symbols, "1h", now - pd.Timedelta(days=self.cfg.history_days), now)

    def marks(self, symbols: list[str]) -> tuple[dict[str, Quote], pd.Series]:
        """Live quotes, and a price per symbol that falls back on the last replayed bar of an open call."""
        quotes = self.feed.quotes(sorted(set(symbols))) if symbols else {}
        mids = pd.Series({s: q.mid for s, q in quotes.items()}, dtype=float)
        last = pd.Series({c.symbol: c.last_price for c in self.open_calls()}, dtype=float)
        return quotes, mids.combine_first(last)

    def weights(self, calls: list[Call], marks: pd.Series, equity: float) -> pd.Series:
        return pd.Series({c.symbol: c.sign * c.quantity * marks[c.symbol] / equity for c in calls}, dtype=float)

    # ---------------------------------------------------------------- plumbing
    def _save(self) -> None:
        self.state.save(self.state_path)

    def _risk_engine(self, equity: float, now: pd.Timestamp) -> RiskEngine:
        st = self.state
        return RiskEngine(self.cfg.risk_limits, RiskState(
            equity_peak=st.equity_peak or equity, day_start_equity=st.day_start_equity or equity,
            day=pd.Timestamp(st.day) if st.day else now.floor("1D"),
            kill_switch=st.kill_switch, kill_reason=st.kill_reason))

    def _store_risk_state(self, engine: RiskEngine) -> None:
        rs = engine.state
        self.state.equity_peak, self.state.day_start_equity = rs.equity_peak, rs.day_start_equity
        self.state.day = rs.day.isoformat()
        self.state.kill_switch, self.state.kill_reason = rs.kill_switch, rs.kill_reason

    def _reconcile(self) -> None:
        """The open calls, the broker state and the broker's fill ledger must describe the same positions."""
        expected: dict[str, float] = {}
        for c in self.open_calls():
            expected[c.symbol] = expected.get(c.symbol, 0.0) + c.sign * c.quantity
        held, ledger = self.broker.positions(), self.broker.ledger_positions()
        diffs = {}
        for s in set(expected) | set(held.index) | set(ledger.index):
            e, h, l = expected.get(s, 0.0), float(held.get(s, 0.0)), float(ledger.get(s, 0.0))
            if max(abs(e - h), abs(h - l)) > 1e-9 * max(1.0, abs(e)):
                diffs[s] = {"calls": e, "broker": h, "ledger": l}
        if diffs:
            self.state.kill_switch = True
            self.state.kill_reason = "open calls, broker state and fill ledger disagree"
            self._save()
            self.journal.log("reconciliation_mismatch", diffs=diffs)
            raise BookError(f"reconciliation mismatch, book halted: {diffs}")

    def _modelled_half_spread(self, symbol: str, bars: pd.DataFrame) -> float:
        per_day = pd.Timedelta(days=1) / interval_to_timedelta(REPLAY_INTERVAL)
        daily = float(bars["quote_volume"].iloc[-int(per_day):].mean() * per_day)
        return float(self.cfg.costs.half_spread_matrix([symbol], np.array([[daily]]))[0, 0])

    # ------------------------------------------------------------ open / close
    def _open(self, decision: Decision, req: CallRequest, weight: float, reasons: list[str], quote: Quote,
              equity: float, snapshot: MarketSnapshot, now: pd.Timestamp) -> Call:
        sign = req.direction.sign
        call_id = f"{decision.date}:{req.symbol}"
        order = Order(client_order_id(f"{self.cfg.book_id}:open", call_id, decision.as_of, sign * req.weight),
                      req.symbol, Side.BUY if sign > 0 else Side.SELL, weight * equity / quote.mid, decision.as_of,
                      quote.mid, strategy_id=self.cfg.book_id, reason=f"call {call_id}")
        try:
            fill = self.broker.submit(
                order, quote.mid, now,
                bar_quote_volume=float(snapshot.avg_bar_quote_volume.get(req.symbol, np.nan)),
                bar_sigma=float(snapshot.daily_vol.get(req.symbol, np.nan)) / math.sqrt(24),
                observed_half_spread_bps=quote.half_spread_bps)
        except DuplicateOrderError as exc:
            raise BookError(f"entry order for {call_id} was already sent but the call is missing from the book; "
                            f"inspect {self.dir / 'broker' / 'journal.jsonl'}") from exc
        step = interval_to_timedelta(REPLAY_INTERVAL)
        call = Call(
            call_id=call_id, decision_date=decision.date, symbol=req.symbol, direction=req.direction.value,
            requested_weight=req.weight, weight=fill.quantity * quote.mid / equity, stop_pct=req.stop_pct,
            target_pct=req.target_pct, horizon_days=req.horizon_days, confidence=req.confidence, thesis=req.thesis,
            invalidation=req.invalidation, sources=list(req.sources), risk_reasons=list(reasons),
            entry_time=now.isoformat(), entry_mid=quote.mid, entry_price=fill.price, quantity=fill.quantity,
            entry_fee=fill.fee,
            entry_cost_bps=sign * (fill.price / quote.mid - 1) * 1e4 + fill.fee / (fill.quantity * fill.price) * 1e4,
            equity_at_entry=equity, expiry=(now + pd.Timedelta(days=req.horizon_days)).isoformat(),
            replayed_until=now.ceil(step).isoformat(), last_price=quote.mid)
        self.state.calls[call_id] = call
        self._save()
        self.journal.log("call_opened", **call.to_dict())
        return call

    def _exit_cost_bps(self, symbol: str, reason: str, half_spread_bps: float) -> float:
        """Price concession an exit pays on top of the fee: spread, slippage, and more when it is a stop."""
        return half_spread_bps + self.cfg.costs.slippage_bps + (self.cfg.stop_slippage_bps if reason == "stop" else 0.0)

    def _close_shadow(self, call: Call, exit_: Exit, half_spread_bps: float) -> None:
        """The shadow pays what the call would have paid for the same exit, or the comparison is biased."""
        call.shadow_exit_time, call.shadow_exit_price = exit_.time.isoformat(), exit_.price
        call.shadow_exit_reason = exit_.reason
        call.shadow_cost_bps = (call.entry_cost_bps + self._exit_cost_bps(call.symbol, exit_.reason, half_spread_bps)
                                + self.cfg.costs.fee_bps_for(call.symbol))

    def _close(self, call: Call, exit_: Exit, half_spread_bps: float, note: str = "") -> dict:
        order = Order(client_order_id(f"{self.cfg.book_id}:close", call.call_id, pd.Timestamp(call.entry_time),
                                      call.quantity),
                      call.symbol, Side.SELL if call.sign > 0 else Side.BUY, call.quantity, exit_.time, exit_.price,
                      reduce_only=True, strategy_id=self.cfg.book_id, reason=exit_.reason)
        # the paper broker adds its own slippage_bps to whatever spread it is given
        concession = self._exit_cost_bps(call.symbol, exit_.reason, half_spread_bps) - self.cfg.costs.slippage_bps
        fill = self.broker.submit(order, exit_.price, exit_.time, bar_quote_volume=float("inf"),
                                  observed_half_spread_bps=concession)
        if fill is None or abs(fill.quantity - call.quantity) > 1e-9 * call.quantity:
            raise BookError(f"{call.call_id}: exit did not close the whole position")
        call.status, call.exit_time, call.exit_price = "closed", exit_.time.isoformat(), fill.price
        call.exit_reason, call.exit_note, call.exit_fee, call.last_price = exit_.reason, note, fill.fee, exit_.price
        call.pnl = (call.sign * call.quantity * (fill.price - call.entry_price)
                    - call.entry_fee - fill.fee - call.funding_paid)
        if exit_.reason in ("manual", "kill") and call.shadow_open:
            self._close_shadow(call, exit_, half_spread_bps)   # a discretionary exit ends both legs together
        self._save()
        record = {"call_id": call.call_id, "symbol": call.symbol, "direction": call.direction,
                  "exit_reason": exit_.reason, "exit_note": note, "exit_time": call.exit_time,
                  "exit_price": fill.price, "pnl": call.pnl, "net_return": call.net_return,
                  "r_multiple": call.net_return / call.stop_pct}
        self.journal.log("call_closed", **record)
        return record

    # ------------------------------------------------------------------ replay
    def _settle_funding(self, call: Call, bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                        real: Exit | None, shadow: Exit | None) -> None:
        entry = pd.Timestamp(call.entry_time)
        first_pass = start <= entry.ceil(interval_to_timedelta(REPLAY_INTERVAL))
        try:
            events = self.feed.funding_since(call.symbol, entry if first_pass else start, end)
        except Exception as exc:  # funding is a second-order term: a failed fetch must not block the exits
            self.journal.alert("funding fetch failed", symbol=call.symbol, error=str(exc))
            return
        for ts, rate in events["funding_rate"].items():
            if not (entry < ts < end) or not math.isfinite(rate):
                continue
            mark = bars["open"].asof(ts)
            mark = float(mark) if pd.notna(mark) else call.entry_mid
            if call.is_open and (real is None or ts < real.time):
                call.funding_paid += self.broker.apply_funding(call.symbol, float(rate), mark, ts)
            if call.shadow_open and (shadow is None or ts < shadow.time):
                call.shadow_funding_return += call.sign * float(rate) * mark / call.entry_mid

    def settle(self, now: pd.Timestamp) -> list[dict]:
        """Replay the closed bars since the last session, for the calls and for their shadows."""
        self._reconcile()
        step = interval_to_timedelta(REPLAY_INTERVAL)
        closed = []
        for call in sorted(self.state.calls.values(), key=lambda c: c.entry_time):
            if not (call.is_open or call.shadow_open):
                continue
            start = pd.Timestamp(call.replayed_until)
            bars = self.feed.bars([call.symbol], REPLAY_INTERVAL, start, now).get(call.symbol)
            bars = bars[bars.index >= start] if bars is not None else None
            if bars is None or bars.empty:
                continue
            end, expiry = bars.index[-1] + step, pd.Timestamp(call.expiry)
            real = find_exit(bars, ExitRule(call.sign, call.entry_price, call.stop_pct, call.target_pct, expiry),
                             step) if call.is_open else None
            shadow = find_exit(bars, ExitRule(-call.sign, call.entry_mid, call.stop_pct, call.target_pct, expiry),
                               step) if call.shadow_open else None
            self._settle_funding(call, bars, start, end, real, shadow)
            half_spread = self._modelled_half_spread(call.symbol, bars)
            if shadow:
                self._close_shadow(call, shadow, half_spread)
            if real:
                closed.append(self._close(call, real, half_spread))
            elif call.is_open:
                call.last_price = float(bars["close"].iloc[-1])
            call.replayed_until = end.isoformat()
            self._save()
        return closed

    # ------------------------------------------------------------ kill and mark
    def kill(self, now: pd.Timestamp, reason: str) -> list[dict]:
        """Close every open call at live quotes and refuse new ones until a human resets the switch."""
        self.state.kill_switch, self.state.kill_reason = True, reason
        self._save()
        calls = self.open_calls()
        quotes, _ = self.marks([c.symbol for c in calls])
        closed = []
        for call in calls:
            quote = quotes.get(call.symbol)
            if quote is None:
                self.journal.alert("cannot flatten: no quote", call_id=call.call_id)
                continue
            closed.append(self._close(call, Exit(now, quote.mid, "kill"), quote.half_spread_bps, note=reason))
        self.journal.log("kill_switch", reason=reason, closed=[c["call_id"] for c in closed])
        return closed

    def reset_kill(self, operator: str, now: pd.Timestamp) -> None:
        """A human decision, never an automatic one: re-arms the book and restarts the drawdown from here."""
        self.kill_file.unlink(missing_ok=True)
        equity = self.broker.equity(self.marks([c.symbol for c in self.open_calls()])[1])
        self.journal.alert("kill switch reset", operator=operator, previous_reason=self.state.kill_reason,
                           equity=equity, previous_peak=self.state.equity_peak)
        self.state.kill_switch, self.state.kill_reason = False, ""
        self.state.equity_peak = self.state.day_start_equity = equity
        self.state.day = now.floor("1D").isoformat()
        self._save()

    def mark(self, now: pd.Timestamp) -> dict:
        """Mark the book at live quotes, show the equity to the risk engine, record the point."""
        calls = self.open_calls()
        _, marks = self.marks([c.symbol for c in calls] + [BENCHMARK])
        equity = self.broker.equity(marks)
        weights = self.weights(calls, marks, equity)
        engine = self._risk_engine(equity, now)
        empty = pd.Series(dtype=float)
        engine.evaluate(equity, weights, weights, MarketSnapshot(marks, empty, empty, empty, pd.DataFrame(dtype=float)),
                        now)
        self._store_risk_state(engine)
        if self.state.kill_switch and calls:
            self.kill(now, self.state.kill_reason)
            equity = self.broker.equity(marks)
            weights = self.weights(self.open_calls(), marks, equity)
        point = {"ts": now.isoformat(), "equity": float(equity), "cash": self.broker.cash(),
                 "gross": float(weights.abs().sum()), "net": float(weights.sum()), "open_calls": len(weights),
                 "benchmark": float(marks.get(BENCHMARK, np.nan)), "kill_switch": self.state.kill_switch}
        with self.equity_path.open("a") as fh:
            fh.write(json.dumps(point) + "\n")
        self._save()
        return point

    def refresh(self, now: pd.Timestamp) -> dict:
        """Bring the book up to ``now``: kill file, replay, mark, drawdown check."""
        if self.kill_file.exists() and not self.state.kill_switch:
            self.state.kill_switch, self.state.kill_reason = True, f"kill file {self.kill_file.name}"
        exits = self.kill(now, self.state.kill_reason) if self.state.kill_switch and self.open_calls() else []
        exits += self.settle(now)      # after a kill this still finishes the shadows of calls closed earlier
        point = self.mark(now)
        return {"exits": exits, "killed": self.state.kill_switch, "kill_reason": self.state.kill_reason, **point}

    # ----------------------------------------------------------------- execute
    def _context_problems(self, decision: Decision, pending: list[CallRequest], staying: list[Call],
                          whitelist: list[str], quotes: dict[str, Quote], now: pd.Timestamp) -> list[str]:
        lim, problems = self.cfg.limits, []
        if decision.date != now.strftime("%Y-%m-%d"):
            problems.append(f"decision dated {decision.date}, today is {now:%Y-%m-%d} (UTC)")
        age_hours = (now - decision.as_of).total_seconds() / 3600
        if age_hours > lim.max_decision_age_hours:
            problems.append(f"decision is {age_hours:.1f}h old, limit {lim.max_decision_age_hours:g}h: research again")
        if age_hours < -5 / 60:
            problems.append("as_of is in the future")
        for request in decision.closes:
            if request.call_id not in self.state.calls:
                problems.append(f"close: unknown call {request.call_id}")
        if len(staying) + len(pending) > lim.max_open_calls:
            problems.append(f"{len(staying)} open calls + {len(pending)} new, limit {lim.max_open_calls}")
        held = {c.symbol for c in staying}
        for req in pending:
            if req.symbol in held:
                problems.append(f"{req.symbol}: a call is already open on this symbol")
            if req.symbol not in whitelist:
                problems.append(f"{req.symbol}: not in today's whitelist (top {lim.whitelist_top_n} perpetuals "
                                f"with at least {lim.min_quote_volume_24h / 1e6:.0f}M USDT of 24h volume)")
            elif req.symbol not in quotes:
                problems.append(f"{req.symbol}: no live quote")
        return problems

    def committed_without_calls(self, date: str) -> bool:
        """True for a day whose executed decision took no position. Such decisions predate the rule that
        every session ends with a call; they changed nothing in the book, so one may still be replaced."""
        archive = self.dir / "decisions" / f"{date}.json"
        if not archive.exists():
            return False
        raw = json.loads(archive.read_text())
        return not raw.get("calls") and not raw.get("close")

    def execute(self, decision: Decision, now: pd.Timestamp | None = None, dry_run: bool = False) -> dict:
        """Trade one day's decision. ``dry_run`` stops after the risk engine: nothing is committed or sent.

        Re-running the same decision resumes it (calls already opened are skipped); a different
        decision for a day that already has one is refused, unless that one took no position.
        """
        now = now or pd.Timestamp.now(tz="UTC")
        status = self.refresh(now)
        if status["killed"]:
            return {"status": "killed", "reason": status["kill_reason"], "exits": status["exits"]}
        committed = self.state.decisions.get(decision.date)
        if committed and committed["sha256"] != decision.sha256:
            if not self.committed_without_calls(decision.date):
                raise DecisionError([f"another decision was already executed for {decision.date}; "
                                     "decisions are immutable"])
            committed = None     # superseded when this one is committed, below

        pending = [c for c in decision.calls if f"{decision.date}:{c.symbol}" not in self.state.calls]
        closing = {r.call_id: r for r in decision.closes
                   if r.call_id in self.state.calls and self.state.calls[r.call_id].is_open}
        staying = [c for c in self.open_calls() if c.call_id not in closing]
        symbols = sorted({c.symbol for c in self.open_calls()} | {c.symbol for c in pending} | {BENCHMARK})
        quotes, marks = self.marks(symbols)
        problems = self._context_problems(decision, pending, staying, self.whitelist() if pending else [], quotes, now)
        if problems:
            raise DecisionError(problems)

        closed = []
        if not dry_run:
            if not committed:
                archive = self.dir / "decisions" / f"{decision.date}.json"
                archive.parent.mkdir(exist_ok=True)
                previous = self.state.decisions.get(decision.date)
                if previous:     # kept next to its replacement: nothing that was decided is erased
                    archive.replace(archive.with_suffix(".no-trade.json"))
                    self.journal.alert("decision without calls superseded", decision_date=decision.date,
                                       previous_sha256=previous["sha256"], sha256=decision.sha256)
                self.state.decisions[decision.date] = {"sha256": decision.sha256, "as_of": decision.as_of.isoformat(),
                                                       "executed_at": now.isoformat()}
                archive.write_text(json.dumps(decision.raw, indent=1, ensure_ascii=False))
                self._save()
            for call_id, request in closing.items():
                call = self.state.calls[call_id]
                quote = quotes.get(call.symbol)
                if quote is None:
                    self.journal.alert("cannot close: no quote", call_id=call_id)
                    staying.append(call)
                    continue
                closed.append(self._close(call, Exit(now, quote.mid, "manual"), quote.half_spread_bps,
                                          note=request.reason))

        equity = self.broker.equity(marks)
        current = self.weights(staying, marks, equity)
        targets = current.copy()
        for req in pending:
            targets[req.symbol] = req.direction.sign * req.weight
        tradable = sorted(targets.index)
        snapshot = MarketSnapshot.from_hourly_bars(self.hourly_bars(tradable, now), tradable, marks)
        engine = self._risk_engine(equity, now)
        decisions = engine.evaluate(equity, current, targets, snapshot, now)

        opened, blocked, plan = [], [], []
        for req in pending:
            verdict = decisions[req.symbol]
            approved = verdict.approved_weight * req.direction.sign
            row = {"symbol": req.symbol, "direction": req.direction.value, "requested_weight": req.weight,
                   "approved_weight": max(approved, 0.0), "action": verdict.action.value, "reasons": verdict.reasons}
            plan.append(row)
            if approved < self.cfg.limits.min_approved_weight:
                blocked.append(row)
                if not dry_run:
                    self.journal.log("call_blocked", decision_date=decision.date, **row)
            elif not dry_run:
                opened.append(self._open(decision, req, approved, verdict.reasons, quotes[req.symbol], equity,
                                         snapshot, now))
        if dry_run:
            return {"status": "dry_run", "plan": plan, "would_close": sorted(closing), "equity": equity,
                    "exits": status["exits"]}

        self._store_risk_state(engine)
        point = self.mark(now)
        summary = {"status": "executed", "decision_date": decision.date, "exits": status["exits"], "closed": closed,
                   "opened": [c.to_dict() for c in opened], "blocked": blocked, **point}
        self.journal.log("session", decision_date=decision.date, sha256=decision.sha256,
                         opened=[c.call_id for c in opened], closed=[c["call_id"] for c in closed],
                         blocked=[b["symbol"] for b in blocked], equity=point["equity"], gross=point["gross"],
                         net=point["net"])
        return summary
