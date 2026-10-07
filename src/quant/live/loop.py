"""Paper trading loop: Strategy -> Risk Engine -> Portfolio (orders) -> Broker.

One ``step`` per closed strategy bar:
 1. kill switch (file or risk state) -> flatten and stop;
 2. idempotency: a bar already processed is skipped; a late step (machine asleep, API outage)
    is skipped rather than trading on a stale decision;
 3. reconciliation: broker positions must equal the positions rebuilt from the fill ledger;
 4. point-in-time universe + bars (Vision history + REST tail), strategy weights for the last closed bar;
 5. funding settlements, risk evaluation (every order gets its risk metrics), orders, fills at the
    observed bid/ask, journal, state persistence.

The same ``Strategy`` objects as in research are used unchanged. Swapping ``PaperBroker`` for a
live broker would not change this loop, but no live broker exists (see brokers/live.py).

Trading rule (mirrors the backtester's ``rebalance="on_change"``): a symbol is traded only when its
target weight changes. Between two decisions of the strategy the book drifts with prices and is left
alone; re-aligning it every bar costs turnover the backtest never paid for.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from quant.brokers.base import Broker, DuplicateOrderError
from quant.data.market_data import MarketData, resample_bars
from quant.data.timeframes import bars_per_day, interval_to_timedelta
from quant.data.universe import UniverseRule, expand_to_bars, point_in_time_eligibility
from quant.data.validation import mask_stale_runs
from quant.execution.orders import Order, Side, client_order_id
from quant.live.feed import SPOT_SUFFIX, MarketFeed
from quant.monitoring.journal import EventJournal
from quant.risk.engine import Action, MarketSnapshot, RiskEngine, RiskLimits, RiskState
from quant.strategies.base import Strategy


@dataclass
class LoopConfig:
    strategy_id: str
    strategy: Strategy
    interval: str
    universe: UniverseRule
    candidate_symbols: list[str]
    history_days: int = 240
    max_candidates: int = 80
    min_order_notional: float = 5.0
    max_decision_delay: pd.Timedelta = pd.Timedelta(minutes=45)
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    rebalance_tolerance: float = 0.005  # skip orders that change a weight by less than this (tracking vs cost)
    shadow_note: str = ""
    pair_mode: bool = False   # trade SYMBOL and SYMBOL@spot together (basis strategies)
    with_metrics: bool = False  # attach open interest / positioning frames to the market data
    metric_fields: list[str] | None = None  # metric columns the strategy reads (None = all endpoints)
    drift_leverage_tolerance: float = 0.10  # re-target everything if gross drifts this far above the cap
    # "on_change": trade a symbol only when its target changes (what the research validated).
    # "drift": also pull a position back to its target weight once it has drifted by
    #          ``rebalance_tolerance`` (the backtester's ``rebalance="every_bar"``).
    rebalance_mode: str = "on_change"


@dataclass
class LoopState:
    last_decision_bar: str | None = None
    funding_applied_until: dict = field(default_factory=dict)
    equity_peak: float | None = None
    day_start_equity: float | None = None
    day: str | None = None
    kill_switch: bool = False
    kill_reason: str = ""
    venue: str | None = None     # which broker produced the equity history below
    last_acted_change_bar: str | None = None       # last strategy decision the loop has traded
    last_acted_targets: dict = field(default_factory=dict)   # the target weights it traded towards

    @classmethod
    def load(cls, path: Path) -> "LoopState":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)


class TradingLoop:
    def __init__(self, config: LoopConfig, feed: MarketFeed, broker: Broker, state_dir: Path,
                 journal: EventJournal | None = None):
        if config.rebalance_mode not in ("on_change", "drift"):
            raise ValueError(f"unknown rebalance_mode {config.rebalance_mode!r}")
        self.cfg = config
        self.feed = feed
        self.broker = broker
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "loop_state.json"
        self.kill_file = self.state_dir / "KILL"
        self.journal = journal or EventJournal(self.state_dir / "events.jsonl")
        self.state = LoopState.load(self.state_path)
        self._check_venue()

    def _check_venue(self) -> None:
        """Equity history belongs to one account. Carrying a peak from a 10,000 paper account into a
        5,000 testnet account reads as a -50% drawdown and trips the kill switch on the first step."""
        venue = type(self.broker).__name__
        if self.state.venue == venue:
            return
        if self.state.venue is not None:
            self.journal.alert("venue changed, resetting equity history",
                               previous=self.state.venue, current=venue,
                               previous_peak=self.state.equity_peak)
            self.state.equity_peak = None
            self.state.day_start_equity = None
            self.state.kill_switch = False
            self.state.kill_reason = ""
        self.state.venue = venue
        self._persist()

    # ------------------------------------------------------------------ helpers
    def _persist(self) -> None:
        self.state.save(self.state_path)

    def _risk_engine(self, equity: float, now: pd.Timestamp) -> RiskEngine:
        st = self.state
        risk_state = RiskState(equity_peak=st.equity_peak or equity, day_start_equity=st.day_start_equity or equity,
                               day=pd.Timestamp(st.day) if st.day else now.floor("1D"),
                               kill_switch=st.kill_switch, kill_reason=st.kill_reason)
        return RiskEngine(self.cfg.risk_limits, risk_state)

    def _store_risk_state(self, engine: RiskEngine) -> None:
        rs = engine.state
        self.state.equity_peak = rs.equity_peak
        self.state.day_start_equity = rs.day_start_equity
        self.state.day = rs.day.isoformat()
        self.state.kill_switch = rs.kill_switch
        self.state.kill_reason = rs.kill_reason

    def _reconcile(self) -> bool:
        ledger = getattr(self.broker, "ledger_positions", None)
        if ledger is None:
            return True
        book = self.broker.positions()
        rebuilt = ledger()
        symbols = set(book.index) | set(rebuilt.index)
        diffs = {s: (float(book.get(s, 0.0)), float(rebuilt.get(s, 0.0))) for s in symbols
                 if abs(book.get(s, 0.0) - rebuilt.get(s, 0.0)) > 1e-9 * max(1.0, abs(book.get(s, 0.0)))}
        if diffs:
            self.journal.log("reconciliation_mismatch", diffs=diffs)
            self.state.kill_switch = True
            self.state.kill_reason = "reconciliation mismatch between broker state and fill ledger"
            self._persist()
            return False
        return True

    def _load_data(self, decision_bar: pd.Timestamp, now: pd.Timestamp) -> tuple[MarketData, dict[str, pd.DataFrame]]:
        start = decision_bar - pd.Timedelta(days=self.cfg.history_days)
        symbols = self.cfg.candidate_symbols
        try:
            volume = self.feed.quote_volume_24h().reindex(symbols).fillna(0.0)
            symbols = list(volume.sort_values(ascending=False).index[: self.cfg.max_candidates])
        except Exception as exc:  # the loop can still run on the configured list
            self.journal.alert("24h volume unavailable; using configured candidates", error=str(exc))
        # a held symbol must stay priced and ranked even on a day its volume drops out of the shortlist
        held = [s for s in self.broker.positions().index if not s.endswith(SPOT_SUFFIX)]
        symbols = symbols + [s for s in held if s not in symbols and s in self.cfg.candidate_symbols]
        if self.cfg.pair_mode:
            symbols = symbols + [s + "@spot" for s in symbols]
        raw = self.feed.bars(symbols, "1h", start, now)
        bars = {}
        for sym, df in raw.items():
            cleaned, _ = mask_stale_runs(df)
            cleaned = cleaned.dropna(subset=["close"])
            if len(cleaned):
                bars[sym] = resample_bars(cleaned, "1h", self.cfg.interval)
        bars = {s: b for s, b in bars.items() if len(b)}
        try:
            funding = self.feed.funding_history(list(bars), start, now)
        except Exception as exc:
            self.journal.alert("funding history unavailable", error=str(exc))
            funding = None
        tradable_at_venue = self.broker.supported_symbols()
        if tradable_at_venue is not None:
            dropped = [s for s in bars if s.split(SPOT_SUFFIX)[0] not in tradable_at_venue]
            if dropped:
                self.journal.alert("symbols not listed at the venue", count=len(dropped),
                                   examples=sorted(dropped)[:8])
                bars = {s: f for s, f in bars.items() if s not in dropped}
        data = MarketData.from_frames(bars, self.cfg.interval, funding)
        if self.cfg.with_metrics:
            try:
                frames = self.feed.metrics_history(list(bars), start, now, self.cfg.interval,
                                                   fields=self.cfg.metric_fields)
            except Exception as exc:
                self.journal.alert("metrics history unavailable", error=str(exc))
                frames = {}
            for warning in getattr(self.feed, "last_metrics_warnings", []):
                self.journal.alert("stale metrics", detail=warning)
            ages = pd.Series(getattr(self.feed, "last_snapshot_age_seconds", {}), dtype=float)
            self._metrics_age = (float(ages.median()), float(ages.max())) if len(ages) else None
            bar_seconds = interval_to_timedelta(self.cfg.interval).total_seconds()
            late = ages[ages >= bar_seconds]
            if len(late):   # research always had the snapshot taken at the bar close
                self.journal.alert("positioning data older than one bar", symbols=sorted(late.index)[:10],
                                   worst_seconds=float(late.max()))
            if frames:
                data = data.with_extras(**frames)
        daily_qv = pd.DataFrame({s: df["quote_volume"].resample("1D").sum(min_count=1) for s, df in raw.items()
                                 if len(df) and not s.endswith("@spot")})
        mask = point_in_time_eligibility(daily_qv, self.cfg.universe)
        daily = mask.reindex(columns=[c.removesuffix("@spot") for c in data.symbols], fill_value=False)
        daily.columns = data.symbols
        if self.cfg.pair_mode:
            spot_cols = [c for c in data.symbols if c.endswith("@spot")]
            spot_qv = pd.DataFrame({c: raw[c]["quote_volume"].resample("1D").sum(min_count=1) for c in spot_cols})
            alive = (spot_qv.rolling(7, min_periods=1).median() > 0).shift(1, fill_value=False)
            daily.loc[:, spot_cols] = daily[spot_cols] & alive.reindex(daily.index).fillna(False).astype(bool)
        return data.with_universe(expand_to_bars(daily, data.index)), raw

    def _flatten(self, now: pd.Timestamp, reason: str) -> list[dict]:
        positions = self.broker.positions()
        if positions.empty:
            return []
        quotes = self.feed.quotes(list(positions.index))
        fills = []
        for sym, qty in positions.items():
            q = quotes.get(sym)
            if q is None:
                self.journal.alert("cannot flatten: no quote", symbol=sym)
                continue
            side = Side.SELL if qty > 0 else Side.BUY
            order = Order(client_order_id(f"{self.cfg.strategy_id}:KILL", sym, now.floor("1min"), -qty), sym, side,
                          abs(qty), now, q.mid, reduce_only=True, strategy_id=self.cfg.strategy_id, reason=reason)
            try:
                fill = self.broker.submit(order, q.mid, now, bar_quote_volume=float("inf"),
                                          observed_half_spread_bps=q.half_spread_bps)
                if fill:
                    fills.append(fill.to_dict())
            except DuplicateOrderError:
                pass
        self.journal.log("kill_switch", reason=reason, fills=fills)
        return fills

    # --------------------------------------------------------------------- step
    def step(self, now: pd.Timestamp | None = None) -> dict:
        now = now or pd.Timestamp.now(tz="UTC")
        bar = interval_to_timedelta(self.cfg.interval)
        decision_bar = now.floor(bar) - bar
        decision_close = decision_bar + bar

        if self.kill_file.exists() or self.state.kill_switch:
            reason = self.state.kill_reason or f"kill file {self.kill_file}"
            self.state.kill_switch = True
            self.state.kill_reason = reason
            fills = self._flatten(now, reason)
            self._persist()
            return {"status": "killed", "reason": reason, "fills": len(fills)}

        if self.state.last_decision_bar and pd.Timestamp(self.state.last_decision_bar) >= decision_bar:
            return {"status": "already_processed", "decision_bar": decision_bar}

        if now - decision_close > self.cfg.max_decision_delay:
            self.journal.alert("late step skipped", decision_bar=decision_bar, delay=str(now - decision_close))
            self.state.last_decision_bar = decision_bar.isoformat()
            self._persist()
            return {"status": "late_skipped", "decision_bar": decision_bar}

        if not self._reconcile():
            return {"status": "halted", "reason": "reconciliation mismatch"}

        data, raw = self._load_data(decision_bar, now)
        if data.index[-1] != decision_bar:
            self.journal.alert("latest closed bar missing; step skipped", expected=decision_bar, got=data.index[-1])
            return {"status": "data_incomplete", "decision_bar": decision_bar}
        weights = self.cfg.strategy.target_weights(data)
        change_bar = self._last_change_bar(weights)
        targets = weights.iloc[-1]
        targets = targets[targets.abs() > 1e-12]

        positions = self.broker.positions()
        symbols = sorted(set(targets.index) | set(positions.index))
        quotes = self.feed.quotes(symbols) if symbols else {}
        missing = [s for s in symbols if s not in quotes]
        if missing:
            self.journal.alert("no live quote; symbols excluded this step", symbols=missing)
        mids = pd.Series({s: q.mid for s, q in quotes.items()}, dtype=float)
        marks = mids.combine_first(data.close.ffill().iloc[-1])
        equity = self.broker.equity(marks) if len(marks) else self.broker.cash()

        funding_paid = self._settle_funding(positions, marks, now)
        # re-read rather than subtract: a venue has already debited funding from the balance it reports
        equity = self.broker.equity(marks) if len(marks) else self.broker.cash()

        engine = self._risk_engine(equity, now)
        current = (positions.reindex(symbols).fillna(0.0) * marks.reindex(symbols)) / equity

        last_acted = pd.Timestamp(self.state.last_acted_change_bar) if self.state.last_acted_change_bar else None
        rebalance_due = change_bar is not None and (last_acted is None or change_bar > last_acted)
        gross_now = float(current.abs().sum())
        drifted = gross_now > self.cfg.risk_limits.max_gross_leverage * (1.0 + self.cfg.drift_leverage_tolerance)
        acted_targets = self.state.last_acted_targets or {}
        tradable_symbols = [s for s in symbols if s in quotes]
        snapshot = MarketSnapshot.from_hourly_bars(raw, tradable_symbols, mids)
        decisions = engine.evaluate(equity, current.reindex(tradable_symbols).fillna(0.0),
                                    targets.reindex(tradable_symbols).fillna(0.0), snapshot, now)
        self._store_risk_state(engine)

        fills, blocked = [], 0
        bpd = bars_per_day("1h")
        for sym, d in decisions.items():
            self.journal.log("risk_decision", decision_bar=decision_bar, **{k: v for k, v in asdict(d).items()
                                                                          if k != "action"}, action=d.action.value)
            if d.action in (Action.BLOCK,):
                blocked += 1
            target_w = float(targets.get(sym, 0.0))
            has_position = float(positions.get(sym, 0.0)) != 0.0
            target_changed = abs(target_w - float(acted_targets.get(sym, 0.0))) > 1e-9
            # a position nobody asked for (state reset, manual trade) or a target that never got opened
            # (blocked or unquoted last time) is settled at the next decision, never left to linger
            orphan = has_position and target_w == 0.0
            unopened = not has_position and target_w != 0.0
            forced = d.action is Action.FLATTEN or drifted
            if self.cfg.rebalance_mode == "drift":
                eligible = True   # the tolerance check below decides: any position off target is pulled back
            else:
                eligible = rebalance_due and (target_changed or orphan or unopened)
            if not forced and not eligible:
                continue  # the strategy has not changed its mind about this symbol: let it drift
            delta_w = d.approved_weight - float(current.get(sym, 0.0))
            closing = d.approved_weight == 0.0 and float(positions.get(sym, 0.0)) != 0.0
            if abs(delta_w) < self.cfg.rebalance_tolerance and not closing:
                continue
            q = quotes[sym]
            qty_target = d.approved_weight * equity / q.mid
            dq = qty_target - float(positions.get(sym, 0.0))
            if abs(dq) * q.mid < self.cfg.min_order_notional and not closing:
                continue
            side = Side.BUY if dq > 0 else Side.SELL
            reduce_only = abs(qty_target) < abs(float(positions.get(sym, 0.0))) and qty_target * positions.get(sym, 0.0) >= 0
            order = Order(client_order_id(self.cfg.strategy_id, sym, decision_close, qty_target), sym, side, abs(dq),
                          decision_close, float(data.close[sym].iloc[-1]) if sym in data.symbols else q.mid,
                          reduce_only=bool(reduce_only), strategy_id=self.cfg.strategy_id,
                          reason=f"target {d.approved_weight:+.4f} ({d.action.value})")
            bar_sigma = float(snapshot.daily_vol.get(sym, np.nan)) / math.sqrt(bpd)
            try:
                fill = self.broker.submit(order, q.mid, now, bar_quote_volume=float(snapshot.avg_bar_quote_volume.get(sym, np.nan)),
                                          bar_sigma=bar_sigma, bars_per_day=bpd,
                                          observed_half_spread_bps=q.half_spread_bps)
            except DuplicateOrderError:
                self.journal.log("duplicate_order_skipped", client_order_id=order.client_order_id)
                continue
            if fill:
                self.state.funding_applied_until.setdefault(sym, now.isoformat())
                rec = fill.to_dict()
                rec["slippage_vs_decision_close_bps"] = (fill.price / order.reference_price - 1) * 1e4 * (1 if side is Side.BUY else -1)
                rec["latency_seconds"] = (now - decision_close).total_seconds()
                fills.append(rec)

        if rebalance_due:
            self.state.last_acted_change_bar = change_bar.isoformat()
            self.state.last_acted_targets = {s: float(w) for s, w in targets.items()}

        marks_after = marks.copy()
        equity_after = self.broker.equity(marks_after)
        gross_after = float(sum(abs(q_) * marks_after.get(s, np.nan) for s, q_ in self.broker.positions().items())) / equity_after
        self.state.last_decision_bar = decision_bar.isoformat()
        self._persist()
        summary = {"status": "traded", "strategy_id": self.cfg.strategy_id, "decision_bar": decision_bar,
                   "universe_size": int(data.universe.iloc[-1].sum()), "n_targets": int(len(targets)),
                   "n_orders_filled": len(fills), "n_blocked": blocked, "equity": equity_after,
                   "gross_exposure": gross_after, "funding_paid": funding_paid,
                   "rebalanced": bool(rebalance_due), "strategy_change_bar": change_bar,
                   "rebalance_mode": self.cfg.rebalance_mode,
                   "metrics_age_s": getattr(self, "_metrics_age", None),   # (median, max) age at the bar close
                   "kill_switch": self.state.kill_switch, "note": self.cfg.shadow_note}
        self.journal.log("step_summary", **summary, fills=fills)
        return summary

    @staticmethod
    def _last_change_bar(weights: pd.DataFrame) -> pd.Timestamp | None:
        """Most recent bar whose target row differs from the previous bar's. The backtester trades
        there and nowhere else, and recomputing the strategy on a sliding window leaves this bar
        unchanged even when the shortlist of candidates moves between two steps."""
        values = weights.to_numpy(dtype=float)
        if len(values) == 0:
            return None
        if len(values) == 1:
            return weights.index[0] if np.any(values[0] != 0.0) else None
        changed = np.any(np.abs(values[1:] - values[:-1]) > 1e-12, axis=1)
        idx = np.flatnonzero(changed)
        return weights.index[idx[-1] + 1] if len(idx) else None

    def _settle_funding(self, positions: pd.Series, marks: pd.Series, now: pd.Timestamp) -> float:
        if getattr(self.broker, "settles_funding_itself", False):
            # the venue debits funding on its own; only report what it charged since the last step
            try:
                return float(self.broker.funding_paid_since_last_check(now))
            except Exception as exc:
                self.journal.alert("venue funding query failed", error=str(exc))
                return 0.0
        paid = 0.0
        for sym, qty in positions.items():
            since = pd.Timestamp(self.state.funding_applied_until.get(sym)) if sym in self.state.funding_applied_until \
                else (pd.Timestamp(self.state.last_decision_bar) if self.state.last_decision_bar else now)
            try:
                events = self.feed.funding_since(sym, since + pd.Timedelta(milliseconds=1), now)
            except Exception as exc:
                self.journal.alert("funding fetch failed", symbol=sym, error=str(exc))
                continue
            for ts, row in events.iterrows():
                paid += self.broker.apply_funding(sym, float(row["funding_rate"]), float(marks.get(sym, np.nan)), ts)
                self.state.funding_applied_until[sym] = ts.isoformat()
        return paid
