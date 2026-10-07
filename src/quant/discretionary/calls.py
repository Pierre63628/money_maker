"""Discretionary calls: the contract between the author of a daily decision and the code that trades it.

The author reads the open web before writing a decision, so the decision file is untrusted input:
every bound in ``BookLimits`` is checked here or by the risk engine, never left to the author, and a
decision that breaks one is rejected whole rather than clipped. Once executed a decision is
immutable: the book keeps its hash and refuses a different file for the same day.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path

import pandas as pd

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,24}USDT$")      # perpetuals only: dated contracts carry an underscore
DECISION_KEYS = {"date", "as_of", "market_view", "calls", "close"}
CALL_KEYS = {"symbol", "direction", "weight", "stop_pct", "target_pct", "horizon_days", "confidence", "thesis",
             "invalidation", "sources"}


class DecisionError(ValueError):
    """Every problem found in a decision, reported together so it is rewritten once."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("decision rejected:\n  - " + "\n  - ".join(problems))


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


@dataclass(frozen=True)
class BookLimits:
    max_open_calls: int = 8
    min_new_calls_per_day: int = 1           # the owner's rule: a session always ends with a call taken
    max_new_calls_per_day: int = 4
    max_position_weight: float = 0.10
    max_risk_per_call: float = 0.01          # weight * stop distance, as a fraction of equity
    min_stop_pct: float = 0.01
    max_stop_pct: float = 0.15
    max_target_pct: float = 0.60
    max_horizon_days: int = 14
    min_confidence: float = 0.05
    max_confidence: float = 0.95
    whitelist_top_n: int = 40                # by 24h quote volume on the venue
    min_quote_volume_24h: float = 50e6
    min_thesis_chars: int = 40
    max_decision_age_hours: float = 3.0      # a decision executed later than this trades on stale information
    min_approved_weight: float = 0.01        # a call the risk engine shrinks below this is dropped, not traded as dust


@dataclass(frozen=True)
class CallRequest:
    symbol: str
    direction: Direction
    weight: float                # fraction of equity, before the risk engine
    stop_pct: float              # adverse move from the entry that closes the call
    target_pct: float | None
    horizon_days: int
    confidence: float            # stated probability that the call closes with a net gain
    thesis: str
    invalidation: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CloseRequest:
    call_id: str
    reason: str


@dataclass(frozen=True)
class Decision:
    date: str
    as_of: pd.Timestamp
    market_view: str
    calls: tuple[CallRequest, ...]
    closes: tuple[CloseRequest, ...]
    sha256: str
    raw: dict


def _number(obj: dict, key: str, lo: float, hi: float, where: str, problems: list[str], *,
            optional: bool = False, integer: bool = False) -> float | None:
    value = obj.get(key)
    if value is None:
        if not optional:
            problems.append(f"{where}: {key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or (integer and int(value) != value):
        problems.append(f"{where}: {key} must be {'an integer' if integer else 'a number'}, got {value!r}")
        return None
    if not lo <= value <= hi:
        problems.append(f"{where}: {key} {value:g} outside [{lo:g}, {hi:g}]")
        return None
    return float(value)


def _text(obj: dict, key: str, min_chars: int, where: str, problems: list[str]) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or len(value.strip()) < min_chars:
        problems.append(f"{where}: {key} must be text of at least {min_chars} characters")
        return ""
    return value.strip()


def _parse_call(raw: object, index: int, limits: BookLimits, problems: list[str]) -> CallRequest | None:
    where = f"calls[{index}]"
    if not isinstance(raw, dict):
        problems.append(f"{where}: must be an object")
        return None
    where = f"{where} {raw.get('symbol', '?')}"
    before = len(problems)
    if set(raw) - CALL_KEYS:
        problems.append(f"{where}: unknown keys {sorted(set(raw) - CALL_KEYS)}")
    symbol = raw.get("symbol")
    if not isinstance(symbol, str) or not SYMBOL_RE.match(symbol):
        problems.append(f"{where}: symbol must be a USDT perpetual such as BTCUSDT")
    if raw.get("direction") not in ("long", "short"):
        problems.append(f"{where}: direction must be 'long' or 'short'")
    weight = _number(raw, "weight", 1e-6, limits.max_position_weight, where, problems)
    stop = _number(raw, "stop_pct", limits.min_stop_pct, limits.max_stop_pct, where, problems)
    target = _number(raw, "target_pct", 1e-6, limits.max_target_pct, where, problems, optional=True)
    horizon = _number(raw, "horizon_days", 1, limits.max_horizon_days, where, problems, integer=True)
    confidence = _number(raw, "confidence", limits.min_confidence, limits.max_confidence, where, problems)
    if weight is not None and stop is not None and weight * stop > limits.max_risk_per_call + 1e-12:
        problems.append(f"{where}: weight {weight:.1%} x stop {stop:.1%} risks {weight * stop:.2%} of equity, "
                        f"limit {limits.max_risk_per_call:.2%}")
    thesis = _text(raw, "thesis", limits.min_thesis_chars, where, problems)
    invalidation = _text(raw, "invalidation", 10, where, problems)
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(s, str) and re.match(r"^https?://\S+$", s) for s in sources):
        problems.append(f"{where}: sources must be a non-empty list of http(s) URLs")
    if len(problems) > before:
        return None
    return CallRequest(symbol, Direction(raw["direction"]), weight, stop, target, int(horizon), confidence, thesis,
                       invalidation, tuple(sources))


def parse_decision(source: str | Path | dict, limits: BookLimits = BookLimits()) -> Decision:
    """Parse and validate everything that needs no market data. Raises ``DecisionError``."""
    raw = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if not isinstance(raw, dict):
        raise DecisionError(["a decision is a JSON object"])
    problems: list[str] = []
    if set(raw) - DECISION_KEYS:
        problems.append(f"unknown keys {sorted(set(raw) - DECISION_KEYS)}")

    date, as_of = raw.get("date"), None
    if not isinstance(date, str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        problems.append("date must be YYYY-MM-DD (UTC)")
    try:
        as_of = pd.Timestamp(raw.get("as_of"))
        if as_of is pd.NaT or as_of.tzinfo is None:
            raise ValueError
        as_of = as_of.tz_convert("UTC")
        if isinstance(date, str) and as_of.strftime("%Y-%m-%d") != date:
            problems.append("as_of does not fall on the decision date (UTC)")
    except (ValueError, TypeError):
        problems.append("as_of must be an ISO timestamp with a timezone")
    market_view = _text(raw, "market_view", 20, "decision", problems)

    calls_raw, closes_raw = raw.get("calls", []), raw.get("close", [])
    if not isinstance(calls_raw, list) or not isinstance(closes_raw, list):
        problems.append("calls and close must be lists")
        calls_raw, closes_raw = [], []
    if len(calls_raw) > limits.max_new_calls_per_day:
        problems.append(f"{len(calls_raw)} new calls, limit {limits.max_new_calls_per_day} per day")
    # Conviction is expressed through weight and confidence, not by abstaining: a weak day is a small
    # call at a confidence near one half, and it is scored like any other.
    if len(calls_raw) < limits.min_new_calls_per_day:
        problems.append(f"{len(calls_raw)} new calls, at least {limits.min_new_calls_per_day} required every day")
    calls = [c for c in (_parse_call(item, i, limits, problems) for i, item in enumerate(calls_raw)) if c]
    symbols = [c.symbol for c in calls]
    for dup in sorted({s for s in symbols if symbols.count(s) > 1}):
        problems.append(f"{dup}: several calls on the same symbol")

    closes = []
    for i, item in enumerate(closes_raw):
        if not isinstance(item, dict) or set(item) != {"call_id", "reason"} \
                or not all(isinstance(v, str) and v.strip() for v in item.values()):
            problems.append(f"close[{i}]: must be {{call_id, reason}} with non-empty text")
            continue
        closes.append(CloseRequest(item["call_id"].strip(), item["reason"].strip()))

    if problems:
        raise DecisionError(problems)
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return Decision(date, as_of, market_view, tuple(calls), tuple(closes), digest, raw)


@dataclass
class Call:
    """One call in the book, from entry to exit, with its mirror trade (see ``scoring.py``)."""

    call_id: str
    decision_date: str
    symbol: str
    direction: str
    requested_weight: float
    weight: float                 # what the risk engine approved and the broker filled
    stop_pct: float
    target_pct: float | None
    horizon_days: int
    confidence: float
    thesis: str
    invalidation: str
    sources: list[str]
    risk_reasons: list[str]
    entry_time: str
    entry_mid: float
    entry_price: float
    quantity: float               # base units, unsigned
    entry_fee: float
    entry_cost_bps: float         # spread, slippage, impact and fee paid to get in, per unit of notional
    equity_at_entry: float
    expiry: str
    replayed_until: str           # bars opening before this are already replayed, for both legs
    last_price: float
    funding_paid: float = 0.0
    status: str = "open"
    exit_time: str | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    exit_note: str = ""
    exit_fee: float = 0.0
    pnl: float | None = None      # quote currency, net of fees and funding
    shadow_exit_time: str | None = None
    shadow_exit_price: float | None = None
    shadow_exit_reason: str = ""
    shadow_funding_return: float = 0.0
    shadow_cost_bps: float = 0.0

    @property
    def sign(self) -> int:
        return Direction(self.direction).sign

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def shadow_open(self) -> bool:
        return self.shadow_exit_time is None

    @property
    def net_return(self) -> float:
        """Net result per unit of entry notional (NaN while the call is open)."""
        return float("nan") if self.pnl is None else self.pnl / (self.quantity * self.entry_mid)

    @property
    def shadow_return(self) -> float:
        """Net result of the mirror trade per unit of notional (NaN while it is open)."""
        if self.shadow_exit_price is None:
            return float("nan")
        gross = -self.sign * (self.shadow_exit_price / self.entry_mid - 1.0)
        return gross + self.shadow_funding_return - self.shadow_cost_bps / 1e4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Call":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
