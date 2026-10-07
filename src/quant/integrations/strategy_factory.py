"""Read-only client for the public Strategy Factory / Trader.dev endpoints.

What is public (no account): leaderboard search, backtest summaries, trade lists
and equity curves. Pine source code requires the owner's or a paid API key and is
NOT used here. Consequently strategies are re-implemented from their described
logic, and every re-implementation must record its source ``resultId``.

Reported metrics come from a TradingView-style bar emulator with 0.05% commission,
no slippage and 100%-of-equity sizing: treat them as claims to falsify, not evidence.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field

import pandas as pd
import requests

BROWSE_API = "https://mcp-api.strategyfactory.ai"
REPORT_API = "https://mcp-api.trader.dev"


@dataclass
class StrategyFactoryClient:
    timeout: float = 30.0
    session: requests.Session = field(default_factory=requests.Session)

    def _json(self, url: str, params: dict | None = None) -> dict:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> dict:
        return self._json(f"{BROWSE_API}/strategies/stats")

    def search(self, sort: str = "sharpe", limit: int = 16, offset: int = 0, **filters) -> list[dict]:
        """Filters: symbol, timeframe, minNetProfitPct, minProfitFactor, minSharpeRatio, minSortinoRatio,
        minWinRatePct, maxDrawdownPct, minTrades, maxTrades. Sort: profit|sharpe|sortino|winrate|trades|drawdown|recent."""
        params = {"sort": sort, "limit": limit, "offset": offset, **{k: v for k, v in filters.items() if v is not None}}
        return self._json(f"{BROWSE_API}/strategies/search", params)["results"]

    def result(self, result_id: str) -> dict:
        return self._json(f"{REPORT_API}/backtest-results/{result_id}")

    def report_blob(self, result: dict) -> dict:
        resp = self.session.get(result["r2Url"], timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.content
        try:
            return json.loads(gzip.decompress(raw))
        except OSError:
            return json.loads(raw)

    def trades(self, result_id: str) -> tuple[dict, pd.DataFrame]:
        """Summary and closed trades (times as UTC bar-open timestamps)."""
        summary = self.result(result_id)
        blob = self.report_blob(summary)
        df = pd.DataFrame(blob.get("trades", []))
        if df.empty:
            return summary, df
        df = df[~df["isOpen"]].copy()
        df["entry_time"] = pd.to_datetime(df["entryTime"], unit="ms", utc=True)
        df["exit_time"] = pd.to_datetime(df["exitTime"], unit="ms", utc=True)
        df = df.rename(columns={"entryPrice": "entry_price", "exitPrice": "exit_price", "profitPct": "profit_pct"})
        return summary, df[["direction", "entry_time", "entry_price", "exit_time", "exit_price", "profit_pct", "barsInTrade"]]
