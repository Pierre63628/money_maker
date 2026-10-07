"""Is there skill in the calls, or only a market that moved?

Net PnL cannot answer on its own: twenty shorts in a falling month look like skill. Each call is
therefore compared with its shadow, the mirror trade taken from the same mid with the same stop,
target and expiry. If the direction of a call carries no information, call and shadow are
exchangeable, so swapping them at random, call by call, gives the distribution of the book's result
under pure chance. The p-value is the share of those swaps that do at least as well as the book.

One limit of the test: calls made the same day in the same direction are not independent bets, so
the p-value flatters a one-sided book. ``long_share`` and the benchmark return are reported next to
it for that reason.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quant.discretionary.calls import Call

COLUMNS = ["call_id", "symbol", "direction", "weight", "confidence", "entry_time", "exit_time", "exit_reason",
           "net_return", "r_multiple", "contribution", "shadow_return", "edge"]


@dataclass(frozen=True)
class PromotionGate:
    """Fixed before the first call and not to be tuned afterwards.

    Passing makes the book eligible for a conversation with the account owner about real money. It
    authorises nothing: live trading stays disabled in code (``brokers/live.py``).
    """

    min_scored_calls: int = 100
    min_days: int = 60
    max_p_value: float = 0.05
    max_brier: float = 0.25          # what always answering 50% would score
    max_drawdown: float = 0.15


def call_frame(calls: Iterable[Call]) -> pd.DataFrame:
    """One row per closed call. ``edge`` is the call's result in excess of chance, in fraction of equity."""
    rows = [{
        "call_id": c.call_id, "symbol": c.symbol, "direction": c.direction, "weight": c.weight,
        "confidence": c.confidence, "entry_time": pd.Timestamp(c.entry_time), "exit_time": pd.Timestamp(c.exit_time),
        "exit_reason": c.exit_reason, "net_return": c.net_return, "r_multiple": c.net_return / c.stop_pct,
        "contribution": c.pnl / c.equity_at_entry, "shadow_return": c.shadow_return,
        "edge": c.weight * (c.net_return - c.shadow_return) / 2,
    } for c in calls if not c.is_open]
    return pd.DataFrame(rows, columns=COLUMNS)


def sign_flip_pvalue(edges: np.ndarray, draws: int = 20_000, seed: int = 0) -> float:
    """One-sided p-value of ``sum(edges)`` under random sign flips; exact while 2**n stays small."""
    edges = np.asarray(edges, dtype=float)
    if len(edges) == 0:
        return float("nan")
    observed = edges.sum()
    if len(edges) <= 16:
        sums = np.array(list(itertools.product((-1.0, 1.0), repeat=len(edges)))) @ edges
        return float((sums >= observed - 1e-15).mean())
    sums = np.random.default_rng(seed).choice((-1.0, 1.0), size=(draws, len(edges))) @ edges
    return float(((sums >= observed - 1e-15).sum() + 1) / (draws + 1))


def scorecard(calls: Iterable[Call], equity_points: list[dict], gate: PromotionGate = PromotionGate()) -> dict:
    calls = list(calls)
    frame = call_frame(calls)
    scored = frame.dropna(subset=["shadow_return"])
    wins = (frame["net_return"] > 0).astype(float)
    out = {
        "open_calls": sum(c.is_open for c in calls), "closed_calls": len(frame), "scored_calls": len(scored),
        "hit_rate": float(wins.mean()) if len(frame) else float("nan"),
        "mean_return": float(frame["net_return"].mean()) if len(frame) else float("nan"),
        "mean_r": float(frame["r_multiple"].mean()) if len(frame) else float("nan"),
        "contribution": float(frame["contribution"].sum()),
        "mean_confidence": float(frame["confidence"].mean()) if len(frame) else float("nan"),
        "brier": float(((frame["confidence"] - wins) ** 2).mean()) if len(frame) else float("nan"),
        "long_share": float((frame["direction"] == "long").mean()) if len(frame) else float("nan"),
        "edge": float(scored["edge"].sum()),
        "p_value": sign_flip_pvalue(scored["edge"].to_numpy()),
        "exits": frame["exit_reason"].value_counts().to_dict(),
    }
    equity = pd.DataFrame(equity_points)
    if len(equity):
        curve = equity["equity"].astype(float)
        bench = equity["benchmark"].astype(float).dropna()
        first_entry = min((pd.Timestamp(c.entry_time) for c in calls), default=None)
        out |= {
            "equity": float(curve.iloc[-1]), "total_return": float(curve.iloc[-1] / curve.iloc[0] - 1),
            "max_drawdown": float((curve / curve.cummax() - 1).min()),
            "benchmark_return": float(bench.iloc[-1] / bench.iloc[0] - 1) if len(bench) else float("nan"),
            "days": (pd.Timestamp(equity["ts"].iloc[-1]) - first_entry).days if first_entry is not None else 0,
            "since": equity["ts"].iloc[0], "until": equity["ts"].iloc[-1],
        }
    else:
        out |= {"equity": float("nan"), "total_return": float("nan"), "max_drawdown": 0.0,
                "benchmark_return": float("nan"), "days": 0, "since": None, "until": None}
    checks = [
        (f"au moins {gate.min_scored_calls} paris clos et comparés à leur miroir", out["scored_calls"] >= gate.min_scored_calls,
         f"{out['scored_calls']}"),
        (f"au moins {gate.min_days} jours de suivi", out["days"] >= gate.min_days, f"{out['days']} j"),
        (f"p-value de l'excès sur le hasard ≤ {gate.max_p_value:g}", bool(out["p_value"] <= gate.max_p_value),
         f"{out['p_value']:.3f}"),
        ("rendement net positif", bool(out["total_return"] > 0), f"{out['total_return']:+.2%}"),
        (f"score de Brier < {gate.max_brier:g}", bool(out["brier"] < gate.max_brier), f"{out['brier']:.3f}"),
        (f"drawdown maximal moins profond que -{gate.max_drawdown:.0%}", out["max_drawdown"] > -gate.max_drawdown,
         f"{out['max_drawdown']:.2%}"),
    ]
    out["gate"] = [{"check": label, "passed": bool(ok), "value": value} for label, ok, value in checks]
    out["eligible"] = all(item["passed"] for item in out["gate"])
    return out


def calibration(frame: pd.DataFrame) -> pd.DataFrame:
    """Stated confidence against realised hit rate, by bucket."""
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]
    wins = (frame["net_return"] > 0).astype(float)
    grouped = pd.DataFrame({"confidence": frame["confidence"], "win": wins}).groupby(
        pd.cut(frame["confidence"], bins, right=False), observed=True)
    return grouped.agg(paris=("win", "size"), confiance=("confidence", "mean"), réussite=("win", "mean"))


def render_scorecard(score: dict, frame: pd.DataFrame) -> str:
    lines = ["LIVRE DIRECTIONNEL : tableau de score (paper, prix mainnet)"]
    if score["since"]:
        lines.append(f"du {score['since'][:10]} au {score['until'][:10]} ({score['days']} j) | "
                     f"équité {score['equity']:,.2f} USDT | rendement {score['total_return']:+.2%} | "
                     f"drawdown max {score['max_drawdown']:.2%} | BTC sur la période {score['benchmark_return']:+.2%}")
    lines.append(f"paris : {score['closed_calls']} clos, {score['open_calls']} ouverts, "
                 f"{score['scored_calls']} comparés à leur miroir")
    if score["closed_calls"]:
        lines.append(f"réussite {score['hit_rate']:.0%} (confiance moyenne annoncée {score['mean_confidence']:.0%}) | "
                     f"rendement moyen {score['mean_return']:+.2%} du notionnel | R moyen {score['mean_r']:+.2f} | "
                     f"part de longs {score['long_share']:.0%}")
        lines.append(f"Brier {score['brier']:.3f} (0,250 = répondre 50 % à chaque fois) | sorties : "
                     + ", ".join(f"{k} {v}" for k, v in score["exits"].items()))
    if score["scored_calls"]:
        lines.append(f"excès sur le hasard : {score['edge']:+.2%} de l'équité, p-value {score['p_value']:.3f} "
                     "(test par inversion de signe contre les paris miroirs)"
                     + ("" if score["scored_calls"] >= 30 else " : trop peu de paris pour que ce chiffre signifie "
                                                               "quoi que ce soit"))
    lines.append("")
    lines.append("Porte fixée avant le premier pari (la franchir n'autorise rien, elle ouvre la discussion) :")
    for item in score["gate"]:
        lines.append(f"  [{'x' if item['passed'] else ' '}] {item['check']} : {item['value']}")
    lines.append("=> " + ("ÉLIGIBLE à une discussion sur le réel" if score["eligible"] else "NON ÉLIGIBLE"))
    if len(frame) >= 10:
        lines += ["", "Calibration :", calibration(frame).to_string(float_format=lambda x: f"{x:.2f}")]
    if len(frame):
        table = frame.sort_values("exit_time").tail(15).assign(
            entry_time=lambda f: f["entry_time"].dt.strftime("%m-%d %H:%M"),
            exit_time=lambda f: f["exit_time"].dt.strftime("%m-%d %H:%M"))
        lines += ["", "Derniers paris clos :",
                  table.drop(columns=["call_id"]).to_string(index=False, float_format=lambda x: f"{x:+.4f}")]
    return "\n".join(lines)
