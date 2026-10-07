"""Daily directional calls on a paper book, at live Binance USDT-M prices. No account, no key, no real order.

    .venv/bin/python scripts/discretionary.py brief                     # replay exits, mark, print the state
    .venv/bin/python scripts/discretionary.py execute decision.json --dry-run
    .venv/bin/python scripts/discretionary.py execute decision.json     # commits the day's decision
    .venv/bin/python scripts/discretionary.py score
    .venv/bin/python scripts/discretionary.py kill --reason "..."       # close everything, block new calls
    .venv/bin/python scripts/discretionary.py reset-kill --operator NAME  # the account owner's decision only

The decision format and the limits are in src/quant/discretionary/calls.py; the daily procedure is in
.claude/skills/trading-session/SKILL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.store import ParquetStore  # noqa: E402
from quant.discretionary.book import BookConfig, BookError, DiscretionaryBook  # noqa: E402
from quant.discretionary.brief import build_brief, render_brief  # noqa: E402
from quant.discretionary.calls import DecisionError, parse_decision  # noqa: E402
from quant.discretionary.scoring import call_frame, render_scorecard, scorecard  # noqa: E402
from quant.live.feed import BinancePublicFeed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "var" / "data"))
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief")
    brief.add_argument("--no-positioning", action="store_true", help="skip the retail long/short ratios (faster)")
    execute = sub.add_parser("execute")
    execute.add_argument("decision")
    execute.add_argument("--dry-run", action="store_true", help="validate and show the risk engine's verdict only")
    sub.add_parser("score")
    kill = sub.add_parser("kill")
    kill.add_argument("--reason", required=True)
    reset = sub.add_parser("reset-kill")
    reset.add_argument("--operator", required=True)
    args = parser.parse_args()

    config = BookConfig()
    feed = BinancePublicFeed(ParquetStore(Path(args.data_dir)), ROOT / "var" / "live_cache")
    book = DiscretionaryBook(config, feed, ROOT / "var" / "discretionary" / config.book_id)
    now = pd.Timestamp.now(tz="UTC")

    try:
        if args.command == "brief":
            text = render_brief(build_brief(book, now, book.refresh(now), positioning=not args.no_positioning))
            archive = book.dir / "briefs" / f"{now:%Y-%m-%dT%H%M}.txt"
            archive.parent.mkdir(exist_ok=True)
            archive.write_text(text)
            print(text)
            print(f"\nbrief archivé : {archive.relative_to(ROOT)}")
        elif args.command == "execute":
            result = book.execute(parse_decision(args.decision, config.limits), now, dry_run=args.dry_run)
            print(render_execution(result))
        elif args.command == "score":
            book.refresh(now)
            print(render_scorecard(scorecard(book.state.calls.values(), book.equity_points()),
                                   call_frame(book.state.calls.values())))
        elif args.command == "kill":
            closed = book.kill(now, f"manual: {args.reason}")
            print(f"kill switch armé, {len(closed)} paris soldés ; équité {book.mark(now)['equity']:,.2f} USDT")
        elif args.command == "reset-kill":
            book.reset_kill(args.operator, now)
            print(f"kill switch réarmé par {args.operator} ; le drawdown repart de l'équité actuelle")
    except DecisionError as exc:
        print(exc, file=sys.stderr)
        return 2
    except BookError as exc:
        print(f"LIVRE ARRÊTÉ : {exc}", file=sys.stderr)
        return 3
    return 0


def render_execution(result: dict) -> str:
    if result["status"] == "killed":
        return f"KILL SWITCH ACTIF ({result['reason']}) : rien n'a été exécuté, {len(result['exits'])} paris soldés"
    lines = []
    if result["status"] == "dry_run":
        lines.append(f"ESSAI À BLANC, rien n'est engagé (équité {result['equity']:,.2f} USDT)")
        for call_id in result["would_close"]:
            lines.append(f"  clôture prévue : {call_id}")
        for row in result["plan"]:
            lines.append(f"  {row['symbol']} {row['direction']} : demandé {row['requested_weight']:.2%}, accordé "
                         f"{row['approved_weight']:.2%} ({row['action']})"
                         + (f" : {'; '.join(row['reasons'])}" if row["reasons"] else ""))
        return "\n".join(lines or ["décision valide, aucun ordre"])
    lines.append(f"DÉCISION DU {result['decision_date']} EXÉCUTÉE ET FIGÉE")
    for c in result["closed"]:
        lines.append(f"  clos  {c['call_id']} {c['direction']} à {c['exit_price']:.6g} : {c['net_return']:+.2%} du "
                     f"notionnel ({c['r_multiple']:+.2f} R), {c['pnl']:+.2f} USDT")
    for c in result["opened"]:
        sign = 1 if c["direction"] == "long" else -1
        target = f"{c['entry_price'] * (1 + sign * c['target_pct']):.6g}" if c["target_pct"] else "aucun"
        lines.append(f"  ouvert {c['call_id']} {c['direction']} {c['weight']:.2%} à {c['entry_price']:.6g} "
                     f"(coût {c['entry_cost_bps']:.1f} bps) | stop {c['entry_price'] * (1 - sign * c['stop_pct']):.6g}, "
                     f"objectif {target}, échéance {c['expiry'][:16]}"
                     + (f" | réduit par le risque : {'; '.join(c['risk_reasons'])}" if c["risk_reasons"] else ""))
    for b in result["blocked"]:
        lines.append(f"  BLOQUÉ {b['symbol']} {b['direction']} : {'; '.join(b['reasons']) or 'poids accordé trop faible'}")
    lines.append(f"équité {result['equity']:,.2f} USDT | brut {result['gross']:.1%} | net {result['net']:+.1%} | "
                 f"{result['open_calls']} paris ouverts")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
