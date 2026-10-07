"""Is the trading loop actually alive and sane? Exits non-zero when it is not.

    .venv/bin/python scripts/healthcheck.py                     # all known journals
    .venv/bin/python scripts/healthcheck.py --journal var/paper/xxx --max-age-bars 2

Designed for cron or a systemd timer: the dangerous failure is not a crash (systemd restarts it) but
a loop that keeps running while silently skipping every bar.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from quant.data.timeframes import interval_to_timedelta  # noqa: E402


def check(journal_dir: Path, max_age_bars: float, interval: str) -> list[str]:
    problems: list[str] = []
    events_file = journal_dir / "events.jsonl"
    if not events_file.exists():
        return [f"{journal_dir.name}: no journal yet"]
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    steps = [e for e in events if e.get("event") == "step_summary"]
    if not steps:
        return [f"{journal_dir.name}: journal has no completed step"]
    last = steps[-1]
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(last["ts"])
    bar = interval_to_timedelta(interval)
    if age > max_age_bars * bar:
        problems.append(f"{journal_dir.name}: last step {age} ago (over {max_age_bars} bars)")
    if last.get("kill_switch"):
        problems.append(f"{journal_dir.name}: KILL SWITCH active")
    recent = [e for e in events[-200:] if e.get("event") == "alert"]
    skipped = [e for e in recent if "late step" in str(e.get("message", ""))]
    if len(skipped) >= 3:
        problems.append(f"{journal_dir.name}: {len(skipped)} late steps skipped recently")
    errors = [e for e in events[-200:] if e.get("event") == "error"]
    if errors:
        problems.append(f"{journal_dir.name}: {len(errors)} errors, last: {str(errors[-1].get('error'))[:120]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--journal", action="append", default=None, help="journal directory (repeatable)")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--max-age-bars", type=float, default=1.5)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()

    journals = [Path(j) for j in args.journal] if args.journal else sorted(
        d for d in (ROOT / "var" / "paper").glob("*/") if not d.name.startswith("_"))   # "_archive" is not a loop
    problems: list[str] = []
    for journal in journals:
        problems += check(journal, args.max_age_bars, args.interval)

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < args.min_free_gb:
        problems.append(f"disk: only {free_gb:.1f} GB free")

    if problems:
        print("UNHEALTHY")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"healthy: {len(journals)} loop(s), {free_gb:.0f} GB free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
