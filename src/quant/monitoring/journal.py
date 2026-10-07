"""Structured event journal and alerts for paper/live loops."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EventJournal:
    path: Path
    echo: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        if self.echo and event in {"alert", "kill_switch", "reconciliation_mismatch", "error", "step_summary"}:
            print(json.dumps(record, default=str), file=sys.stderr if event != "step_summary" else sys.stdout)

    def alert(self, message: str, **fields) -> None:
        self.log("alert", message=message, **fields)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
