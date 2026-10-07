"""Append-only JSONL trace of every harness decision, for replay and eval."""
from __future__ import annotations

import json
import time
from pathlib import Path


class Trace:
    def __init__(self, directory: str = "runs", name: str | None = None):
        Path(directory).mkdir(exist_ok=True)
        self.path = Path(directory) / f"{name or time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.records: list[dict] = []

    def log(self, kind: str, **data) -> dict:
        rec = {"ts": round(time.time(), 3), "kind": kind, **data}
        self.records.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec
