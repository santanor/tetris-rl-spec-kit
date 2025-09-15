"""Simple export utilities for manual demo run."""
from __future__ import annotations

from pathlib import Path
import json
import csv
from typing import Iterable

from tetris_rl.models.session import Session

def export_session_metrics(session: Session, path: Path) -> None:
    data = session.summary_dict()
    path.write_text(json.dumps(data, indent=2))


def export_session_episodes(session: Session, path: Path) -> None:
    fieldnames = [
        "index","total_reward","lines_cleared","steps","terminated","truncated","interrupted","max_height","holes_final","notable_flags"
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in session.episodes:
            writer.writerow(ep.summary_dict())
