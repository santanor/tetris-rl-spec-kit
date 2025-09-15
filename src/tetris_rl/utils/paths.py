"""Path utilities for run artifact organization."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

RUNS_ENV_VAR = "TETRIS_RL_RUNS_DIR"
DEFAULT_RUNS_DIR = "runs"


def get_runs_root() -> Path:
    base = os.getenv(RUNS_ENV_VAR, DEFAULT_RUNS_DIR)
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def allocate_session_dir(session_id: str | None = None) -> Path:
    sid = session_id or uuid.uuid4().hex[:8]
    root = get_runs_root()
    d = root / sid
    d.mkdir(parents=True, exist_ok=False)
    return d
