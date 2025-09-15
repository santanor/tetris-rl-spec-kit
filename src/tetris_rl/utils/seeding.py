"""Seeding utility to enforce reproducibility best-effort."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional at import time for some tests
    torch = None  # type: ignore


_DEF_ENVAR = "TETRIS_RL_GLOBAL_SEED"


def seed_everything(seed: int | None = None) -> int:
    """Seed python, numpy, torch (if available) and set env flag.

    Returns the resolved integer seed.
    """
    if seed is None:
        env_seed = os.getenv(_DEF_ENVAR)
        if env_seed is not None:
            seed = int(env_seed)
        else:
            seed = 42
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:  # pragma: no branch
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - may not run in CI
            torch.cuda.manual_seed_all(seed)
        # Determinism toggles (may reduce performance)
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass
    os.environ[_DEF_ENVAR] = str(seed)
    return seed
