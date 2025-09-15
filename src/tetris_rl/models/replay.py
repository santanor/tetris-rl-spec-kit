from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class ReplayArtifact:
    episode_index: int
    frame_indices: List[int]
    highlight_reason: str | None = None
