"""Replay storage helpers stub.

The test ensures compute_keyframe_indices is not yet present.

Future implementation outline:
- compute_keyframe_indices(total_frames: int, stride: int) -> list[int]
- save_replay(artifact: ReplayArtifact, path: Path)
- load_replay(path: Path) -> ReplayArtifact
"""
from __future__ import annotations

from typing import List


def compute_keyframe_indices(total_frames: int, stride: int) -> List[int]:
    """Return keyframe indices selecting every stride-th frame including first and last.

    Ensures 0 and total_frames-1 are present (if total_frames>0) and stride>=1.
    """
    if total_frames <= 0:
        return []
    if stride < 1:
        raise ValueError("stride must be >=1")
    indices = list(range(0, total_frames, stride))
    last = total_frames - 1
    if indices[-1] != last:
        indices.append(last)
    return indices
