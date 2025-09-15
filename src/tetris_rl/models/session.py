from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from .episode import Episode

@dataclass
class Session:
    """Aggregate lifetime of a training/evaluation run."""
    session_id: str
    mode: str  # 'train' | 'eval'
    seed: int
    config: Dict[str, Any]
    episodes_total: int = 0
    total_reward: float = 0.0
    total_lines_cleared: int = 0
    warnings: List[str] = field(default_factory=list)
    started_ts: float = field(default_factory=lambda: time.time())
    finished_ts: Optional[float] = None
    episodes: List[Episode] = field(default_factory=list)

    def record_episode(self, reward: float, lines: int, episode: Episode | None = None) -> None:
        self.episodes_total += 1
        self.total_reward += reward
        self.total_lines_cleared += lines
        if episode is not None:
            self.episodes.append(episode)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.episodes_total if self.episodes_total else 0.0

    @property
    def avg_lines_cleared(self) -> float:
        return self.total_lines_cleared / self.episodes_total if self.episodes_total else 0.0

    def finish(self) -> None:
        if self.finished_ts is None:
            self.finished_ts = time.time()

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "episodes_total": self.episodes_total,
            "avg_reward": self.avg_reward,
            "avg_lines_cleared": self.avg_lines_cleared,
            "warnings": self.warnings,
        }
