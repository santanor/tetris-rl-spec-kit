from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Episode:
    index: int
    steps: int = 0
    total_reward: float = 0.0
    lines_cleared: int = 0
    holes_traj: List[int] = field(default_factory=list)
    height_traj: List[int] = field(default_factory=list)
    reward_traj: List[float] = field(default_factory=list)
    notable_flags: List[str] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    interrupted: bool = False
    termination_reason: Optional[str] = None

    def record_step(self, reward: float, lines_delta: int, holes: int, height: int) -> None:
        self.steps += 1
        self.total_reward += reward
        if lines_delta:
            self.lines_cleared += lines_delta
        self.holes_traj.append(holes)
        self.height_traj.append(height)
        self.reward_traj.append(reward)

    def add_flag(self, flag: str) -> None:
        if flag not in self.notable_flags:
            self.notable_flags.append(flag)

    def finalize(self, terminated: bool, truncated: bool, interrupted: bool, reason: Optional[str]) -> None:
        self.terminated = terminated
        self.truncated = truncated
        self.interrupted = interrupted
        self.termination_reason = reason

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "total_reward": self.total_reward,
            "lines_cleared": self.lines_cleared,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "interrupted": self.interrupted,
            "max_height": max(self.height_traj) if self.height_traj else 0,
            "holes_final": self.holes_traj[-1] if self.holes_traj else 0,
            "notable_flags": self.notable_flags,
        }
