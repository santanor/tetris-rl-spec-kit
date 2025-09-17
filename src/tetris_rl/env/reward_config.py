from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RewardConfig:
    """Minimal reward configuration with mild structural shaping.

    Components
    -----------
    line_reward_[1-4]:  Monotonic rewards for clearing 1..4 lines.
    survival_reward:    Small per non-terminal step reward (longevity).
    top_out_penalty:    Large negative on episode termination (discourage topping out).
    placement shaping:  On every piece lock we compare hole count (before vs after).
        * If holes_after > holes_before  -> placement_hole_penalty (negative)
        * Else (maintain or reduce)      -> placement_no_hole_reward (positive)

    Design intent: Placement shaping accelerates learning of "avoid creating new holes"
    without encoding full handcrafted heuristics (aggregate height, bumpiness). Its
    magnitudes are intentionally smaller than typical multi-line clear bonuses so that
    clearing lines remains the dominant objective.
    """

    # Line clear rewards (simple & monotonic)
    line_reward_1: float = 1.0
    line_reward_2: float = 3.0
    line_reward_3: float = 5.0
    line_reward_4: float = 8.0

    # Survival incentive
    survival_reward: float = 0.02

    # Terminal penalty
    top_out_penalty: float = -10.0

    # Placement shaping
    placement_no_hole_reward: float = 0.4
    placement_hole_penalty: float = -0.6

    def line_table(self) -> Dict[int, float]:
        return {1: self.line_reward_1, 2: self.line_reward_2, 3: self.line_reward_3, 4: self.line_reward_4}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_partial(cls, data: Dict[str, Any]) -> 'RewardConfig':
        base = cls()
        for k, v in data.items():
            if hasattr(base, k):
                try:
                    setattr(base, k, float(v))
                except (TypeError, ValueError):
                    pass
        return base
