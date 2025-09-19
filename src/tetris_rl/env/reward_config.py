from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class RewardConfig:
    """Reward configuration matching the simplified rules.

    Components
    -----------
    line_reward_[1-4]:      Big rewards for clearing 1..4 lines.
    survival_reward:        Small per non-terminal step reward (staying alive).
    delta_stable_reward:    Small reward when a lock does not increase (max_height - min_height).
    hole_penalty_per:       Penalty per newly created hole on lock.
    top_out_penalty:        Large negative on episode termination.
    """

    # Line clear rewards (bigger than shaping terms)
    line_reward_1: float = 10.0
    line_reward_2: float = 30.0
    line_reward_3: float = 60.0
    line_reward_4: float = 120.0

    # Survival incentive (per step while not terminal)
    survival_reward: float = 0.02

    # Reward when a lock does NOT increase skyline delta (max - min)
    delta_stable_reward: float = 0.2

    # Penalty per newly created hole on lock
    hole_penalty_per: float = -0.2

    # Terminal penalty
    top_out_penalty: float = -8.0

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
