from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class RewardConfig:
    """Minimal reward configuration.

    Philosophy (reset / simplified):
      - Reward ONLY for clearing lines. (Higher for more simultaneous clears.)
      - Small positive reward each step survived to encourage longevity.
      - Large negative penalty when topping out (episode terminates).
    No structural shaping (holes / height / bumpiness) is applied; the agent must
    implicitly learn structural strategies purely from delayed line + survival signals.
    """

    # Line clear rewards (can be tuned but intentionally simple & monotonic)
    line_reward_1: float = 1.0
    line_reward_2: float = 3.0
    line_reward_3: float = 5.0
    line_reward_4: float = 8.0

    # Per-step survival reward (applied on every non-terminal step)
    survival_reward: float = 0.02

    # Penalty applied once when the board tops out
    top_out_penalty: float = -10.0

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
