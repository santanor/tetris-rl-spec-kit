from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

"""Minimal reward configuration.

We intentionally strip out prior shaping terms (holes, bumpiness, density, survival)
and keep only:
  - line clear rewards (configurable non-linear scaling)
  - optional per-step penalty (can be set to 0.0 to disable)
  - a column imbalance penalty discouraging one column towering above the others
  - top-out penalty

Column imbalance penalty:
Given column heights h_i and average of other columns avg_{-i}, we compute
excess_i = max(0, h_i - avg_{-i}). We either take the sum or max (mode configurable)
and multiply by imbalance_penalty_scale (and optionally by imbalance_penalty_power > 1
to accentuate large disparities).
"""

@dataclass
class RewardConfig:
    # Line clear rewards
    line_reward_1: float = 1.0
    line_reward_2: float = 3.0
    line_reward_3: float = 5.0
    line_reward_4: float = 8.0

    # Optional small per-step penalty to avoid aimless stalling (0.0 = disabled)
    step_penalty: float = 0.0

    # Column imbalance settings
    imbalance_penalty_scale: float = 0.05
    imbalance_penalty_power: float = 1.0   # >1 exaggerates large excesses
    imbalance_mode: str = "sum"            # 'sum' or 'max'

    # Per-column hole presence penalty: for each column that contains at least one
    # empty cell below a filled cell (i.e. has a 'hole'), apply this penalty.
    # Set to 0.0 to disable.
    hole_column_penalty: float = 0.0

    # Top-out penalty (episode termination)
    top_out_penalty: float = -10.0

    def line_table(self) -> Dict[int, float]:
        return {1: self.line_reward_1, 2: self.line_reward_2, 3: self.line_reward_3, 4: self.line_reward_4}

    def to_dict(self) -> Dict[str, Any]:  # Backwards-compatible for existing API consumers
        return asdict(self)

    @classmethod
    def from_partial(cls, data: Dict[str, Any]) -> 'RewardConfig':
        base = cls()
        for k, v in data.items():
            if hasattr(base, k):
                # do not coerce mode strings or other non-floats blindly
                current = getattr(base, k)
                if isinstance(current, float):
                    try:
                        setattr(base, k, float(v))
                    except Exception:
                        pass
                else:
                    setattr(base, k, v)
        return base
