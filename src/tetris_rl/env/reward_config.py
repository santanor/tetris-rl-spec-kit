from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class RewardConfig:
    line_reward_1: float = 1.5
    line_reward_2: float = 4.0
    line_reward_3: float = 8.0
    line_reward_4: float = 14.0
    step_penalty: float = -0.005
    survival_bonus: float = 0.0015
    holes_weight: float = -0.25       # stronger discouragement of new holes
    holes_abs_weight: float = -0.02   # penalty per existing hole each step (small persistent pressure)
    weighted_holes_weight: float = -0.0  # delta-based penalty using depth-weighted holes
    weighted_holes_abs_weight: float = -0.0  # absolute penalty using depth-weighted holes count
    holes_depth_power: float = 1.0     # exponent for depth weighting (>=1 amplifies deeper holes)
    height_weight: float = -0.003     # gentler height pressure to allow stacking for combos
    bumpiness_weight: float = -0.008  # slightly softer
    top_out_penalty: float = -2.0
    height_threshold: int = 100       # aggregate height threshold before applying height penalty (approx 10 columns * mean height 10)
    conditional_height: bool = True   # if True only penalize height when aggregate exceeds threshold
    # Row fill density shaping: encourage leaving rows more filled (fewer empties / holes across horizontal lines)
    # density is defined per occupied row: filled_cells / width. We compute average density across non-empty rows.
    row_density_delta_weight: float = 0.5   # positive reward when average density increases (fewer gaps)
    row_density_abs_weight: float = 0.0     # optional absolute shaping: reward *current* density each step
    row_density_line_clear_scale: float = 2.0  # multiplier applied to density delta reward on steps with line clears

    def line_table(self) -> Dict[int, float]:
        return {1: self.line_reward_1, 2: self.line_reward_2, 3: self.line_reward_3, 4: self.line_reward_4}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_partial(cls, data: Dict[str, Any]) -> 'RewardConfig':
        base = cls()
        for k,v in data.items():
            if hasattr(base, k):
                setattr(base, k, float(v))
        return base
