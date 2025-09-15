from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
import uuid

DEFAULTS = {
    "observation_mode": "features",
    "max_episodes": 300,
    "line_clear_weight": 1.0,
    "hole_weight": 0.35,
    "height_weight": 0.5,
    "step_penalty": 0.01,
    "stagnation_no_line_steps": 500,
    "plateau_window": 50,
    "plateau_delta": 0.01,
    "eval_episodes": 10,
    "ui_update_interval_steps": 5,
    "replay_keyframe_stride": 4,
}


def generate_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Configuration:
    observation_mode: str = DEFAULTS["observation_mode"]
    max_episodes: int = DEFAULTS["max_episodes"]
    line_clear_weight: float = DEFAULTS["line_clear_weight"]
    hole_weight: float = DEFAULTS["hole_weight"]
    height_weight: float = DEFAULTS["height_weight"]
    step_penalty: float = DEFAULTS["step_penalty"]
    stagnation_no_line_steps: int = DEFAULTS["stagnation_no_line_steps"]
    plateau_window: int = DEFAULTS["plateau_window"]
    plateau_delta: float = DEFAULTS["plateau_delta"]
    eval_episodes: int = DEFAULTS["eval_episodes"]
    ui_update_interval_steps: int = DEFAULTS["ui_update_interval_steps"]
    replay_keyframe_stride: int = DEFAULTS["replay_keyframe_stride"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
