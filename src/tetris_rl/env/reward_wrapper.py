"""Reward shaping wrapper stubs.

Intentionally minimal at this stage of TDD: tests previously expected missing
RewardShapingWrapper attribute; after adding this skeleton we will adjust tests
(or add new ones) to drive behavior.

Planned implementation will include:
- class RewardShapingWrapper(gym.Wrapper) with custom reward calculation
- weight composition logic and reward components (line clears, holes penalty, height, step penalty)
"""
from __future__ import annotations

from typing import Dict, Any

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore

BaseWrapper = gym.Wrapper if gym else object  # type: ignore[misc]

class RewardShapingWrapper(BaseWrapper):  # type: ignore[misc]
    """Skeleton reward shaping wrapper.

    Currently passes through environment unchanged; only stores provided weights.
    Accepts any object with step/reset to ease testing before full gym integration.
    """
    def __init__(self, env: Any, *, line_clear_weight: float = 1.0, hole_weight: float = -0.5,
                 height_weight: float = -0.1, step_penalty: float = -0.01) -> None:  # noqa: D401
        if gym and isinstance(env, gym.Env):  # normal path
            super().__init__(env)  # type: ignore[arg-type]
            self.env = env  # explicit for mypy when gym present
        else:  # fallback simple assignment (no gym inheritance semantics)
            self.env = env  # type: ignore
        self.weights: Dict[str, float] = {
            "line_clear": line_clear_weight,
            "hole": hole_weight,
            "height": height_weight,
            "step": step_penalty,
        }

    def step(self, action):  # type: ignore[override]
        return self.env.step(action)

    def reset(self, *args, **kwargs):  # type: ignore[override]
        return self.env.reset(*args, **kwargs)
