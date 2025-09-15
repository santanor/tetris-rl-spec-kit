"""Stagnation / early termination wrapper stub.

Deliberately minimal: provides a tracking structure but no termination yet.

Planned implementation will:
- Track recent reward / line clear activity
- Trigger done when stagnation threshold reached
"""
from __future__ import annotations

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None  # type: ignore

BaseWrapper = gym.Wrapper if gym else object  # type: ignore[misc]

class StagnationTerminationWrapper(BaseWrapper):  # type: ignore[misc]
    """Tracks steps since last 'progress' signal (placeholder).

    For now 'progress' is any positive reward in the raw environment tuple.
    Does not yet terminate; just stores internal counters for future tests.
    """
    def __init__(self, env, stagnation_limit: int = 200):
        if gym and isinstance(env, gym.Env):  # normal path
            super().__init__(env)  # type: ignore[arg-type]
            self.env = env  # explicit for mypy
        else:  # fallback simple assignment
            self.env = env  # type: ignore
        self.stagnation_limit = stagnation_limit
        self.steps_since_progress = 0
        self.total_steps = 0

    def reset(self, *args, **kwargs):  # type: ignore[override]
        self.steps_since_progress = 0
        self.total_steps = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        positive = False
        try:  # attempt numeric positivity check
            positive = float(reward) > 0.0
        except Exception:  # pragma: no cover
            positive = bool(reward)
        if positive:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        return obs, reward, terminated, truncated, info
