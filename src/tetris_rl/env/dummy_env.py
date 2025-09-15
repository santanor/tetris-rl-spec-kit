"""A minimal placeholder Tetris-like environment for manual demo runs.

This is NOT a full Tetris implementation—just enough to exercise the session
recording and model scaffolding. The observation is a simple feature vector:
[state_step, lines_cleared_total, holes_estimate, aggregate_height_estimate].
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

try:  # runtime import
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    gym = object  # type: ignore
    class spaces:  # type: ignore
        class Box:  # minimal shim
            def __init__(self, *_, **__):
                pass
        class Discrete:
            def __init__(self, *_, **__):
                pass

class DummyTetrisEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps: int = 200):
        super().__init__()  # type: ignore
        self.max_steps = max_steps
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(4,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # type: ignore[arg-type]
        self._step = 0
        self._lines_cleared = 0
        self._holes_estimate = 0
        self._height_estimate = 0

    def _get_obs(self):
        return np.array([
            float(self._step),
            float(self._lines_cleared),
            float(self._holes_estimate),
            float(self._height_estimate),
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        try:
            super().reset(seed=seed)  # type: ignore
        except Exception:  # pragma: no cover
            pass
        self._step = 0
        self._lines_cleared = 0
        self._holes_estimate = 0
        self._height_estimate = 4
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        self._step += 1
        if self._step % 10 == 0:
            self._lines_cleared += 1
            self._holes_estimate = max(0, self._holes_estimate - 1)
            reward = 1.0
        else:
            if self._step % 7 == 0:
                self._holes_estimate += 1
            if self._step % 5 == 0:
                self._height_estimate += 1
            if self._step % 12 == 0:
                self._height_estimate = max(0, self._height_estimate - 1)
            reward = -0.01
        terminated = False
        truncated = self._step >= self.max_steps
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        print(f"Step={self._step} lines={self._lines_cleared} holes={self._holes_estimate} height={self._height_estimate}")

    def close(self):  # pragma: no cover
        pass
