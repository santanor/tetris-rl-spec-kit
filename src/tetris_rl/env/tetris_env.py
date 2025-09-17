"""Gymnasium environment wrapper for the real TetrisBoard core."""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

try:
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

from tetris_rl.core.board import TetrisBoard
from tetris_rl.env.reward_config import RewardConfig

class TetrisEnv(gym.Env):  # type: ignore[misc]
    metadata = {"render.modes": ["human"]}

    def __init__(self, seed: int = 0, max_steps: int = 500, reward_config: RewardConfig | None = None):
        super().__init__()  # type: ignore
        self.board = TetrisBoard(seed=seed)
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        # Simplified observation: 10 normalized column heights + lines_cleared_fraction + step_fraction.
        # Dimension = 12.
        self._obs_dim = 12
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # type: ignore[arg-type]
        self._step_count = 0

    def _build_obs(self):
        heights = self.board.heights()  # list of 10 ints (0..20)
        max_h = 20.0
        norm_heights = [h / max_h for h in heights]
        lines_fraction = self.board.lines_cleared_total / 200.0  # rough cap
        step_fraction = float(self._step_count) / float(max(1, self.max_steps))
        feats = norm_heights + [lines_fraction, step_fraction]
        return np.array(feats, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        self.board = TetrisBoard(seed=seed if seed is not None else 0)
        self._step_count = 0
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        self._step_count += 1
        lines_delta, top_out, locked = self.board.step(int(action))
        cfg = self.reward_config
        # Line reward
        line_reward = cfg.line_table().get(lines_delta, 0.0) if lines_delta else 0.0
        # Survival reward (only if not top-out)
        survival = cfg.survival_reward if not top_out else 0.0
        reward = line_reward + survival
        if top_out:
            reward += cfg.top_out_penalty
        terminated = bool(top_out)
        truncated = self._step_count >= self.max_steps
        obs = self._build_obs()
        info: Dict[str, Any] = {
            "lines_delta": lines_delta,
            "locked": locked,
            "lines_cleared_total": self.board.lines_cleared_total,
            "reward_components": {
                "line_reward": line_reward,
                "survival": survival,
                "top_out": cfg.top_out_penalty if top_out else 0.0,
                "total": reward,
                "config_hash": hash(tuple(sorted(cfg.to_dict().items()))),
            },
        }
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        grid_chars = []
        for row in self.board.grid:
            grid_chars.append(''.join('#' if c else '.' for c in row))
        print('\n'.join(grid_chars[-10:]))

    def close(self):  # pragma: no cover
        pass
