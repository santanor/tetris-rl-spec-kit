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
        # feature vector length 3 from board.feature_vector + step count
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(4,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # 5 actions defined in board step mapping (# type: ignore[arg-type])
        self._step_count = 0

    def _build_obs(self):
        feats = self.board.feature_vector()
        return np.array([float(self._step_count)] + feats, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        if seed is not None:
            # re-seed board RNG
            self.board = TetrisBoard(seed=seed)
        else:
            self.board = TetrisBoard(seed=0)
        self._step_count = 0
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        # Minimal reward computation: line clears + column imbalance penalty.
        self._step_count += 1
        lines_delta, top_out, locked = self.board.step(int(action))

        cfg = self.reward_config
        base = cfg.line_table().get(lines_delta, 0.0) if lines_delta else 0.0
        step_penalty = cfg.step_penalty

        heights = self.board.heights()  # list length = board width (10)
        imbalance_excesses = []
        if len(heights) > 1:
            for i, h in enumerate(heights):
                others = heights[:i] + heights[i+1:]
                avg_others = sum(others) / len(others) if others else 0.0
                excess = max(0.0, h - avg_others)
                imbalance_excesses.append(excess)
        else:
            imbalance_excesses = [0.0]

        if cfg.imbalance_mode == 'max':
            imbalance_metric = max(imbalance_excesses)
        else:  # sum mode by default
            imbalance_metric = sum(imbalance_excesses)

        # Apply power then scale
        if cfg.imbalance_penalty_power != 1.0:
            imbalance_penalty_value = (imbalance_metric ** cfg.imbalance_penalty_power) * cfg.imbalance_penalty_scale
        else:
            imbalance_penalty_value = imbalance_metric * cfg.imbalance_penalty_scale

        reward = base + step_penalty - imbalance_penalty_value
        if top_out:
            reward += cfg.top_out_penalty

        terminated = top_out
        truncated = self._step_count >= self.max_steps
        obs = self._build_obs()
        info: Dict[str, Any] = {
            "lines_delta": lines_delta,
            "locked": locked,
            "lines_cleared_total": self.board.lines_cleared_total,
            "heights": heights,
            "imbalance_excesses": imbalance_excesses,
            "imbalance_metric": imbalance_metric,
            "reward_components": {
                "base_line": base,
                "step_penalty": step_penalty,
                "imbalance_penalty": -imbalance_penalty_value,
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
