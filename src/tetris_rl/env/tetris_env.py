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
        # Observation (Option D structural set):
        #   10 normalized column heights
        #   + holes_norm (total holes / 200)
        #   + bumpiness_norm (sum |h_i-h_{i+1}| / 180)
        #   + aggregate_height_norm (sum heights / 200)
        #   + lines_fraction (lines_cleared_total / 200)
        # => 14 features total
        self._obs_dim = 14
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # type: ignore[arg-type]
        self._step_count = 0

    def _build_obs(self):
        heights = self.board.heights()  # list of 10 ints (0..20)
        norm_heights = [h / 20.0 for h in heights]
        holes = self.board.holes()
        bump = self.board.bumpiness()
        agg = sum(heights)
        # Normalizations
        holes_norm = holes / 200.0  # max theoretical holes = 200 (empty below filled top rows)
        bump_norm = bump / 180.0    # worst-case 9 gaps * 20 height diff each = 180
        agg_norm = agg / 200.0      # max aggregate height 10 * 20
        lines_fraction = self.board.lines_cleared_total / 200.0
        feats = norm_heights + [holes_norm, bump_norm, agg_norm, lines_fraction]
        return np.array(feats, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        self.board = TetrisBoard(seed=seed if seed is not None else 0)
        self._step_count = 0
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        self._step_count += 1
        cfg = self.reward_config

        # Track holes before action IF a lock might happen (we just inspect before calling board.step)
        holes_before = self.board.holes()
        heights_before = self.board.heights()
        max_before = max(heights_before) if heights_before else 0
        sorted_before = sorted(heights_before, reverse=True)
        second_before = sorted_before[1] if len(sorted_before) > 1 else max_before

        lines_delta, top_out, locked = self.board.step(int(action))

        # Core components
        line_reward = cfg.line_table().get(lines_delta, 0.0) if lines_delta else 0.0
        survival = cfg.survival_reward if not top_out else 0.0

        # Placement structural shaping (evaluated only when a piece locks and not due to top-out early spawn collision)
        placement_reward = 0.0
        lock_flat = 0.0
        skyline_reward = 0.0
        if locked:
            holes_after = self.board.holes()
            if holes_after > holes_before:  # created at least one new hole
                placement_reward = cfg.placement_hole_penalty
            else:  # maintained or reduced hole count
                placement_reward = cfg.placement_no_hole_reward
            lock_flat = cfg.lock_reward
            # Skyline shaping: compare tallest column behavior
            heights_after = self.board.heights()
            if heights_after:
                max_after = max(heights_after)
                sorted_after = sorted(heights_after, reverse=True)
                second_after = sorted_after[1] if len(sorted_after) > 1 else max_after
                # If we raised the skyline and the spread is above threshold -> penalty
                if max_after > max_before and (max_after - second_after)/20.0 > cfg.skyline_spread_threshold:
                    skyline_reward = cfg.skyline_raise_penalty
                else:
                    # Reward any lock that doesn't exacerbate a spike (including filling gaps / flattening)
                    skyline_reward = cfg.skyline_flat_reward
        # NOTE: If top_out, we still apply placement shaping for the final lock; can reconsider if noisy.

        reward = line_reward + survival + placement_reward + lock_flat + skyline_reward
        if top_out:
            reward += cfg.top_out_penalty

        terminated = bool(top_out)
        truncated = self._step_count >= self.max_steps
        obs = self._build_obs()
        info: Dict[str, Any] = {
            "lines_delta": lines_delta,
            "locked": locked,
            "lines_cleared_total": self.board.lines_cleared_total,
            "state_features": {
                "holes": self.board.holes(),
                "bumpiness": self.board.bumpiness(),
                "aggregate_height": sum(self.board.heights()),
            },
            "reward_components": {
                "line_reward": line_reward,
                "survival": survival,
                "placement": placement_reward,
                "lock": lock_flat,
                "skyline": skyline_reward,
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
