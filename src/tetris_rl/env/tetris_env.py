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
        # Build one observation to infer dynamic feature dimension
        sample_obs = self._build_obs()
        self._obs_dim = int(len(sample_obs))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # type: ignore[arg-type]
        self._step_count = 0

    def _build_obs(self):
        # 1) Current per-column heights (normalized)
        heights = self.board.heights()  # len 10, ints 0..20
        norm_heights = [min(max(h / 20.0, 0.0), 1.0) for h in heights]

        # 2) Predicted per-column height if current piece hard-dropped at that column (left alignment)
        # 6) Predicted holes created if dropped there
        holes_before_total = self.board.holes()
        predicted_heights = [0.0] * 10
        predicted_created_holes = [0.0] * 10
        # If no active piece, keep zeros (top-out case)
        if self.board.active is not None:
            # Precompute first-filled row per column to speed up repeated sims
            heights_int = self.board.heights()
            y_first = [20 - h if h > 0 else 20 for h in heights_int]
            for c in range(10):
                sim = self.board.simulate_drop_stats_at_left(c, y_first)
                if sim.get("valid"):
                    h_after_c = sim.get("h_after_c") or 0
                    predicted_heights[c] = min(max(h_after_c / 20.0, 0.0), 1.0)
                    created = max(int(sim.get("created_new") or 0), 0)
                    predicted_created_holes[c] = min(max(created / 20.0, 0.0), 1.0)
                else:
                    # Invalid alignment: discourage via max created holes; keep height as current
                    predicted_heights[c] = norm_heights[c]
                    predicted_created_holes[c] = 1.0

        # 3) Current piece and rotation (one-hot 7 + one-hot 4)
        piece_one_hot = [0.0] * 7
        rot_one_hot = [0.0] * 4
        x_norm = 0.0
        y_norm = 0.0
        if self.board.active is not None:
            kind = self.board.active.kind
            kind_order = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
            if kind in kind_order:
                piece_one_hot[kind_order.index(kind)] = 1.0
            r = int(self.board.active.rotation % 4)
            rot_one_hot[r] = 1.0
            x_norm = min(max(self.board.active.x / 9.0, 0.0), 1.0)
            # active.y may be <0 during spawn; clamp to [0,19]
            y_clamped = min(max(self.board.active.y, 0), 19)
            y_norm = y_clamped / 19.0 if 19 > 0 else 0.0

        # 4) Delta between tallest and shortest columns (normalized)
        if heights:
            skyline_delta = (max(heights) - min(heights)) / 20.0
        else:
            skyline_delta = 0.0

        # 5) Per-column holes (normalized)
        holes_per_col = self.board.per_column_holes()
        norm_holes_per_col = [min(max(hc / 20.0, 0.0), 1.0) for hc in holes_per_col]

        feats = (
            norm_heights
            + predicted_heights
            + piece_one_hot
            + rot_one_hot
            + [skyline_delta]
            + norm_holes_per_col
            + predicted_created_holes
            + [x_norm, y_norm]
        )
        return np.array(feats, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        self.board = TetrisBoard(seed=seed if seed is not None else 0)
        self._step_count = 0
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        self._step_count += 1
        cfg = self.reward_config

        # Track state before action for shaping on locks
        holes_before = self.board.holes()
        heights_before = self.board.heights()
        delta_before = (max(heights_before) - min(heights_before)) if heights_before else 0

        lines_delta, top_out, locked = self.board.step(int(action))

        # Core components
        line_reward = cfg.line_table().get(lines_delta, 0.0) if lines_delta else 0.0
        survival = cfg.survival_reward if not top_out else 0.0

        # Shaping on lock: delta stability and hole penalty
        delta_stable = 0.0
        hole_penalty = 0.0
        if locked:
            holes_after = self.board.holes()
            created = max(0, holes_after - holes_before)
            hole_penalty = cfg.hole_penalty_per * float(created)

            heights_after = self.board.heights()
            delta_after = (max(heights_after) - min(heights_after)) if heights_after else delta_before
            if delta_after <= delta_before:
                delta_stable = cfg.delta_stable_reward

        reward = line_reward + survival + delta_stable + hole_penalty
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
                "delta_skyline": (max(self.board.heights()) - min(self.board.heights())) if self.board.heights() else 0,
            },
            "reward_components": {
                "line_reward": line_reward,
                "survival": survival,
                "delta_stable": delta_stable,
                "hole_penalty": hole_penalty,
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
