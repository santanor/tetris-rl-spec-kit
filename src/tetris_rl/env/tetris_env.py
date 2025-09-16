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
        # Extended feature vector (dynamic) will be built on first reset; placeholder shape here.
        self._obs_dim = 0
        self.observation_space = spaces.Box(low=-10_000.0, high=10_000.0, shape=(1,), dtype=np.float32)  # type: ignore[arg-type]
        self.action_space = spaces.Discrete(5)  # type: ignore[arg-type]
        self._step_count = 0

    def _build_obs(self):
        """Construct extended normalized feature vector.

        Components (example order):
          0: step_fraction (step / max_steps)
          1: holes (normalized)
          2: aggregate_height (normalized)
          3: bumpiness (normalized)
          4: lines_cleared_total (normalized)
          5: holes_delta
          6: height_delta
          7: bumpiness_delta
          8: weighted_holes (normalized)
          9: weighted_holes_delta
          Piece one-hot (7): indices 10-16
        Total = 17 features.
        """
        max_holes = 200.0
        max_agg_height = 10 * 20.0
        max_bump = 8 * 20.0
        max_lines = 200.0
        max_weighted = 20.0 ** 4  # loose upper bound with power weighting

        holes = self.board.holes()
        agg_h = self.board.aggregate_height()
        bump = self.board.bumpiness()
        weighted_h = self.board.weighted_holes(getattr(self.reward_config, 'holes_depth_power', 1.0))

        step_fraction = float(self._step_count) / float(max(1, self.max_steps))

        # Deltas use last cached values
        holes_delta = holes - getattr(self, '_prev_holes', 0.0)
        height_delta = agg_h - getattr(self, '_prev_agg_height', 0.0)
        bump_delta = bump - getattr(self, '_prev_bump', 0.0)
        weighted_delta = weighted_h - getattr(self, '_prev_weighted_h', 0.0)

        piece_one_hot = [0.0]*7
        if self.board.active:
            kinds = ['I','O','T','S','Z','J','L']
            try:
                piece_one_hot[kinds.index(self.board.active.kind)] = 1.0
            except ValueError:
                pass

        feats = [
            step_fraction,
            holes / max_holes,
            agg_h / max_agg_height,
            bump / max_bump,
            self.board.lines_cleared_total / max_lines,
            holes_delta,
            height_delta / 20.0,
            bump_delta / 50.0,
            weighted_h / max_weighted,
            weighted_delta / max_weighted,
        ] + piece_one_hot

        arr = np.array(feats, dtype=np.float32)
        self._obs_dim = arr.shape[0]
        return arr

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        if seed is not None:
            self.board = TetrisBoard(seed=seed)
        else:
            self.board = TetrisBoard(seed=0)
        self._step_count = 0
        # reset historical metrics for deltas
        self._prev_holes = 0.0
        self._prev_agg_height = 0.0
        self._prev_bump = 0.0
        self._prev_weighted_h = 0.0
        obs = self._build_obs()
        # Update observation_space to reflect real dimension
        if self.observation_space.shape != (obs.shape[0],):  # type: ignore[truthy-bool]
            self.observation_space = spaces.Box(low=-10_000.0, high=10_000.0, shape=(obs.shape[0],), dtype=np.float32)  # type: ignore[arg-type]
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        # Capture pre-step structural metrics
        prev_holes = self.board.holes()
        prev_weighted_holes = self.board.weighted_holes(getattr(self.reward_config, 'holes_depth_power', 1.0))
        prev_height = self.board.aggregate_height()
        # Average row density (exclude fully empty rows) before step
        grid = self.board.grid
        width = len(grid[0]) if grid else 0
        def _avg_density(g):
            total = 0.0
            count = 0
            for row in g:
                filled = sum(1 for c in row if c)
                if filled == 0:
                    continue
                total += filled / width if width else 0.0
                count += 1
            return (total / count) if count else 0.0
        prev_density = _avg_density(grid)
        prev_bump = self.board.bumpiness()

        self._step_count += 1
        lines_delta, top_out, locked = self.board.step(int(action))

        # Base line clear reward (weighted) + step penalty
        cfg = self.reward_config
        if lines_delta:
            base = cfg.line_table().get(lines_delta, float(lines_delta))
        else:
            base = 0.0
        step_penalty = cfg.step_penalty

        # Post-step metrics
        holes_after = self.board.holes()
        weighted_holes_after = self.board.weighted_holes(getattr(cfg, 'holes_depth_power', 1.0))
        height_after = self.board.aggregate_height()
        bump_after = self.board.bumpiness()
        density_after = _avg_density(self.board.grid)

        # Deltas
        holes_delta = holes_after - prev_holes
        weighted_holes_delta = weighted_holes_after - prev_weighted_holes
        height_delta = height_after - prev_height
        bump_delta = bump_after - prev_bump
        density_delta = density_after - prev_density

        # Structural shaping
        structural_holes = cfg.holes_weight * holes_delta
        weighted_structural = 0.0
        if getattr(cfg, 'weighted_holes_weight', 0.0) != 0.0:
            weighted_structural = cfg.weighted_holes_weight * weighted_holes_delta
        if cfg.conditional_height and (prev_height < cfg.height_threshold) and (height_after < cfg.height_threshold):
            structural_height = 0.0
        else:
            structural_height = cfg.height_weight * height_delta
        structural_bump = cfg.bumpiness_weight * bump_delta
        density_structural = 0.0
        if getattr(cfg, 'row_density_delta_weight', 0.0) != 0.0:
            density_structural = cfg.row_density_delta_weight * density_delta
            if lines_delta:  # amplify on successful clears (compact play leading to clears)
                scale = getattr(cfg, 'row_density_line_clear_scale', 1.0)
                density_structural *= scale
        structural = structural_holes + structural_height + structural_bump + weighted_structural + density_structural

        # Absolute penalties
        abs_holes_penalty = cfg.holes_abs_weight * holes_after if hasattr(cfg, 'holes_abs_weight') else 0.0
        abs_weighted_penalty = 0.0
        if hasattr(cfg, 'weighted_holes_abs_weight') and cfg.weighted_holes_abs_weight != 0.0:
            abs_weighted_penalty = cfg.weighted_holes_abs_weight * weighted_holes_after

        # Survival bonus
        survival_bonus = cfg.survival_bonus if not top_out else 0.0

        abs_density_shaping = 0.0
        if getattr(cfg, 'row_density_abs_weight', 0.0) != 0.0:
            abs_density_shaping = cfg.row_density_abs_weight * density_after
        reward = base + step_penalty + structural + survival_bonus + abs_holes_penalty + abs_weighted_penalty + abs_density_shaping
        if top_out:
            reward += cfg.top_out_penalty

        terminated = top_out
        truncated = self._step_count >= self.max_steps
        obs = self._build_obs()
        # cache prev metrics for next delta
        self._prev_holes = holes_after
        self._prev_agg_height = height_after
        self._prev_bump = bump_after
        self._prev_weighted_h = weighted_holes_after
        info: Dict[str, Any] = {
            "lines_delta": lines_delta,
            "locked": locked,
            "lines_cleared_total": self.board.lines_cleared_total,
            "reward_components": {
                "base_line": base,
                "step_penalty": step_penalty,
                "structural": structural,
                "abs_holes": abs_holes_penalty,
                "abs_weighted_holes": abs_weighted_penalty,
                "survival": survival_bonus,
                "top_out": cfg.top_out_penalty if top_out else 0.0,
                "structural_breakdown": {
                    "holes": structural_holes,
                    "height": structural_height,
                    "bumpiness": structural_bump,
                    "weighted_holes_delta": weighted_structural,
                    "weighted_holes_abs": abs_weighted_penalty,
                    "abs_holes_penalty": abs_holes_penalty,
                    "density_delta": density_structural,
                    "density_abs": abs_density_shaping,
                },
                "config_hash": hash(tuple(sorted(cfg.to_dict().items()))),
            },
            "holes_delta": holes_delta,
            "height_delta": height_delta,
            "bump_delta": bump_delta,
            "weighted_holes_delta": weighted_holes_delta,
            "weighted_holes_after": weighted_holes_after,
            "density_delta": density_delta,
            "density_after": density_after,
        }
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover
        grid_chars = []
        for row in self.board.grid:
            grid_chars.append(''.join('#' if c else '.' for c in row))
        print('\n'.join(grid_chars[-10:]))

    def close(self):  # pragma: no cover
        pass
