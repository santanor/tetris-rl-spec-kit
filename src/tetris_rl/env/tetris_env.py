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
        self.next_piece_onehot: list[float] | None = None  # optional preview placeholder
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

        # 7) Additional scalar features requested
        # Aggregate height (sum heights) normalized by max 10*20=200
        agg_height = sum(heights) / 200.0 if heights else 0.0
        # Bumpiness normalized: worst-case ~ 9 * 20 = 180 (adjacent diffs of 20)
        bump_raw = self.board.bumpiness()
        bump_norm = min(max(bump_raw / 180.0, 0.0), 1.0)
        # Max well depth normalized by 20
        well_max = self.board.max_well_depth() / 20.0
        # Lines cleared possible from current placement at current alignment
        lines_possible_here = 0.0
        if self.board.active is not None:
            try:
                lines_possible_here = float(self.board.simulate_lines_cleared_if_drop_current()) / 4.0
            except Exception:
                lines_possible_here = 0.0
        # Piece dependency: I-piece dependency flag (0/1)
        i_dep = 1.0 if self.board.has_i_dependency(threshold=4) else 0.0

        # 8) Row/column transitions (normalized by simple upper-bounds)
        row_trans = self.board.row_transitions_total()
        col_trans = self.board.col_transitions_total()
        # Rough upper-bounds: per row up to 11 transitions → 220, per column up to 21 → 210
        row_trans_norm = min(max(row_trans / 220.0, 0.0), 1.0)
        col_trans_norm = min(max(col_trans / 210.0, 0.0), 1.0)
        # 9) Covered (buried) and overhang counts (normalized by board cells 200)
        buried = self.board.covered_by_blocks_count() / 200.0
        overhang = self.board.overhang_cells_count() / 200.0
        # 10) Total blocks placed (normalized by 200)
        total_blocks = self.board.total_blocks() / 200.0
        # 11) Landing height (normalized), contact, ready lines after placement
        landing_h = self.board.landing_height_current() / 20.0
        contact = min(self.board.piece_contact_current() / 40.0, 1.0)  # rough scale
        ready_lines = min(self.board.count_ready_lines_after_current() / 4.0, 1.0)
        # 12) Dependency counts (I wells, O gaps) normalized
        dep_i = min(self.board.count_deep_i_wells(threshold=4) / 10.0, 1.0)
        dep_o = min(self.board.count_o_gaps() / 10.0, 1.0)
        # 13) Next piece one-hot if available (placeholder: zeros)
        next_piece = self.next_piece_onehot or [0.0]*7

        feats = (
            norm_heights
            + predicted_heights
            + piece_one_hot
            + rot_one_hot
            + [skyline_delta]
            + norm_holes_per_col
            + predicted_created_holes
            + [x_norm, y_norm]
            + [agg_height, bump_norm, well_max, lines_possible_here, i_dep]
            + [row_trans_norm, col_trans_norm, buried, overhang, total_blocks]
            + [landing_h, contact, ready_lines, dep_i, dep_o]
            + next_piece
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
        bumpiness_before = self.board.bumpiness()
        max_h_before = max(heights_before) if heights_before else 0

        lines_delta, top_out, locked = self.board.step(int(action))

        # Core components
        # Base line rewards (exaggerate Tetris clears via config)
        line_reward = cfg.line_table().get(lines_delta, 0.0) if lines_delta else 0.0
        # Efficiency bonus encourages multi-line clears vs separate singles
        if lines_delta and lines_delta > 1:
            line_reward += cfg.efficiency_bonus_base * float(lines_delta - 1)
        survival = cfg.survival_reward if not top_out else 0.0

        # Shaping on lock: delta stability and hole penalty
        delta_stable = 0.0
        hole_penalty = 0.0
        bumpiness_penalty = 0.0
        height_penalty = 0.0
        # Initialize additional shaping components to zero (set on lock)
        transition_penalty = 0.0
        buried_penalty = 0.0
        overhang_penalty = 0.0
        landing_penalty = 0.0
        ready_bonus = 0.0
        well_bonus = 0.0
        blocked_well_pen = 0.0
        if locked:
            holes_after = self.board.holes()
            created = max(0, holes_after - holes_before)
            hole_penalty = cfg.hole_penalty_per * float(created)

            heights_after = self.board.heights()
            delta_after = (max(heights_after) - min(heights_after)) if heights_after else delta_before
            if delta_after <= delta_before:
                delta_stable = cfg.delta_stable_reward
            # Penalize roughness change and high stacks
            bump_after = self.board.bumpiness()
            max_h_after = max(heights_after) if heights_after else max_h_before
            # Small negative proportional to bumpiness (normalized inside penalty scale choice)
            bumpiness_penalty = cfg.bumpiness_penalty_scale * float(bump_after)
            height_penalty = cfg.height_penalty_scale * float(max_h_after)

            # Row/column transition penalties (global surface jaggedness)
            row_trans = self.board.row_transitions_total()
            col_trans = self.board.col_transitions_total()
            transition_penalty = cfg.row_transition_penalty * float(row_trans) + cfg.col_transition_penalty * float(col_trans)

            # Buried blocks and overhang penalties
            buried_blocks = self.board.covered_by_blocks_count()
            overhang_cells = self.board.overhang_cells_count()
            buried_penalty = cfg.buried_block_penalty * float(buried_blocks)
            overhang_penalty = cfg.overhang_penalty * float(overhang_cells)

            # Landing height penalty (higher landings are riskier)
            landing_h = self.board.landing_height_current()
            landing_penalty = cfg.landing_height_penalty * float(landing_h)

            # Ready-line bonus and well-usage bonus (heuristics)
            ready_lines = self.board.count_ready_lines_after_current()
            ready_bonus = cfg.ready_line_bonus * float(ready_lines)
            # Well usage bonus: if we just cleared lines and there exists a deep I-well, add bonus (simple proxy)
            well_bonus = cfg.well_usage_bonus * float(1 if (lines_delta >= 1 and self.board.count_deep_i_wells(threshold=4) > 0) else 0)

            # Blocked well penalty: if max well got blocked by placement (approximate by increase in min neighbor above well)
            try:
                hs_b = heights_before
                hs_a = heights_after
                if len(hs_b) == len(hs_a) and len(hs_a) >= 3:
                    for i in range(1, len(hs_a)-1):
                        depth_b = max(0, min(hs_b[i-1], hs_b[i+1]) - hs_b[i])
                        depth_a = max(0, min(hs_a[i-1], hs_a[i+1]) - hs_a[i])
                        if depth_b >= 4 and depth_a < depth_b:
                            blocked_well_pen += cfg.blocked_well_penalty
                            break
            except Exception:
                pass

        # Sum all components (transition, buried/overhang, landing, planning bonuses)
        reward = (line_reward + survival + delta_stable + hole_penalty + bumpiness_penalty + height_penalty)
        if locked:
            reward += (transition_penalty + buried_penalty + overhang_penalty + landing_penalty + ready_bonus + well_bonus + blocked_well_pen)
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
                "max_well_depth": self.board.max_well_depth(),
                "i_dependency": int(self.board.has_i_dependency(threshold=4)),
            },
            "reward_components": {
                "line_reward": line_reward,
                "survival": survival,
                "delta_stable": delta_stable,
                "hole_penalty": hole_penalty,
                "bumpiness_penalty": bumpiness_penalty,
                "height_penalty": height_penalty,
                "transition_penalty": (transition_penalty if locked else 0.0),
                "buried_penalty": (buried_penalty if locked else 0.0),
                "overhang_penalty": (overhang_penalty if locked else 0.0),
                "landing_penalty": (landing_penalty if locked else 0.0),
                "ready_bonus": (ready_bonus if locked else 0.0),
                "well_bonus": (well_bonus if locked else 0.0),
                "blocked_well_penalty": (blocked_well_pen if locked else 0.0),
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
