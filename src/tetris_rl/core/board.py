"""Minimal Tetris board logic for RL demo (not fully optimized).

Implements:
- Standard 10x20 board
- Seven tetrominoes with rotation states
- Spawn at top, simple collision, hard drop/left/right/rotate actions
- Line clearing and aggregate feature computation (holes, aggregate height)

Simplifications:
- No lock delay, gravity: a step auto drops piece by 1; action 'drop' hard drops.
- Rotation system simplified (no wall kicks beyond bounding box shift if possible).
- No scoring beyond line clear count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random

WIDTH = 10
HEIGHT = 20

# Tetromino shapes (rotation states as list of coordinate sets)
PIECES: Dict[str, List[List[Tuple[int,int]]]] = {
    'I': [
        [(0,0),(1,0),(2,0),(3,0)],
        [(2,-1),(2,0),(2,1),(2,2)],
    ],
    'O': [
        [(0,0),(1,0),(0,1),(1,1)]
    ],
    'T': [
        [(1,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(2,1),(1,2)],
        [(0,1),(1,1),(2,1),(1,2)],
        [(1,0),(0,1),(1,1),(1,2)],
    ],
    'S': [
        [(1,0),(2,0),(0,1),(1,1)],
        [(1,0),(1,1),(2,1),(2,2)],
    ],
    'Z': [
        [(0,0),(1,0),(1,1),(2,1)],
        [(2,0),(1,1),(2,1),(1,2)],
    ],
    'J': [
        [(0,0),(0,1),(1,1),(2,1)],
        [(1,0),(2,0),(1,1),(1,2)],
        [(0,1),(1,1),(2,1),(2,2)],
        [(1,0),(1,1),(0,2),(1,2)],
    ],
    'L': [
        [(2,0),(0,1),(1,1),(2,1)],
        [(1,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,1),(0,2)],
        [(0,0),(1,0),(1,1),(1,2)],
    ],
}

PIECE_ORDER = list(PIECES.keys())

@dataclass
class ActivePiece:
    kind: str
    rotation: int
    x: int
    y: int

class TetrisBoard:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # grid holds either 0 (empty) or piece kind letter (str)
        self.grid: List[List[object]] = [[0]*WIDTH for _ in range(HEIGHT)]  # type: ignore[var-annotated]
        self.lines_cleared_total = 0
        self.active: Optional[ActivePiece] = None
        self.spawn_new_piece()

    # --- Piece utilities ---
    def spawn_new_piece(self):
        kind = self.rng.choice(PIECE_ORDER)
        self.active = ActivePiece(kind=kind, rotation=0, x=3, y=0)
        if self._collision(self.active):
            # immediate collision -> top out; mark active None
            self.active = None

    def _current_cells(self, piece: ActivePiece) -> List[Tuple[int,int]]:
        shape = PIECES[piece.kind][piece.rotation % len(PIECES[piece.kind])]
        return [(piece.x+dx, piece.y+dy) for dx,dy in shape]

    def _collision(self, piece: ActivePiece) -> bool:
        for x,y in self._current_cells(piece):
            if x < 0 or x >= WIDTH or y >= HEIGHT:
                return True
            if y >=0 and self.grid[y][x]:
                return True
        return False

    def _lock_piece(self):
        if not self.active:
            return
        for x,y in self._current_cells(self.active):
            if 0 <= y < HEIGHT:
                self.grid[y][x] = self.active.kind  # type: ignore[index]
        self._clear_lines()
        self.spawn_new_piece()

    def _clear_lines(self):
        new_grid = [row for row in self.grid if not all(row)]
        cleared = HEIGHT - len(new_grid)
        if cleared:
            self.lines_cleared_total += cleared
            for _ in range(cleared):
                new_grid.insert(0, [0]*WIDTH)
        self.grid = new_grid

    # --- Actions ---
    def step(self, action: int) -> Tuple[int,int,bool]:
        """Perform action.
        Actions: 0 noop/down, 1 left, 2 right, 3 rotate, 4 hard drop.
        Returns (lines_cleared_delta, top_out, locked)
        """
        if not self.active:
            return 0, True, False
        lines_before = self.lines_cleared_total
        piece = self.active
        if action == 1:  # left
            test = ActivePiece(piece.kind, piece.rotation, piece.x-1, piece.y)
            if not self._collision(test):
                self.active = test
        elif action == 2:  # right
            test = ActivePiece(piece.kind, piece.rotation, piece.x+1, piece.y)
            if not self._collision(test):
                self.active = test
        elif action == 3:  # rotate
            test = ActivePiece(piece.kind, piece.rotation+1, piece.x, piece.y)
            # simple wall nudge
            for dx in (0,-1,1,-2,2):
                t2 = ActivePiece(test.kind, test.rotation, test.x+dx, test.y)
                if not self._collision(t2):
                    self.active = t2
                    break
        elif action == 4:  # hard drop
            while self.active and not self._collision(ActivePiece(piece.kind, piece.rotation, piece.x, piece.y+1)):
                piece = self.active
                self.active = ActivePiece(piece.kind, piece.rotation, piece.x, piece.y+1)
            # lock
            self._lock_piece()
            return self.lines_cleared_total - lines_before, self.active is None, True
        # gravity (down 1)
        piece = self.active
        if piece and not self._collision(ActivePiece(piece.kind, piece.rotation, piece.x, piece.y+1)):
            self.active = ActivePiece(piece.kind, piece.rotation, piece.x, piece.y+1)
            locked = False
        else:
            # lock
            self._lock_piece()
            locked = True
        return self.lines_cleared_total - lines_before, self.active is None, locked

    # --- Features ---
    def holes(self) -> int:
        holes = 0
        grid = self.grid
        for x in range(WIDTH):
            block_seen = False
            col_has = False
            for y in range(HEIGHT):
                cell = grid[y][x]
                if cell:
                    block_seen = True
                    col_has = True
                elif block_seen:
                    holes += 1
            # small branch prediction hint variable kept (col_has)
        return holes

    def per_column_holes(self) -> List[int]:
        """Return count of holes per column (empty after the first filled cell)."""
        counts: List[int] = [0]*WIDTH
        grid = self.grid
        for x in range(WIDTH):
            block_seen = False
            c = 0
            for y in range(HEIGHT):
                if grid[y][x]:
                    block_seen = True
                elif block_seen:
                    c += 1
            counts[x] = c
        return counts

    def hole_columns(self) -> int:
        """Number of columns that contain at least one hole (empty after first filled)."""
        count = 0
        for x in range(WIDTH):
            block_seen = False
            has_hole = False
            for y in range(HEIGHT):
                filled = bool(self.grid[y][x])
                if filled:
                    block_seen = True
                elif block_seen and not filled:
                    has_hole = True
                    break
            if has_hole:
                count += 1
        return count

    def weighted_holes(self, power: float = 1.0) -> float:
        """Depth-weighted hole measure.

        For each column, after the first block is seen, every empty cell below it contributes (depth ** power)
        where depth = (y_index_from_top + 1). Higher power >1 emphasizes deeper holes.
        """
        total = 0.0
        # Clamp power to a sane range
        p = max(0.5, min(power, 4.0))
        for x in range(WIDTH):
            block_seen = False
            for y in range(HEIGHT):
                filled = bool(self.grid[y][x])
                if filled:
                    block_seen = True
                elif block_seen and not filled:
                    depth = y + 1  # 1-based depth
                    total += (depth ** p)
        return total

    def aggregate_height(self) -> int:
        h = 0
        grid = self.grid
        for x in range(WIDTH):
            col_h = 0
            for y in range(HEIGHT):
                if grid[y][x]:
                    col_h = HEIGHT - y
                    break
            h += col_h
        return h

    def heights(self) -> List[int]:
        vals: List[int] = [0]*WIDTH
        grid = self.grid
        for x in range(WIDTH):
            col_h = 0
            for y in range(HEIGHT):
                if grid[y][x]:
                    col_h = HEIGHT - y
                    break
            vals[x] = col_h
        return vals

    def bumpiness(self) -> int:
        hs = self.heights()
        return sum(abs(hs[i]-hs[i+1]) for i in range(len(hs)-1))

    def max_well_depth(self) -> int:
        """Return the maximum well depth.

        A well at column i is defined as max(0, min(h[i-1], h[i+1]) - h[i]). Edges are ignored.
        """
        hs = self.heights()
        if len(hs) < 3:
            return 0
        max_depth = 0
        for i in range(1, len(hs)-1):
            depth = min(hs[i-1], hs[i+1]) - hs[i]
            if depth > max_depth:
                max_depth = depth
        return max(0, max_depth)

    def has_i_dependency(self, threshold: int = 4) -> bool:
        """Heuristic: returns True if there exists a 1-wide deep well that likely requires a vertical I.

        We consider any internal column i such that min(h[i-1], h[i+1]) - h[i] >= threshold.
        """
        return self.max_well_depth() >= max(1, int(threshold))

    def simulate_lines_cleared_if_drop_current(self) -> int:
        """Simulate a hard drop of the current active piece at its current x/rotation and
        return how many lines would be cleared by locking it in place. Does not mutate real state.
        """
        if not self.active:
            return 0
        # find final y by descending until collision
        piece = self.active
        y = piece.y
        while True:
            test = ActivePiece(piece.kind, piece.rotation, piece.x, y+1)
            if self._collision(test):
                break
            y += 1
            if y > HEIGHT:
                break
        final_y = y
        # temp grid with piece locked
        temp_grid: List[List[object]] = [row[:] for row in self.grid]
        for dx, dy in self.active_shape_cells():
            ax = piece.x + dx
            ay = final_y + dy
            if 0 <= ax < WIDTH and 0 <= ay < HEIGHT:
                temp_grid[ay][ax] = piece.kind  # type: ignore[index]
        # count full rows
        cleared = 0
        for row in temp_grid:
            if all(row):
                cleared += 1
        return cleared

    def feature_vector(self) -> List[float]:
        return [float(self.lines_cleared_total), float(self.holes()), float(self.aggregate_height())]

    # --- Additional surface/shape planning features ---
    def row_transitions_total(self) -> int:
        """Count filled↔empty transitions across each row, including boundaries.
        Boundary treated as empty. So for a row, transitions = (left boundary → c0) + Σ(ci-1→ci) + (c9 → right boundary).
        """
        total = 0
        for y in range(HEIGHT):
            prev_filled = False  # boundary empty
            for x in range(WIDTH):
                cur_filled = bool(self.grid[y][x])
                if cur_filled != prev_filled:
                    total += 1
                prev_filled = cur_filled
            # right boundary (empty)
            if prev_filled:  # filled → empty
                total += 1
        return total

    def col_transitions_total(self) -> int:
        """Count filled↔empty transitions down each column, including top/bottom boundaries.
        Boundary treated as empty.
        """
        total = 0
        for x in range(WIDTH):
            prev_filled = False  # top boundary empty
            for y in range(HEIGHT):
                cur_filled = bool(self.grid[y][x])
                if cur_filled != prev_filled:
                    total += 1
                prev_filled = cur_filled
            if prev_filled:  # filled → empty at bottom boundary
                total += 1
        return total

    def overhang_cells_count(self) -> int:
        """Count filled cells that have an empty cell directly above (floating/overhang)."""
        count = 0
        for y in range(1, HEIGHT):
            row = self.grid[y]
            row_above = self.grid[y-1]
            for x in range(WIDTH):
                if row[x] and not row_above[x]:
                    count += 1
        return count

    def covered_by_blocks_count(self) -> int:
        """Count filled cells that have another filled cell directly above (buried under blocks)."""
        count = 0
        for y in range(1, HEIGHT):
            row = self.grid[y]
            row_above = self.grid[y-1]
            for x in range(WIDTH):
                if row[x] and row_above[x]:
                    count += 1
        return count

    def total_blocks(self) -> int:
        return sum(1 for y in range(HEIGHT) for x in range(WIDTH) if self.grid[y][x])

    def landing_final_y_current(self) -> Optional[int]:
        """Return final y of the active piece's origin after drop (does not mutate)."""
        if not self.active:
            return None
        piece = self.active
        y = piece.y
        while True:
            test = ActivePiece(piece.kind, piece.rotation, piece.x, y+1)
            if self._collision(test):
                break
            y += 1
            if y > HEIGHT:
                break
        return y

    def landing_height_current(self) -> float:
        """Compute landing height as the height of the highest block of the piece after placement (0..HEIGHT).
        Returns float height (not normalized).
        """
        if not self.active:
            return 0.0
        final_y = self.landing_final_y_current()
        if final_y is None:
            return 0.0
        max_block_y = max(final_y + dy for _, dy in self.active_shape_cells())
        height = max(0, HEIGHT - (max_block_y + 1))
        return float(height)

    def piece_contact_current(self) -> int:
        """Compute contact sides for the active piece when dropped at current x/rotation.
        Counts side contacts (left/right) and bottom contacts against either border or filled cells.
        """
        if not self.active:
            return 0
        final_y = self.landing_final_y_current()
        if final_y is None:
            return 0
        contact = 0
        piece = self.active
        cells = [(piece.x + dx, final_y + dy) for dx, dy in self.active_shape_cells()]
        occ = set(cells)
        for (ax, ay) in cells:
            # left
            nx, ny = ax-1, ay
            if nx < 0 or (0 <= ny < HEIGHT and self.grid[ny][nx] and (nx,ny) not in occ):
                contact += 1
            # right
            nx = ax+1; ny = ay
            if nx >= WIDTH or (0 <= ny < HEIGHT and self.grid[ny][nx] and (nx,ny) not in occ):
                contact += 1
            # bottom
            nx = ax; ny = ay+1
            if ny >= HEIGHT or (0 <= ny < HEIGHT and self.grid[ny][nx] and (nx,ny) not in occ):
                contact += 1
        return contact

    def _temp_grid_with_current_locked(self) -> List[List[object]]:
        """Return a copy of the grid with current piece locked at its landing position."""
        if not self.active:
            return [row[:] for row in self.grid]
        final_y = self.landing_final_y_current()
        temp = [row[:] for row in self.grid]
        if final_y is None:
            return temp
        piece = self.active
        for dx, dy in self.active_shape_cells():
            ax = piece.x + dx; ay = final_y + dy
            if 0 <= ax < WIDTH and 0 <= ay < HEIGHT:
                temp[ay][ax] = piece.kind  # type: ignore[index]
        return temp

    def count_ready_lines_after_current(self) -> int:
        """Count rows that would be 1 cell away from clearing after locking the current piece.
        A ready line = row with exactly 1 empty cell (i.e., 9 filled).
        """
        temp = self._temp_grid_with_current_locked()
        ready = 0
        for y in range(HEIGHT):
            filled = sum(1 for x in range(WIDTH) if temp[y][x])
            if filled == WIDTH - 1:
                ready += 1
        return ready

    def count_deep_i_wells(self, threshold: int = 4) -> int:
        hs = self.heights()
        cnt = 0
        for i in range(1, len(hs)-1):
            depth = min(hs[i-1], hs[i+1]) - hs[i]
            if depth >= threshold:
                cnt += 1
        return cnt

    def count_o_gaps(self) -> int:
        """Count simple 2x2 empty squares that could fit an O piece (support heuristic: bottom row supported or bottom boundary)."""
        cnt = 0
        for y in range(HEIGHT-1):
            for x in range(WIDTH-1):
                if not self.grid[y][x] and not self.grid[y][x+1] and not self.grid[y+1][x] and not self.grid[y+1][x+1]:
                    # support: either bottom row is HEIGHT-1 (bottom boundary) or cells below are filled
                    if y+1 == HEIGHT-1 or (self.grid[y+2][x] if y+2 < HEIGHT else True) and (self.grid[y+2][x+1] if y+2 < HEIGHT else True):
                        cnt += 1
        return cnt

    # --- Serialization helpers ---
    def board_state(self, include_active: bool = True) -> List[str]:
        """Return a simple textual representation of the board rows (top->bottom).
        Each row is a string of characters: '.' empty, '#' locked block, '*' active piece block.
        """
        rows = []
        active_cells = set()
        if include_active and self.active:
            for x,y in self._current_cells(self.active):
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    active_cells.add((x,y))
        for y in range(HEIGHT):
            chars = []
            for x in range(WIDTH):
                if (x,y) in active_cells:
                    chars.append('*')
                else:
                    cell = self.grid[y][x]
                    chars.append('#' if cell else '.')
            rows.append(''.join(chars))
        return rows

    def snapshot(self) -> dict:
        """Return JSON-serializable snapshot for UI streaming."""
        palette = {
            'I': '#00f0f0',
            'O': '#f0f000',
            'T': '#a000f0',
            'S': '#00f000',
            'Z': '#f00000',
            'J': '#0000f0',
            'L': '#f0a000'
        }
        # build matrix of color codes or empty
        matrix = []
        for y in range(HEIGHT):
            row_colors = []
            for x in range(WIDTH):
                val = self.grid[y][x]
                row_colors.append(palette.get(str(val), '') if val else '')
            matrix.append(row_colors)
        active_cells = []
        if self.active:
            for x,y in self._current_cells(self.active):
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    active_cells.append({"x": x, "y": y, "kind": self.active.kind, "color": palette.get(self.active.kind, '#ffffff')})
        return {
            "lines_cleared_total": self.lines_cleared_total,
            "holes": self.holes(),
            "aggregate_height": self.aggregate_height(),
            "bumpiness": self.bumpiness(),
            "board": self.board_state(),
            "matrix": matrix,
            "active_cells": active_cells,
            "palette": palette,
        }

    # --- Helpers for observation simulation ---
    def active_shape_cells(self, rotation: Optional[int] = None) -> List[Tuple[int, int]]:
        """Return the relative shape cells of the current active piece at given rotation.

        If rotation is None, uses the current active rotation.
        Returns an empty list if there is no active piece.
        """
        if not self.active:
            return []
        kind = self.active.kind
        r = self.active.rotation if rotation is None else rotation
        shape = PIECES[kind][r % len(PIECES[kind])]
        return list(shape)

    def simulate_drop_at_left(self, x_left: int) -> dict:
        """Simulate hard-dropping the current piece with its leftmost occupied cell at x_left.

        Returns a dict with:
          { 'valid': bool, 'y': int | None, 'heights_after': List[int] | None, 'holes_after': int | None }

        Policy:
          - If invalid alignment (out of bounds or no non-colliding y), returns {'valid': False, ...}.
          - Does not mutate board state.
        """
        if not self.active:
            return {"valid": False, "y": None, "heights_after": None, "holes_after": None}

        # Determine shape extents
        shape = self.active_shape_cells()
        if not shape:
            return {"valid": False, "y": None, "heights_after": None, "holes_after": None}
        min_dx = min(dx for dx, _ in shape)
        max_dx = max(dx for dx, _ in shape)
        width_shape = (max_dx - min_dx + 1)

        # Compute piece.x so that leftmost occupied cell aligns with x_left
        piece_x = x_left - min_dx
        # Quick bounds check for horizontal range
        if piece_x + min_dx < 0 or piece_x + max_dx >= WIDTH:
            return {"valid": False, "y": None, "heights_after": None, "holes_after": None}

        # Scan downward to find final y using collision on the current grid
        last_ok_y: Optional[int] = None
        # Allow starting above the top to accommodate negative dy in shapes
        for y in range(-4, HEIGHT):
            test = ActivePiece(self.active.kind, self.active.rotation, piece_x, y)
            if not self._collision(test):
                last_ok_y = y
            else:
                # If we've already had a valid y and now colliding, we can stop
                if last_ok_y is not None:
                    break
        if last_ok_y is None:
            return {"valid": False, "y": None, "heights_after": None, "holes_after": None}

        final_y = last_ok_y

        # Build a temp grid with the piece locked at (piece_x, final_y)
        temp_grid: List[List[object]] = [row[:] for row in self.grid]
        for dx, dy in shape:
            ax = piece_x + dx
            ay = final_y + dy
            if 0 <= ax < WIDTH and 0 <= ay < HEIGHT:
                temp_grid[ay][ax] = self.active.kind  # type: ignore[index]
        # Compute heights_after
        heights_after: List[int] = []
        for x in range(WIDTH):
            col_h = 0
            for y in range(HEIGHT):
                if temp_grid[y][x]:
                    col_h = HEIGHT - y
                    break
            heights_after.append(col_h)

        # Compute holes_after
        holes_after = 0
        for x in range(WIDTH):
            block_seen = False
            for y in range(HEIGHT):
                if temp_grid[y][x]:
                    block_seen = True
                elif block_seen and not temp_grid[y][x]:
                    holes_after += 1

        return {"valid": True, "y": final_y, "heights_after": heights_after, "holes_after": holes_after}

    def simulate_drop_stats_at_left(self, x_left: int, y_first: Optional[List[int]] = None) -> dict:
        """Fast, lightweight drop simulation for a given left alignment.

        Returns a dict with keys:
          - valid: bool
          - final_y: int | None
          - h_after_c: int | None  (predicted height for column=x_left after placement)
          - created_new: int | None (number of new holes created by placing the piece)

        Notes:
          - Does not mutate board state.
          - Does not simulate line clearing (consistent with simulate_drop_at_left).
          - Uses analytical drop based on first-filled row per column and shape column profiles.
        """
        if not self.active:
            return {"valid": False, "final_y": None, "h_after_c": None, "created_new": None}

        # Shape and horizontal span
        shape = self.active_shape_cells()
        if not shape:
            return {"valid": False, "final_y": None, "h_after_c": None, "created_new": None}
        min_dx = min(dx for dx, _ in shape)
        max_dx = max(dx for dx, _ in shape)
        piece_x = x_left - min_dx
        if piece_x + min_dx < 0 or piece_x + max_dx >= WIDTH:
            return {"valid": False, "final_y": None, "h_after_c": None, "created_new": None}

        # Precompute per-dx min/max dy for the shape columns
        per_dx_min_dy: Dict[int, int] = {}
        per_dx_max_dy: Dict[int, int] = {}
        for dx, dy in shape:
            if dx not in per_dx_min_dy or dy < per_dx_min_dy[dx]:
                per_dx_min_dy[dx] = dy
            if dx not in per_dx_max_dy or dy > per_dx_max_dy[dx]:
                per_dx_max_dy[dx] = dy

        # Determine the first filled cell (from top) per column -> y_first
        if y_first is None:
            # Derive from current heights for efficiency
            heights = self.heights()
            y_first = [HEIGHT - h if h > 0 else HEIGHT for h in heights]

        # Compute analytical final y: constrained by each column's lowest shape block
        final_y = HEIGHT  # start high, take min
        for dx, dy_max in per_dx_max_dy.items():
            col = piece_x + dx
            if col < 0 or col >= WIDTH:
                return {"valid": False, "final_y": None, "h_after_c": None, "created_new": None}
            limit = y_first[col] - 1 - dy_max
            if limit < final_y:
                final_y = limit
        if final_y is None:
            return {"valid": False, "final_y": None, "h_after_c": None, "created_new": None}

        # Predicted height for column c == x_left (leftmost occupied column)
        c = x_left
        dy_min_left = per_dx_min_dy.get(min_dx, 0)
        y_new_top_c = final_y + dy_min_left
        old_y_first_c = y_first[c]
        new_y_first_c = old_y_first_c if y_new_top_c >= old_y_first_c else y_new_top_c
        h_after_c = 0 if new_y_first_c >= HEIGHT else (HEIGHT - new_y_first_c)

        # Compute newly created holes by scanning only affected ranges per overlapped column
        created_new = 0
        for dx, dy_min in per_dx_min_dy.items():
            col = piece_x + dx
            y_top_new = final_y + dy_min  # topmost new block in this column
            old_yf = y_first[col]
            if y_top_new < old_yf:
                # Empty cells in (y_top_new+1 .. old_yf-1) become holes if empty in original grid
                start_y = y_top_new + 1
                end_y = old_yf  # exclusive
                if start_y < 0:
                    start_y = 0
                if end_y > HEIGHT:
                    end_y = HEIGHT
                for y in range(start_y, end_y):
                    # Only count if originally empty (we don't modify grid here)
                    if 0 <= y < HEIGHT and not self.grid[y][col]:
                        created_new += 1

        return {"valid": True, "final_y": final_y, "h_after_c": h_after_c, "created_new": created_new}
