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
        for x in range(WIDTH):
            block_seen = False
            for y in range(HEIGHT):
                if self.grid[y][x]:
                    block_seen = True
                elif block_seen and not self.grid[y][x]:
                    holes += 1
        return holes

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
        for x in range(WIDTH):
            col_h = 0
            for y in range(HEIGHT):
                if self.grid[y][x]:
                    col_h = HEIGHT - y
                    break
            h += col_h
        return h

    def heights(self) -> List[int]:
        vals: List[int] = []
        for x in range(WIDTH):
            col_h = 0
            for y in range(HEIGHT):
                if self.grid[y][x]:
                    col_h = HEIGHT - y
                    break
            vals.append(col_h)
        return vals

    def bumpiness(self) -> int:
        hs = self.heights()
        return sum(abs(hs[i]-hs[i+1]) for i in range(len(hs)-1))

    def feature_vector(self) -> List[float]:
        return [float(self.lines_cleared_total), float(self.holes()), float(self.aggregate_height())]

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
