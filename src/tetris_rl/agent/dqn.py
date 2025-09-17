"""Minimal MLP Q-network for the simplified Tetris RL demo.

Removed features: dueling heads, layer norm, dropout – keeping the code small
and easy to read. Hidden layer sizes are still configurable via a list.
"""
from __future__ import annotations

from typing import Tuple, List
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], num_actions: int, hidden_layers: List[int]):
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("Only flat vectors supported")
        if not hidden_layers:
            raise ValueError("hidden_layers must be non-empty")
        self.num_actions = num_actions
        layers: List[nn.Module] = []
        prev = input_shape[0]
        for h in hidden_layers:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU()])
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_actions)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.float()
        feats = self.body(x)
        return self.head(feats)

    # --- Introspection helper for UI ---
    def forward_with_activations(self, x: torch.Tensor):
        """Return (q_values, {'layers': [...]}) for lightweight UI visualization."""
        was_training = self.training
        if was_training:
            self.eval()
        with torch.no_grad():
            x = x.float()
            acts: List[List[float]] = []
            cur = x
            for m in self.body:
                cur = m(cur)
                if isinstance(m, nn.ReLU):
                    vec = cur.squeeze(0).cpu().tolist()
                    if isinstance(vec, list) and len(vec) > 96:
                        step = len(vec)/96.0
                        vec = [vec[int(i*step)] for i in range(96)]
                    acts.append(vec if isinstance(vec, list) else [float(vec)])
            q = self.head(cur)
            q_list = q.squeeze(0).cpu().tolist()
            out = q_list, {"layers": acts}
        if was_training:
            self.train()
        return out
