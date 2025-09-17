"""Scalable DQN / Q-network implementation.

Features:
    - Arbitrary hidden layer sizes (list[int])
    - Optional LayerNorm after each linear layer (pre-activation)
    - Optional dropout
    - Optional Dueling architecture (value + advantage streams)
"""
from __future__ import annotations

from typing import Tuple, List, Optional
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        hidden_layers: List[int],
        dueling: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("Only flat feature vectors supported in this version")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        self.dueling = dueling
        self.num_actions = num_actions
        in_f = input_shape[0]
        layers: List[nn.Module] = []
        prev = in_f
        for h in hidden_layers:
            lin = nn.Linear(prev, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.body = nn.Sequential(*layers)

        if dueling:
            self.adv_head = nn.Linear(prev, num_actions)
            self.val_head = nn.Linear(prev, 1)
            nn.init.xavier_uniform_(self.adv_head.weight)
            nn.init.zeros_(self.adv_head.bias)
            nn.init.xavier_uniform_(self.val_head.weight)
            nn.init.zeros_(self.val_head.bias)
        else:
            self.head = nn.Linear(prev, num_actions)
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.float()
        feats = self.body(x)
        if self.dueling:
            adv = self.adv_head(feats)
            val = self.val_head(feats)
            # Broadcast-add value then subtract mean advantage for stability
            adv_mean = adv.mean(dim=-1, keepdim=True)
            q = val + (adv - adv_mean)
            return q
        else:
            return self.head(feats)

    # --- Introspection helper for UI ---
    def forward_with_activations(self, x: torch.Tensor):
        """Forward pass that also returns intermediate activations for visualization.

        Returns (q_values, activations_dict) where activations_dict structure:
            {
              'layers': [list[float], ...],  # post-ReLU activations per hidden layer
              'advantages': list[float] | None,
              'value': float | None
            }
        """
        was_training = self.training
        # Use eval mode so dropout / layernorm behave consistently for display
        if was_training:
            self.eval()
        with torch.no_grad():
            x = x.float()
            acts: List[List[float]] = []
            current = x
            # Manually iterate through body to capture after each ReLU
            for m in self.body:
                current = m(current)
                if isinstance(m, nn.ReLU):
                    vec = current.squeeze(0).detach().cpu().tolist()
                    # Optional lightweight subsampling for very wide layers
                    if isinstance(vec, list) and len(vec) > 96:
                        # even subsample to 96 points (UI density control)
                        step = len(vec) / 96.0
                        sampled = [vec[int(i*step)] for i in range(96)]
                        acts.append(sampled)
                    else:
                        acts.append(vec if isinstance(vec, list) else [float(vec)])
            if self.dueling:
                adv = self.adv_head(current)
                val = self.val_head(current)
                adv_mean = adv.mean(dim=-1, keepdim=True)
                q = val + (adv - adv_mean)
                q_list = q.squeeze(0).cpu().tolist()
                adv_list = adv.squeeze(0).cpu().tolist()
                val_scalar = float(val.squeeze(0).item())
                result = q_list, {"layers": acts, "advantages": adv_list, "value": val_scalar}
            else:
                q = self.head(current)
                q_list = q.squeeze(0).cpu().tolist()
                result = q_list, {"layers": acts, "advantages": None, "value": None}
        if was_training:
            self.train()
        return result
