"""Simple FIFO replay buffer."""
from __future__ import annotations

from typing import List, Tuple
import random
import numpy as np
import torch

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Efficient stacking: states / next_states are tuples of np.ndarray with identical shapes.
        # Using np.stack then from_numpy avoids the PyTorch warning about list-of-ndarray conversion.
        states_np = np.stack(states).astype(np.float32, copy=False)
        next_states_np = np.stack(next_states).astype(np.float32, copy=False)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)
        return (
            torch.from_numpy(states_np),
            actions_t,
            rewards_t,
            torch.from_numpy(next_states_np),
            dones_t,
        )

    def __len__(self):
        return len(self.memory)
