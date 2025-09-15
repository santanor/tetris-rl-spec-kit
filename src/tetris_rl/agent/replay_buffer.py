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
        return (
            torch.tensor(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.memory)
