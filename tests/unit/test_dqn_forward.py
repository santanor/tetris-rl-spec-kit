import torch
import pytest

from tetris_rl.agent.dqn import DQN


def test_dqn_forward_shape():
    model = DQN((4,), 6)  # pretend 4-feature vector, 6 actions
    x = torch.randn(3, 4)  # batch of 3
    out = model(x)
    assert out.shape == (3, 6)
    # zeros placeholder behavior for now
    assert torch.allclose(out, torch.zeros_like(out))
