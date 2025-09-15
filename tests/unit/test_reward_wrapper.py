import pytest
from types import SimpleNamespace

from tetris_rl.env.reward_wrapper import RewardShapingWrapper


class DummyEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(sample=lambda: 0)
        self.observation_space = SimpleNamespace(shape=(1,))
        self._step_count = 0

    def reset(self, *args, **kwargs):  # gymnasium-style return (obs, info)
        return 0, {}

    def step(self, action):  # return (obs, reward, terminated, truncated, info)
        self._step_count += 1
        return self._step_count, 0.0, False, False, {}


def test_reward_wrapper_composes_weights():
    env = DummyEnv()
    w = RewardShapingWrapper(env, line_clear_weight=2.0, hole_weight=-1.0, height_weight=-0.2, step_penalty=-0.05)
    assert w.weights == {
        "line_clear": 2.0,
        "hole": -1.0,
        "height": -0.2,
        "step": -0.05,
    }
    obs, reward, terminated, truncated, info = w.step(0)
    assert obs == 1
    assert reward == 0.0  # still pass-through
    assert not terminated and not truncated
