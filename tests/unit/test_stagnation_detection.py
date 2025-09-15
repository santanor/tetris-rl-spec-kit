import pytest
from types import SimpleNamespace

from tetris_rl.env.stagnation_wrapper import StagnationTerminationWrapper


class PosRewardEveryThird:
    def __init__(self):
        self.step_count = 0

    def reset(self, *args, **kwargs):
        self.step_count = 0
        return 0, {}

    def step(self, action):
        self.step_count += 1
        reward = 1.0 if self.step_count % 3 == 0 else 0.0
        return self.step_count, reward, False, False, {}


def test_stagnation_wrapper_exists():
    env = PosRewardEveryThird()
    w = StagnationTerminationWrapper(env, stagnation_limit=10)
    obs, info = w.reset()
    assert w.steps_since_progress == 0
    seq = []
    for _ in range(5):
        _, reward, *_ = w.step(0)
        seq.append((reward, w.steps_since_progress))
    # After 5 steps, positive rewards at steps 3 only -> counters pattern
    # steps_since_progress should reset to 0 at the positive reward then increment afterwards
    rewards = [r for r, _ in seq]
    counters = [c for _, c in seq]
    assert rewards == [0.0, 0.0, 1.0, 0.0, 0.0]
    # counters: 1,2,0,1,2
    assert counters == [1, 2, 0, 1, 2]
