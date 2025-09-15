import pytest

from tetris_rl.env import observation_wrappers as ow


def test_observation_mode_registry_shapes():
    builder = ow.get_observation_mode("dummy")
    assert callable(builder)
    assert builder(123) == 123  # identity lambda
    with pytest.raises(NotImplementedError):
        ow.get_observation_mode("missing")
