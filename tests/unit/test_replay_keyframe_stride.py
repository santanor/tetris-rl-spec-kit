import pytest

from tetris_rl.replay import store


def test_replay_keyframe_stride_basic():
    assert store.compute_keyframe_indices(0, 5) == []
    assert store.compute_keyframe_indices(1, 5) == [0]
    assert store.compute_keyframe_indices(5, 2) == [0, 2, 4]
    # last frame appended if not aligned
    assert store.compute_keyframe_indices(6, 4) == [0, 4, 5]


def test_replay_keyframe_stride_invalid():
    with pytest.raises(ValueError):
        store.compute_keyframe_indices(10, 0)
