import pytest

def test_stagnation_early_stop_placeholder():
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer._stagnation_check(None)  # type: ignore[attr-defined]
