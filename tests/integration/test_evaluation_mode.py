import pytest

def test_evaluation_mode_placeholder():
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer.evaluate_only(None)  # type: ignore[attr-defined]
