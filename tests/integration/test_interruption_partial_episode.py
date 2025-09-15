import pytest

def test_interruption_partial_episode_placeholder():
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer.simulate_interrupt()  # type: ignore[attr-defined]
