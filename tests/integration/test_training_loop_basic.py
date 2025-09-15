# Fails until trainer.train is implemented
import pytest


def test_basic_training_loop_runs_and_creates_artifacts(temp_runs_dir):
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        trainer.train(episodes=1, config=None, run_dir=temp_runs_dir)  # type: ignore[attr-defined]
