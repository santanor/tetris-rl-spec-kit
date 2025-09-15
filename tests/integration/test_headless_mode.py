import os
import pytest

def test_headless_mode_placeholder(temp_runs_dir, monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer.train(episodes=1, config=None, run_dir=temp_runs_dir)  # type: ignore[attr-defined]
