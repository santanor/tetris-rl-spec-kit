# Placeholder failing test for replay selection heuristics
import pytest

def test_replay_selection_flags_unimplemented():
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer.select_notable_episodes(None)  # type: ignore[attr-defined]
