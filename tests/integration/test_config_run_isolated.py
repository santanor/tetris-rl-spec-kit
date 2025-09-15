import pytest

def test_run_config_isolation_not_implemented():
    from tetris_rl.agent import trainer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        trainer.generate_session_id({})  # type: ignore[attr-defined]
