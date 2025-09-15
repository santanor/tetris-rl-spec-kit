import pytest

# Intent: This will fail until `tetris_rl.agent.trainer` exposes an `initialize_session` function.


def test_notebook_initialization_seed_echo():
    from tetris_rl import agent  # noqa: F401  # ensure package importable
    
    with pytest.raises(AttributeError):
        # initialize_session should eventually return (config, session_id)
        from tetris_rl.agent import trainer  # noqa
        trainer.initialize_session(config=None)  # type: ignore[attr-defined]
