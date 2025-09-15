"""Environment related modules for tetris_rl."""

try:  # noqa: SIM105
    from .dummy_env import DummyTetrisEnv  # noqa: F401
except Exception:  # pragma: no cover
    pass

try:  # noqa: SIM105
    from .tetris_env import TetrisEnv  # noqa: F401
except Exception:  # pragma: no cover
    pass
