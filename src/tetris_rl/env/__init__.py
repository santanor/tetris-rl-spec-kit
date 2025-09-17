"""Environment related modules for tetris_rl."""

try:  # noqa: SIM105
    """Environment package.

    Only the minimal `TetrisEnv` is used in the streamlined demo. The previous
    `DummyTetrisEnv` and auxiliary wrappers have been retained only if other
    legacy scripts import them, but new code should rely on `TetrisEnv`.
    """

    # from .dummy_env import DummyTetrisEnv  # (deprecated)
except Exception:  # pragma: no cover
    pass

try:  # noqa: SIM105
    from .tetris_env import TetrisEnv  # noqa: F401
except Exception:  # pragma: no cover
    pass
