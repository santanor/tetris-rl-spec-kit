import pytest

def test_warning_heuristics_placeholder():
    from tetris_rl.metrics import warnings as warnings_mod  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        warnings_mod.evaluate_warnings(None)  # type: ignore[attr-defined]
