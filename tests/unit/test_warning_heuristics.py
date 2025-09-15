import pytest
from tetris_rl.metrics import warnings


def test_warning_heuristics_module():
    result = warnings.evaluate_warnings(session=None)
    assert result == []
