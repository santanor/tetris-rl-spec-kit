import os
import tempfile
import pytest

@pytest.fixture()
def temp_runs_dir(monkeypatch):
    d = tempfile.mkdtemp(prefix="tetris_rl_runs_")
    monkeypatch.setenv("TETRIS_RL_RUNS_DIR", d)
    return d
