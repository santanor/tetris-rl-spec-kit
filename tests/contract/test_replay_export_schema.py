import json
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError

SCHEMA_PATH = Path("specs/001-a-demo-that/contracts/replay-schema.json")

@pytest.fixture(scope="module")
def replay_schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_replay_export_schema_missing_file(temp_runs_dir, replay_schema):
    export_path = Path(temp_runs_dir) / "replay_0.json"
    assert not export_path.exists(), "replay export should not exist pre-implementation"
    with pytest.raises(ValidationError):
        validate(instance={}, schema=replay_schema)


def test_replay_export_schema_future_valid_example(replay_schema):
    minimal = {
        "session_id": "placeholder",
        "episode_index": 0,
        "config_hash": "abc123",
        "frames": [],
    }
    validate(instance=minimal, schema=replay_schema)
