import json
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError

SCHEMA_PATH = Path("specs/001-a-demo-that/contracts/metrics-schema.json")

@pytest.fixture(scope="module")
def metrics_schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_metrics_export_schema_missing_file(temp_runs_dir, metrics_schema):
    """Expect validation failure because metrics.json not yet produced."""
    export_path = Path(temp_runs_dir) / "metrics.json"
    assert not export_path.exists(), "metrics.json should not exist before implementation"
    # Simulate an empty stub to show failure
    with pytest.raises(ValidationError):
        validate(instance={}, schema=metrics_schema)


def test_metrics_export_schema_future_valid_example(metrics_schema):
    """Provide a minimal (still invalid now) skeleton to clarify required fields.
    This test will be updated once exporter exists.
    """
    minimal = {
        "session": {
            "session_id": "placeholder",
            "mode": "train",
            "seed": 42,
            "config": {},
            "summary": {
                "episodes_total": 0,
                "avg_reward": 0.0,
                "avg_lines_cleared": 0.0,
            },
        },
        "episodes": [],
        "frames_sample": [],
    }
    # Should pass schema shape now (if schema stable) – keep as expectation marker.
    validate(instance=minimal, schema=metrics_schema)
