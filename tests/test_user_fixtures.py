import json
import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.systems import CustomDynamicalSystem

client = TestClient(app)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "user_dynamics")
REGISTRY_PATH = os.path.join(FIXTURE_DIR, "registry.json")


def load_registry():
    """Read the JSON registry to discover user-provided models."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


user_fixtures = load_registry()


@pytest.mark.parametrize("fixture", user_fixtures, ids=lambda f: f["name"])
def test_user_provided_dynamics(fixture):
    """Smoke-test each user-uploaded model via its registered smoke gains."""
    custom_path = os.path.join("tests", "fixtures", "user_dynamics", fixture["path"])
    body = {
        "system_name": "custom",
        "custom_dynamics_path": custom_path,
        "controller_type": "PID",
        "gains": fixture.get("smoke_gains", {}),
        "scenario": fixture.get("default_scenario", {}),
        "dt": 0.01,
        "max_time": 5.0,
    }
    response = client.post("/silo/simulate", json=body)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True, f"Simulation failed for {fixture['name']}: {data.get('message')}"
    assert "mse" in data["metrics"]
    assert "stable" in data["metrics"]
    assert len(data["trajectory"]) > 0


@pytest.mark.parametrize("fixture", user_fixtures, ids=lambda f: f["name"])
def test_user_dynamics_num_states(fixture):
    """The engine-detected state dimension must match the registry's declared
    `num_states` (skipped when not declared)."""
    if "num_states" not in fixture:
        pytest.skip(f"{fixture['name']}: no expected num_states declared in registry")

    abs_path = os.path.join(FIXTURE_DIR, fixture["path"])
    system = CustomDynamicalSystem(abs_path, fixture.get("default_scenario"))

    assert system.num_states == fixture["num_states"], (
        f"{fixture['name']}: registry declares num_states={fixture['num_states']} "
        f"but the engine detected {system.num_states} from {fixture['path']}"
    )