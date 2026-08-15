import json
import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
from src.systems import CustomDynamicalSystem

client = TestClient(app)

# Resolve the absolute path to the registry file
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "user_dynamics")
REGISTRY_PATH = os.path.join(FIXTURE_DIR, "registry.json")

def load_registry():
    """Reads the JSON registry to discover user-provided models."""
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Load fixtures at module import time so Pytest can parameterize them
user_fixtures = load_registry()

@pytest.mark.parametrize("fixture", user_fixtures, ids=lambda f: f["name"])
def test_user_provided_dynamics(fixture, record_sim):
    """
    Dynamically tests user-uploaded models using their registered smoke gains.
    """
    # The API expects a path it can resolve. 
    # Relative to the repo root is generally safest for custom_dynamics_path.
    custom_path = os.path.join("tests", "fixtures", "user_dynamics", fixture["path"])
    
    body = {
        "system_name": "custom",
        "custom_dynamics_path": custom_path,
        "controller_type": "PID",  # Treat missing gains as 0.0
        "gains": fixture.get("smoke_gains", {}),
        "scenario": fixture.get("default_scenario", {}),
        "dt": 0.01,
        "max_time": 5.0,
    }
    
    response = client.post("/silo/simulate", json=body)
    
    # 1. Check HTTP Success
    assert response.status_code == 200
    
    data = response.json()
    
    # 2. Check Simulation Success
    assert data["success"] is True, f"Simulation failed for {fixture['name']}: {data.get('message')}"
    
    # 3. Validate Metrics were calculated
    assert "mse" in data["metrics"]
    assert "stable" in data["metrics"]
    assert len(data["trajectory"]) > 0

    # Feed the fixture's metrics into the Phase-3 progress report.
    record_sim(data["metrics"])


@pytest.mark.parametrize("fixture", user_fixtures, ids=lambda f: f["name"])
def test_user_dynamics_num_states(fixture):
    """The plant's ACTUAL state dimension must match the registry's declared
    `num_states`.

    This turns the registry field from documentation into a checked contract:
    the engine auto-detects the state dimension from the dynamics function
    (src/systems.py::_detect_system_properties), so a mis-declared dimension —
    or a file whose shape drifts over time — fails CI here instead of silently
    passing the smoke test above.

    `num_states` is optional in the registry ("if known"), so we skip when it
    isn't declared rather than fail.
    """
    if "num_states" not in fixture:
        pytest.skip(f"{fixture['name']}: no expected num_states declared in registry")

    # Absolute path (independent of CWD) so detection works wherever pytest runs.
    abs_path = os.path.join(FIXTURE_DIR, fixture["path"])

    # Construct exactly what the simulate path constructs, and read the
    # dimension the engine itself detected.
    system = CustomDynamicalSystem(abs_path, fixture.get("default_scenario"))

    assert system.num_states == fixture["num_states"], (
        f"{fixture['name']}: registry declares num_states={fixture['num_states']} "
        f"but the engine detected {system.num_states} from {fixture['path']}"
    )