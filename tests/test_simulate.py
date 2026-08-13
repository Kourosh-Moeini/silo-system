import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.mark.parametrize("system_name", ["ball_beam", "dc_motor", "inverted_pendulum"])
def test_simulate_builtin_plants(system_name, record_property):
    body = {
        "system_name": system_name,
        "controller_type": "PID",
        "gains": {"Kp": 5.0, "Ki": 0.1, "Kd": 1.0},
        "scenario": {"initial_condition_range": [-0.5, 0.5], "randomness_level": 0.0, "disturbance_level": 0.0},
        "dt": 0.01,
        "max_time": 5.0,
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "mse" in data["metrics"]
    assert len(data["trajectory"]) > 0

# Attach metrics to the Pytest report
    record_property("mse", data["metrics"].get("mse", 0.0))
    record_property("settling_time", data["metrics"].get("settling_time", 0.0))
    record_property("stable", data["metrics"].get("stable", False))