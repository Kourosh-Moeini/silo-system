import statistics
from fastapi.testclient import TestClient
from api.main import app
import numpy as np

client = TestClient(app)

def run_once(seed):
    np.random.seed(seed)          # <-- NOW the seed matters: it fixes the randomness for this run
    body = {"system_name": "ball_beam", "controller_type": "PID",
            "gains": {"Kp": 5.0, "Ki": 0.1, "Kd": 1.0},
            "scenario": {"initial_condition_range": [-0.5, 0.5], "randomness_level": 0.1, "disturbance_level": 0.0},
            "dt": 0.01, "max_time": 5.0}
    return client.post("/silo/simulate", json=body).json()["metrics"]["mse"]

def test_variance_across_seeds():
    values = [run_once(s) for s in range(5)]
    assert all(v >= 0 for v in values)          # mse is never negative
    spread = max(values) - min(values)
    assert spread < 1000                         # sane upper bound; tune after observing