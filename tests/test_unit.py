"""
tests/test_unit.py

This file covers the core unit tests mandated by the assignment:
1. Metrics Calculation (pure math validation)
2. Parameter Validation (API rejecting bad types)
3. Schema Defaults (API handling missing fields safely)
4. Job Store Lifecycle (State machine of background jobs)
"""

import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn import metrics
from api.main import app
from src.simulation import SimulationRunner

client = TestClient(app)

# ===========================================================================
# 1. METRICS CALCULATION (Pure Math Unit Tests)
# ===========================================================================

EXPECTED_METRICS = [
    "mse", "rmse", "settling_time", "overshoot", "stable",
    "rise_time", "zero_crossings", "control_effort",
    "control_zero_crossings", "ss_error", "stability_margin",
]

class FakeSystem:
    """A lightweight mock system for testing calculate_metrics."""
    def __init__(self, dt=0.1, max_time=1.0, min_control=-10.0, max_control=10.0):
        self.dt = dt
        self.max_time = max_time
        self.min_control = min_control
        self.max_control = max_control

def _make_runner(dt=0.1, max_time=0.2):
    system = FakeSystem(dt=dt, max_time=max_time)
    runner = SimulationRunner(type(system))
    runner.system = system
    return runner

# --- Base Math Tests ---
def test_mse_and_rmse_are_correct():
    runner = _make_runner(dt=0.1, max_time=0.2)
    errors = np.array([3.0, 4.0])
    controls = np.array([1.0, -1.0])
    m = runner.calculate_metrics(errors, controls)

    assert m["mse"] == pytest.approx(12.5)
    assert m["rmse"] == pytest.approx(np.sqrt(12.5))

@pytest.mark.parametrize("key", EXPECTED_METRICS)
def test_expected_keys_exist(key):
    runner = _make_runner(dt=0.1, max_time=0.2)
    m = runner.calculate_metrics(np.array([1.0, 0.5, 0.2]), np.array([0.0, 0.0, 0.0]))
    assert key in m

def test_zero_error_is_perfect():
    runner = _make_runner(dt=0.1, max_time=1.0)
    zeros = np.zeros(11)
    m = runner.calculate_metrics(zeros, zeros)
    assert m["mse"] == pytest.approx(0.0)
    assert bool(m["stable"]) is True


# --- Advanced Parameter Tests ---
def test_zero_crossings_are_counted_correctly():
    """Verify that sign changes in the arrays are correctly tallied as crossings."""
    runner = _make_runner(dt=0.1, max_time=0.4)
    
    # Error crosses 0 twice: 1.0 -> -0.5 (cross 1), -0.5 -> 0.2 (cross 2)
    errors = np.array([1.0, -0.5, 0.2, 0.0])
    # Control crosses 0 once: 1.0 -> 1.0 (no), 1.0 -> -1.0 (cross 1)
    controls = np.array([1.0, 1.0, -1.0, -0.5])

    m = runner.calculate_metrics(errors, controls)

    assert m["zero_crossings"] == 2
    assert m["control_zero_crossings"] == 1

def test_steady_state_error_captures_final_values():
    """Verify that ss_error accurately reflects the error at the end of the simulation."""
    runner = _make_runner(dt=0.1, max_time=0.3)
    # System settles at an error of exactly 0.5
    errors = np.array([5.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    controls = np.zeros_like(errors)

    m = runner.calculate_metrics(errors, controls)
    assert m["ss_error"] == np.float64(0.5)

def test_overshoot_is_detected():
    """If the error goes negative (swings past the 0 target), overshoot must be > 0."""
    runner = _make_runner(dt=0.1, max_time=0.3)
    errors = np.array([1.0, -0.01, 0.2, 1.0])
    controls = np.zeros_like(errors)

    m = runner.calculate_metrics(errors, controls)

    assert m["overshoot"] > 0.0
    # assert m["overshoot"] == pytest.approx(0.8*100.0)

def test_control_effort_scales_with_input():
    """Verify that a larger control array results in a strictly larger control effort."""
    dt = 0.1; max_time = 0.2
    runner = _make_runner(dt=dt, max_time=max_time)
    errors = np.zeros(3)

    controls_low = np.array([1.0, 1.0])
    controls_high = np.array([10.0, 10.0])

    m_low = runner.calculate_metrics(errors, controls_low)
    m_high = runner.calculate_metrics(errors, controls_high)

    # low control effort
    approx_low_cont_effort = np.sum(np.abs(controls_low)) / ((max_time/dt) * 10.0) # the value 10 is defined in the FakeSystem as max_control

    assert m_low["control_effort"] > 0.0
    assert m_low["control_effort"] == np.float64(approx_low_cont_effort)
    assert m_high["control_effort"] > m_low["control_effort"]

def test_rise_and_settling_time_logic():
    """Verify time-based metrics yield sane, positive values for a standard decay."""
    dt = 0.1; max_time = 1.0
    runner = _make_runner(dt=dt, max_time=max_time)
    time = np.arange(0, max_time, dt)

    # A standard smooth decay to zero
    errors = np.array([1.0, 0.8, 0.4, 0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    controls = np.zeros_like(errors)

    m = runner.calculate_metrics(errors, controls)

    # rise time approximatly answer
    rise_threshold = 0.05 * errors[0] if errors[0] > 1e-6 else 0.05
    rise_indices = np.where(np.abs(errors) < rise_threshold)[0]
    approx_rise_time = time[rise_indices[0]] if rise_indices.size > 0 else np.inf

    # settling time approximatly answer
    settling_threshold = 0.05 * errors[0] if errors[0] > 1e-6 else 0.05
    approx_settling_time = time[np.where(np.abs(errors) < settling_threshold)[0][0]] if np.any(np.abs(errors) < settling_threshold) else np.inf

    assert m["rise_time"] >= 0.0
    assert m["settling_time"] > 0.0

    assert m["rise_time"] == np.float64(approx_rise_time)
    assert m["settling_time"] == np.float64(approx_settling_time)

    assert np.isfinite(m["settling_time"])

# ===========================================================================
# 2. PARAMETER VALIDATION (Strict Typing)
# ===========================================================================

def test_validation_rejects_malformed_gains():
    """Sending a string instead of a dictionary for gains should trigger a 422."""
    body = {"system_name": "ball_beam", "gains": "empty"}
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 422

def test_validation_rejects_invalid_types():
    """Sending a string instead of a float for time boundaries should trigger a 422."""
    body = {
        "system_name": "ball_beam",
        "controller_type": "PID",
        "dt": "one tenth",  # Invalid type
        "max_time": 5.0
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 422

# ===========================================================================
# 3. SCHEMA DEFAULTS (Safe Fallbacks)
# ===========================================================================

def test_defaults_handle_missing_gains_safely():
    """Missing gains should default to zeros but still execute successfully (200 OK)."""
    body = {
        "system_name": "ball_beam",
        "controller_type": "PID",
        # "gains" is deliberately omitted
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    assert r.json()["success"] is True

def test_defaults_handle_unknown_system():
    """An unknown system name should gracefully fall back to a default (ball_beam)."""
    body = {
        "system_name": "unknown_plant_123",
        "controller_type": "P",
        "gains": {"Kp": 1.0}
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    assert r.json()["success"] is True

# ===========================================================================
# 4. JOB STORE LIFECYCLE (State Machine)
# ===========================================================================

def test_job_store_rejects_unknown_job():
    """Querying the job store for an ID that doesn't exist should return 404."""
    r = client.get("/silo/non_existent_job_999")
    assert r.status_code == 404

def test_job_store_lifecycle_creation_and_completion():
    """A job should be created, assigned an ID, and eventually complete."""
    # 1. Create Job
    start = client.post("/silo/start", json={
        "config": {"system_name": "ball_beam", "max_iter": 1, "controllers": ["P"]},
        "control_objective": "Test Job Store"
    })
    assert start.status_code == 200
    job_id = start.json()["job_id"]
    assert job_id is not None

    # 2. Poll Job Store (Checking Lifecycle transitions)
    deadline = time.time() + 10
    final_status = None
    
    while time.time() < deadline:
        status_req = client.get(f"/silo/{job_id}")
        assert status_req.status_code == 200
        
        final_status = status_req.json()["status"]
        if final_status in ("completed", "failed"):
            break
        time.sleep(0.1)

    # 3. Verify Job Completion
    assert final_status == "completed"
    
def test_job_store_cancellation():
    """Canceling a job updates its lifecycle state in the store."""
    start = client.post("/silo/start", json={
        "config": {"system_name": "ball_beam", "max_iter": 5, "controllers": ["PID"]},
        "control_objective": "Cancel Me"
    })
    job_id = start.json()["job_id"]

    # Depending on how fast the mock runs, it might succeed (200) or already be done (400)
    cancel_req = client.post(f"/silo/{job_id}/cancel")
    assert cancel_req.status_code in (200, 400)