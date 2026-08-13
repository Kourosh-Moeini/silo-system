"""
tests/test_failures.py
 
Failure-mode tests: deliberately feed the system bad or unusual input and check
that it behaves sanely.
 
IMPORTANT LESSON baked into this file:
We actually ran the engine to see what it does, instead of assuming. Two cases
that you'd *expect* to fail actually DON'T:
 
  * an unknown system_name  -> the engine silently falls back to "ball_beam"
  * empty / missing gains   -> it runs a do-nothing controller (all gains = 0)
 
So this file has two kinds of tests:
  1) CLEAN FAILURES     - things that should (and do) fail gracefully.
  2) DOCUMENTED QUIRKS  - surprising things that DON'T fail. We pin down the
                          current behavior so that if someone changes it later,
                          the test flags it. (This is regression testing.)
 
All tests use the `client` fixture defined in tests/conftest.py, so each test
just lists `client` as a parameter and pytest supplies the pretend-browser.
"""
 
import time
import pytest
 
 
# ---------------------------------------------------------------------------
# 1) CLEAN FAILURES — the system should reject these gracefully, not crash.
# ---------------------------------------------------------------------------
 
def test_bad_custom_path_reports_failure(client):
    """A custom dynamics file that doesn't exist should fail cleanly.
 
    The endpoint still returns HTTP 200 (the request itself was well-formed),
    but the *payload* reports success=False with an explanatory message.
    This is the ONE simulate case that genuinely fails.
    """
    body = {
        "system_name": "custom",
        "custom_dynamics_path": "this_file_does_not_exist.py",
        "controller_type": "PID",
        "gains": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200          # the HTTP call succeeded...
    data = r.json()
    assert data["success"] is False      # ...but the simulation itself failed
    assert data["message"]               # and it told us why (non-empty message)

def test_custom_without_path_is_rejected(client):
    """Using system_name='custom' but giving no file path fails with a clear message."""
    r = client.post("/silo/simulate", json={"system_name": "custom", "gains": {"Kp": 1.0}})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is False
    assert "custom_dynamics_path" in (data["message"] or "")
 
def test_status_of_unknown_job_is_404(client):
    """Asking about a job id that was never created should return 404 Not Found."""
    r = client.get("/silo/this-job-id-does-not-exist")
    assert r.status_code == 404
 
 
def test_cancel_unknown_job_is_404(client):
    """Cancelling a job that doesn't exist should return 404 Not Found."""
    r = client.post("/silo/this-job-id-does-not-exist/cancel")
    assert r.status_code == 404
 
 
def test_malformed_body_is_rejected(client):
    """Sending the wrong TYPE for a field should be rejected by validation (422).
 
    `gains` must be an object like {"Kp": 1.0}. Here we send a plain string,
    so FastAPI/Pydantic rejects the request before it ever reaches the engine.
    422 = "Unprocessable Entity" = "your request body is malformed".
    """
    body = {"system_name": "ball_beam", "gains": "not-a-dictionary"}
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 422
 
 
# ---------------------------------------------------------------------------
# 2) DOCUMENTED QUIRKS — these DON'T fail, even though you might expect them to.
#    We assert the *current* behavior on purpose, as a regression guard.
# ---------------------------------------------------------------------------
 
def test_unknown_system_silently_falls_back(client):
    """QUIRK: an unknown system_name does NOT error — it falls back to ball_beam.
 
    See create_system() in src/systems.py: if the name matches no known plant,
    it resolves to ball_beam instead of raising. We document that here so the
    behavior is visible and a future change would trip this test.
    """
    body = {
        "system_name": "totally_made_up_plant",
        "controller_type": "PID",
        "gains": {"Kp": 5.0, "Ki": 0.1, "Kd": 1.0},
        "scenario": {"initial_condition_range": [-0.5, 0.5],
                     "randomness_level": 0.0, "disturbance_level": 0.0},
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True        # it ran (via the ball_beam fallback)
    assert "mse" in data["metrics"]
 
 
def test_missing_gains_still_runs(client):
    """QUIRK: no gains supplied does NOT error — it runs with all gains = 0.
 
    An empty/absent gains dict is treated as a PID with Kp=Ki=Kd=0 (a controller
    that does nothing). It runs to completion; the result is just poor.
    """
    body = {
        "system_name": "ball_beam",
        "controller_type": "PID",
        # no "gains" key at all
        "scenario": {"initial_condition_range": [-0.5, 0.5],
                     "randomness_level": 0.0, "disturbance_level": 0.0},
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True        # it still ran...
    assert "stable" in data["metrics"]    # ...and produced a scorecard (quality aside)
 
 
# ---------------------------------------------------------------------------
# 3) CANCELLING A JOB — inherently racy because the mock job finishes fast.
# ---------------------------------------------------------------------------
 
def test_cancel_running_job_is_accepted_or_already_done(client):
    """Cancel a real job. Because the mock job completes very quickly, by the
    time we send the cancel it may already be finished. Both outcomes are valid:
 
      * 200  -> cancel was accepted while the job was still pending/running
      * 400  -> job had already completed/failed, so it can't be cancelled
 
    We accept either, rather than writing a flaky test that assumes timing.
    (Worth noting this timing quirk in your README.)
    """
    start = client.post("/silo/start", json={
        "config": {"system_name": "ball_beam", "seed": 42,
                   "max_scenarios": 1, "max_iter": 3, "controllers": ["PID"]},
        "control_objective": "stabilize ball position at 0",
    })
    assert start.status_code == 200
    job_id = start.json()["job_id"]
 
    r = client.post(f"/silo/{job_id}/cancel")
    assert r.status_code in (200, 400)