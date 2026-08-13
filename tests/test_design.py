import time
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_design_job_completes():
    start = client.post("/silo/start", json={
        "config": {"system_name": "ball_beam", "seed": 42, "max_scenarios": 1, "max_iter": 3, "controllers": ["PID"]},
        "control_objective": "stabilize ball position at 0",
    })
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    # Poll until done (with a timeout so a stuck job can't hang forever)
    deadline = time.time() + 30
    status = None
    while time.time() < deadline:
        s = client.get(f"/silo/{job_id}")
        status = s.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert status == "completed"
    assert s.json()["result_summary"] is not None