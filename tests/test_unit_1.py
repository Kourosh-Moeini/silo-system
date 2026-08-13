from fastapi.testclient import TestClient
from api.main import app

def test_health_ok(client):
    # client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_case_studies_lists_plants(client):
    # client = TestClient(app)
    r = client.get("/case-studies")
    assert r.status_code == 200
    names = [item["name"] for item in r.json()]
    assert "BallBeam" in names   # a known bundled plant