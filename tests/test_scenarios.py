import pytest

# --- The "what controller" axis: expressed through the GAINS, since the
#     simulate path picks the controller by which gain keys are present. ---
CONTROLLER_GAINS = {
    "P":   {"Kp": 5.0},
    "PI":  {"Kp": 5.0, "Ki": 0.5},
    "PD":  {"Kp": 5.0, "Kd": 1.0},
    "PID": {"Kp": 5.0, "Ki": 0.5, "Kd": 1.0},
}

# --- The "conditions" axis: different scenarios, easy to hard. ---
SCENARIOS = {
    "calm":       {"initial_condition_range": [-0.2, 0.2], "randomness_level": 0.0, "disturbance_level": 0.0},
    "wide_start": {"initial_condition_range": [-2.0, 2.0], "randomness_level": 0.0, "disturbance_level": 0.0},
    "noisy":      {"initial_condition_range": [-0.5, 0.5], "randomness_level": 0.2, "disturbance_level": 0.0},
    "disturbed":  {"initial_condition_range": [-0.5, 0.5], "randomness_level": 0.0, "disturbance_level": 0.5},
}


def _base_body(**overrides):
    """A default simulate request; override any field via keyword args."""
    body = {
        "system_name": "ball_beam",
        "controller_type": "PID",
        "gains": {"Kp": 5.0, "Ki": 0.1, "Kd": 1.0},
        "scenario": {"initial_condition_range": [-0.5, 0.5], "randomness_level": 0.0, "disturbance_level": 0.0},
        "dt": 0.01,
        "max_time": 5.0,
    }
    body.update(overrides)
    return body


def _assert_ran_ok(response):
    """The black-box check: it ran and produced a well-formed scorecard."""
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mse" in data["metrics"]
    assert "stable" in data["metrics"]          # present whether True or False
    assert len(data["trajectory"]) > 0


@pytest.mark.parametrize("kind", list(CONTROLLER_GAINS))
def test_controller_variants(client, kind):
    """P / PI / PD / PID all run without crashing."""
    body = _base_body(controller_type=kind, gains=CONTROLLER_GAINS[kind])
    _assert_ran_ok(client.post("/silo/simulate", json=body))


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_conditions(client, name):
    """Different operating conditions all run without crashing."""
    body = _base_body(scenario=SCENARIOS[name])
    _assert_ran_ok(client.post("/silo/simulate", json=body))


@pytest.mark.parametrize("dt,max_time", [(0.01, 5.0), (0.005, 5.0), (0.02, 3.0), (0.01, 10.0)])
def test_time_settings(client, dt, max_time):
    """Different time steps / horizons all run and produce a trajectory."""
    body = _base_body(dt=dt, max_time=max_time)
    r = client.post("/silo/simulate", json=body)
    _assert_ran_ok(r)
    # sanity: a smaller dt or longer max_time should yield more points
    assert len(r.json()["trajectory"]) > 50

# fe_oa_4c7091a386df6f39145dc35f45bafcb4ae96daa271eb90ca

# @pytest.mark.parametrize("kind", list(CONTROLLER_GAINS))
# @pytest.mark.parametrize("name", list(SCENARIOS))
# def test_controller_x_scenario(client, kind, name):
#     body = _base_body(controller_type=kind, gains=CONTROLLER_GAINS[kind], scenario=SCENARIOS[name])
#     _assert_ran_ok(client.post("/silo/simulate", json=body))