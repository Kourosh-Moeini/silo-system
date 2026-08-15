import pytest

from src.systems import create_system

# --- The "which plant" axis. All three built-ins are SISO case studies. ---
BUILTIN_PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

# --- The "what controller" axis: expressed through the GAINS, since the
#     simulate path picks the controller by which gain keys are present. ---
CONTROLLER_GAINS = {
    "P":   {"Kp": 5.0},
    "PI":  {"Kp": 5.0, "Ki": 0.5},
    "PD":  {"Kp": 5.0, "Kd": 1.0},
    "PID": {"Kp": 5.0, "Ki": 0.5, "Kd": 1.0},
}


def _fsf_gains(system_name):
    """Full-state-feedback gains K1..Kn sized to the plant's state dimension.

    The engine selects the FSF control law only when ALL of K1..K{num_states}
    are present (see src/systems.py), so we DERIVE n from the plant rather than
    hard-coding 2 — this keeps the test correct if a plant's order changes.
    """
    n = create_system(system_name).num_states
    base = [5.0, 2.0, 1.0, 0.5]
    return {f"K{i + 1}": (base[i] if i < len(base) else 1.0) for i in range(n)}

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


def _assert_ran_ok(response, record_sim=None):
    """The black-box check: it ran and produced a well-formed scorecard.

    When ``record_sim`` is supplied, the returned metrics are also fed into the
    Phase-3 progress report (pure instrumentation — it never affects pass/fail).
    """
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mse" in data["metrics"]
    assert "stable" in data["metrics"]          # present whether True or False
    assert len(data["trajectory"]) > 0
    if record_sim is not None:
        record_sim(data["metrics"])
    return data


@pytest.mark.parametrize("kind", list(CONTROLLER_GAINS))
def test_controller_variants(client, kind, record_sim):
    """P / PI / PD / PID all run without crashing."""
    body = _base_body(controller_type=kind, gains=CONTROLLER_GAINS[kind])
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_conditions(client, name, record_sim):
    """Different operating conditions all run without crashing."""
    body = _base_body(scenario=SCENARIOS[name])
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)


@pytest.mark.parametrize("dt,max_time", [(0.01, 5.0), (0.005, 5.0), (0.02, 3.0), (0.01, 10.0)])
def test_time_settings(client, dt, max_time, record_sim):
    """Different time steps / horizons all run and produce a trajectory."""
    body = _base_body(dt=dt, max_time=max_time)
    r = client.post("/silo/simulate", json=body)
    _assert_ran_ok(r, record_sim)
    # sanity: a smaller dt or longer max_time should yield more points
    assert len(r.json()["trajectory"]) > 50

# --- Enabled: the controller x scenario interaction matrix (4 x 4 = 16). ---
@pytest.mark.parametrize("kind", list(CONTROLLER_GAINS))
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_controller_x_scenario(client, kind, name, record_sim):
    """Every controller family under every operating condition (16 cases)."""
    body = _base_body(controller_type=kind, gains=CONTROLLER_GAINS[kind], scenario=SCENARIOS[name])
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)


# --- The FSF (full-state feedback) controller axis, per plant. ---
@pytest.mark.parametrize("plant", BUILTIN_PLANTS)
def test_fsf_controller_runs(client, plant, record_sim):
    """Full-state feedback runs on every built-in plant.

    FSF is the controller family the P/PI/PD/PID axis can't reach: it is
    selected purely by the presence of K1..K{num_states} gain keys, so we feed
    plant-sized gains from _fsf_gains().
    """
    body = _base_body(system_name=plant, controller_type="FSF", gains=_fsf_gains(plant))
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)


# --- The gain-range axis: sweep one gain across orders of magnitude. ---
@pytest.mark.parametrize("kp", [0.1, 1.0, 5.0, 20.0, 50.0])
def test_gain_magnitude_sweep(client, kp, record_sim):
    """A P controller stays well-formed across a wide gain range, including
    values above the nominal schema max (20). It must not crash — only perform
    differently."""
    body = _base_body(controller_type="P", gains={"Kp": kp})
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)


# --- Bring PLANTS into the conditions matrix (3 plants x 4 scenarios = 12). ---
@pytest.mark.parametrize("plant", BUILTIN_PLANTS)
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_plants_x_scenarios(client, plant, name, record_sim):
    """Every built-in plant under every operating condition (12 cases)."""
    body = _base_body(system_name=plant, scenario=SCENARIOS[name])
    _assert_ran_ok(client.post("/silo/simulate", json=body), record_sim)