"""
tests/test_scenarios.py

Scenario / user-input VARIABILITY matrix on the DESIGN path (run_optimization).

MATRIX — 36 design runs
-----------------------
  3 plants     : ball_beam, dc_motor, inverted_pendulum
  3 scenarios  : easy / mid / hard, given as
                 (initial position, disturbance_level, randomness_level)
                     easy -> (0.5, 0.0, 0.0)
                     mid  -> (0.5, 0.5, 0.5)
                     hard -> (0.5, 1.0, 1.0)
                 The initial position is fixed at 0.5 and the target is 0.0, so
                 every run starts from the SAME initial error (0.5); the
                 scenarios differ only in disturbance and measurement noise.
  4 controllers: P, PI, PID, FSF
  1 gain range : [0, 100] applied to every gain of every controller

Controller families are realised by handing the mock agent matching gain KEYS as
param_ranges (P->Kp, PI->Kp/Ki, PID->Kp/Ki/Kd, FSF->K1..Kn); the engine picks the
control law from those keys, so P/PI/PID/FSF are all genuinely exercised (not
silently collapsed to FSF).

dt = 0.01 (the engine/API default). The earlier dt = 0.1 made the stiff DC-motor
loop diverge under forward Euler (mse ~1e89), which would drive every dc_motor
success rate to exactly 0 and hide the real trend.

WHAT IS RECORDED
----------------
The LAST-ITERATION outcome of every run is appended to a FLAT list at
reports/latest_results/test_scenario_report.json -> "runs". Each record holds
only RAW facts: plant, scenario parameters, controller, the gains actually used,
initial_error, and the metrics dict.

Derived quantities — plant complexity, parameter complexity, total complexity and
the success rate — are deliberately NOT computed here. They are produced by
`reports/make_report.py`, which turns this JSON into reports/report.md and the
plots.

This is a recording/characterization test: it asserts that every combination ran
and produced well-formed metrics, then saves the values.
"""

import math
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from src.controllers_mock import run_optimization
from src.systems import create_system

REPORT_NAME = "test_scenario_report"


# --- Matrix axes -----------------------------------------------------------

PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

# name -> (initial position, disturbance_level, randomness_level)
SCENARIO_SPECS = {
    "easy": (0.5, 0.0, 0.0),
    "mid":  (0.5, 0.5, 0.5),
    "hard": (0.5, 1.0, 1.0),
}

CONTROLLERS = ["P", "PI", "PID", "FSF"]

# One gain sampling range, used for every gain of every controller.
GAIN_RANGE = [0.0, 100.0]

# PID-family gain keys per controller; FSF is sized to the plant.
PID_KEYS = {"P": ["Kp"], "PI": ["Kp", "Ki"], "PID": ["Kp", "Ki", "Kd"]}

TARGET = 0.0
SEED = 42
MAX_ITER = 5
MAX_SCENARIOS = 1
DT = 0.01
MAX_TIME = 5.0


def _scenario(name: str) -> dict:
    """Expand a (position, disturbance, randomness) spec into an engine scenario."""
    pos, disturbance, randomness = SCENARIO_SPECS[name]
    return {
        "id": name,
        "initial_condition_range": [pos, pos],   # low == high -> a fixed IC
        "randomness_level": randomness,
        "disturbance_level": disturbance,
        "param_uncertainty": 0.0,
    }


SCENARIOS = {name: _scenario(name) for name in SCENARIO_SPECS}


# --- Helpers ---------------------------------------------------------------

def _json_safe(metrics: dict) -> dict:
    """Metrics -> plain JSON types; non-finite (e.g. inf) -> None."""
    safe = {}
    for key, val in metrics.items():
        if isinstance(val, (bool, np.bool_)):
            safe[key] = bool(val)
        elif isinstance(val, (int, float, np.integer, np.floating)):
            f = float(val)
            safe[key] = f if math.isfinite(f) else None
        else:
            safe[key] = val
    return safe


def _param_ranges(controller: str, gain_range: list, num_states: int) -> dict:
    """Build {controller: {gain_key: range}} so the mock agent proposes the gain
    KEYS that select the requested control law."""
    if controller == "FSF":
        keys = [f"K{i + 1}" for i in range(num_states)]
    else:
        keys = PID_KEYS[controller]
    return {controller: {k: list(gain_range) for k in keys}}


def _last_iteration_entry(system_name, scenario, controller, param_ranges):
    """Run one full mock design job; return the LAST history entry of the last
    completed scenario (params + metrics), or None."""
    with redirect_stdout(StringIO()):
        result = run_optimization(
            llm_model="mock",
            run_id=1,
            seed=SEED,
            system_name=system_name,
            max_scenarios=MAX_SCENARIOS,
            max_iter=MAX_ITER,
            controllers=[controller],
            custom_scenarios=[scenario],
            param_ranges=param_ranges,
            dt=DT,
            max_time=MAX_TIME,
            target=TARGET,
        )
    scen_hist = result.get("all_scenario_history", [])
    if not scen_hist:
        return None
    history = scen_hist[-1].get("history", [])
    if not history:
        return None
    return history[-1], len(history)


# --- The matrix test -------------------------------------------------------

def test_scenario_matrix(record_report):
    runs = []

    for plant in PLANTS:
        system = create_system(plant)
        num_states = system.num_states
        num_actions = getattr(system, "num_inputs", 1)

        for sname, scenario in SCENARIOS.items():
            lo, hi = scenario["initial_condition_range"]
            initial_position = (lo + hi) / 2.0          # lo == hi -> exact
            initial_error = abs(TARGET - initial_position)

            for controller in CONTROLLERS:
                pr = _param_ranges(controller, GAIN_RANGE, num_states)
                outcome = _last_iteration_entry(plant, scenario, controller, pr)

                assert outcome is not None, (
                    f"{plant}/{sname}/{controller}: design produced no history"
                )
                entry, n_iterations = outcome
                metrics = _json_safe(entry.get("metrics", {}))

                assert "mse" in metrics, (
                    f"{plant}/{sname}/{controller}: metrics missing mse"
                )
                assert "ss_error" in metrics, (
                    f"{plant}/{sname}/{controller}: metrics missing ss_error"
                )

                gains = {
                    k: v for k, v in (entry.get("params") or {}).items()
                    if k != "reasoning"
                }

                runs.append({
                    "plant": plant,
                    "scenario": sname,
                    "controller": controller,
                    "gain_range": list(GAIN_RANGE),
                    "gains": gains,
                    "num_states": int(num_states),
                    "num_actions": int(num_actions),
                    "initial_position": initial_position,
                    "target": TARGET,
                    "initial_error": initial_error,
                    "randomness_level": scenario["randomness_level"],
                    "disturbance_level": scenario["disturbance_level"],
                    "iterations": n_iterations,
                    "metrics": metrics,
                })

    assert len(runs) == len(PLANTS) * len(SCENARIOS) * len(CONTROLLERS), (
        f"expected {len(PLANTS) * len(SCENARIOS) * len(CONTROLLERS)} runs, "
        f"recorded {len(runs)}"
    )

    record_report(REPORT_NAME, {
        "title": "Scenario / input variability matrix",
        "description": (
            "Design-path sweep over every built-in plant x 3 scenarios "
            "(easy/mid/hard, given as initial position / disturbance_level / "
            "randomness_level = 0.5/0.0/0.0, 0.5/0.5/0.5, 0.5/1.0/1.0) x 4 "
            "controllers (P, PI, PID, FSF) with a single gain range [0, 100] = "
            "36 runs. Each record is the last-iteration outcome of one "
            "run_optimization run. Controller families are selected via matching "
            "param_ranges keys (P->Kp, PI->Kp/Ki, PID->Kp/Ki/Kd, FSF->K1..Kn). "
            "The target is 0.0 and the initial position is fixed at 0.5, so "
            "initial_error is 0.5 for every run. Derived complexity and success "
            "rate are computed by reports/make_report.py."
        ),
        "config": {
            "plants": PLANTS,
            "scenario_specs": {
                name: {
                    "initial_position": spec[0],
                    "disturbance_level": spec[1],
                    "randomness_level": spec[2],
                }
                for name, spec in SCENARIO_SPECS.items()
            },
            "controllers": CONTROLLERS,
            "gain_range": GAIN_RANGE,
            "target": TARGET,
            "seed": SEED,
            "max_iter": MAX_ITER,
            "max_scenarios": MAX_SCENARIOS,
            "dt": DT,
            "max_time": MAX_TIME,
        },
        "runs": runs,
    })
