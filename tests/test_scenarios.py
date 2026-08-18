"""
tests/test_scenarios.py

Scenario / user-input VARIABILITY matrix on the DESIGN path (run_optimization).

For every built-in plant, sweeps the full combination of:
  * 3 scenarios  — easy / medium / hard (initial_condition_range,
                   randomness_level, disturbance_level),
  * 4 controllers — P, PI, PID, FSF,
  * 3 gain ranges — [0, 1], [1, 10], [-5, 0],

i.e. 3 plants x 3 scenarios x 4 controllers x 3 ranges = 108 combinations, and
records each combination's LAST-ITERATION metrics into
reports/latest_report.json under the "test_scenarios" key (raw, nested as
plant -> scenario -> controller -> gain range).

Controller families are realised by handing the mock agent matching gain KEYS
as param_ranges (P->Kp, PI->Kp/Ki, PID->Kp/Ki/Kd, FSF->K1..Kn); the engine then
picks the control law from those keys, so P/PI/PID/FSF are all genuinely
exercised (not silently collapsed to FSF).

Recording/characterization test: it asserts only that every combination ran and
produced well-formed metrics, then saves the values.
"""

import math
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from src.controllers_mock import run_optimization
from src.systems import create_system


# --- Matrix axes -----------------------------------------------------------

PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

SCENARIOS = {
    "easy":   {"id": "easy",   "initial_condition_range": [0.1, 0.1],
               "randomness_level": 0.1, "disturbance_level": 0.05, "param_uncertainty": 0.0},
    "medium": {"id": "medium", "initial_condition_range": [0.5, 0.5],
               "randomness_level": 0.5, "disturbance_level": 0.2,  "param_uncertainty": 0.0},
    "hard":   {"id": "hard",   "initial_condition_range": [1.0, 1.0],
               "randomness_level": 1.0, "disturbance_level": 1.0,  "param_uncertainty": 0.0},
}

CONTROLLERS = ["P", "PI", "PID", "FSF"]

# Gain sampling ranges handed to the mock agent (per gain).
GAIN_RANGES = [[0.0, 1.0], [1.0, 10.0], [-5.0, 0.0]]

# PID-family gain keys per controller; FSF is sized to the plant.
PID_KEYS = {"P": ["Kp"], "PI": ["Kp", "Ki"], "PID": ["Kp", "Ki", "Kd"]}

SEED = 42
MAX_ITER = 5
DT = 0.1
MAX_TIME = 5.0


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
    """Build {controller: {gain_key: range}} so the mock agent proposes the
    gain KEYS that select the requested control law."""
    if controller == "FSF":
        keys = [f"K{i + 1}" for i in range(num_states)]
    else:
        keys = PID_KEYS[controller]
    return {controller: {k: list(gain_range) for k in keys}}


def _last_iteration_metrics(system_name, scenario, controller, param_ranges):
    """Run one full mock design job; return JSON-safe metrics of the LAST
    iteration of the last completed scenario (or None)."""
    with redirect_stdout(StringIO()):
        result = run_optimization(
            llm_model="mock",
            run_id=1,
            seed=SEED,
            system_name=system_name,
            max_scenarios=1,
            max_iter=MAX_ITER,
            controllers=[controller],
            custom_scenarios=[scenario],
            param_ranges=param_ranges,
            dt=DT,
            max_time=MAX_TIME,
        )
    scen_hist = result.get("all_scenario_history", [])
    if not scen_hist:
        return None
    history = scen_hist[-1].get("history", [])
    if not history:
        return None
    return _json_safe(history[-1].get("metrics", {}))


# --- The matrix test -------------------------------------------------------

def test_scenario_matrix(record_report_section):
    plants_block = {}
    combos = 0
    with_finite_mse = 0

    for plant in PLANTS:
        num_states = create_system(plant).num_states
        scen_block = {}
        for sname, scenario in SCENARIOS.items():
            ctrl_block = {}
            for controller in CONTROLLERS:
                range_block = {}
                for gain_range in GAIN_RANGES:
                    pr = _param_ranges(controller, gain_range, num_states)
                    metrics = _last_iteration_metrics(plant, scenario, controller, pr)
                    assert metrics is not None, (
                        f"{plant}/{sname}/{controller}/gains{gain_range}: "
                        "design produced no history"
                    )
                    assert "mse" in metrics, (
                        f"{plant}/{sname}/{controller}/gains{gain_range}: "
                        "metrics missing mse"
                    )
                    combos += 1
                    if isinstance(metrics.get("mse"), (int, float)):
                        with_finite_mse += 1
                    range_block[f"gains[{gain_range[0]},{gain_range[1]}]"] = metrics
                ctrl_block[controller] = range_block
            scen_block[sname] = ctrl_block
        plants_block[plant] = scen_block

    # assert combos == len(PLANTS) * len(SCENARIOS) * len(CONTROLLERS) * len(GAIN_RANGES)
    # assert with_finite_mse > 0, "no combination produced a finite mse"

    record_report_section("test_scenarios", {
        "title": "Scenario / input variability matrix",
        "description": (
            "Design-path sweep over every built-in plant x 3 scenarios "
            "(easy/medium/hard) x 4 controllers (P, PI, PID, FSF) x 3 gain "
            "ranges ([0,1], [1,10], [-5,0]) = 108 combinations. Each cell holds "
            "the last-iteration metrics of one run_optimization run. Controller "
            "families are selected via matching param_ranges keys "
            "(P->Kp, PI->Kp/Ki, PID->Kp/Ki/Kd, FSF->K1..Kn). Seed fixed at 42; "
            "the mock agent samples gains within each range."
        ),
        "config": {
            "scenarios": SCENARIOS,
            "controllers": CONTROLLERS,
            "gain_ranges": GAIN_RANGES,
            "seed": SEED,
            "max_iter": MAX_ITER,
            "max_scenarios": 1,
            "dt": DT,
            "max_time": MAX_TIME,
        },
        "plants": plants_block,
    })