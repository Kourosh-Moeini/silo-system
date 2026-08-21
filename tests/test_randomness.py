"""
tests/test_randomness.py

Randomness CHARACTERIZATION on the DESIGN path (run_optimization). Both tests
drive the full mock design pipeline and save their raw results into
reports/test_randomness_report.json:

  * test_randomness_same_config_diff_seeds
        Fixed "medium" scenario, PID, 5 seeds, 3 plants. Records the
        LAST-ITERATION metrics of every run (plant -> seed) and aggregates over
        all plants x seeds.

  * test_randomness_same_seed_repeated_runs
        Same config, a SINGLE seed, run N times per plant, but only the LAST
        run's metrics are kept per plant. Aggregates over the plants.

WHY RECORDS, NOT EQUALITY ASSERTIONS
------------------------------------
The design `seed` reseeds only np.random (initial condition + process noise) via
set_global_seed(); the mock agents draw their gains from Python's `random`, which
is seeded once at import and never per job. So different seeds vary, and even the
same seed repeated varies. We assert only that every run produced well-formed
metrics, then save the values.

Only raw facts are recorded (metrics, mean/variance, initial_error). The success
rate is derived in reports/make_report.py so that the scenario and randomness
sections of the report share one definition.
"""

import math
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from src.controllers_mock import run_optimization

REPORT_NAME = "test_randomness_report"

# --- Fixed "medium" scenario + shared design config ------------------------

PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

MEDIUM_SCENARIO = {
    "id": "medium",
    "initial_condition_range": [0.5, 0.5],   # low == high -> a fixed IC
    "randomness_level": 0.5,
    "disturbance_level": 0.2,
    "param_uncertainty": 0.0,
}

TARGET = 0.0
DIFF_SEEDS = [1, 7, 42, 100, 2026]
REPEAT_SEED = 42
N_REPEATS = 5
CONTROLLERS = ["PID"]
MAX_ITER = 5
MAX_SCENARIOS = 1
DT = 0.01
MAX_TIME = 5.0

# Every run starts at 0.5 with target 0.0, so the initial error is the same.
INITIAL_ERROR = abs(TARGET - MEDIUM_SCENARIO["initial_condition_range"][0])


# --- Helpers ---------------------------------------------------------------

def _json_safe(metrics: dict) -> dict:
    """Metrics -> plain JSON types; non-finite (e.g. inf) -> None."""
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, np.bool_)):
            safe[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            safe[key] = number if math.isfinite(number) else None
        else:
            safe[key] = value
    return safe


def _last_iteration_metrics(system_name: str, seed: int):
    """Run one full mock design job; return the JSON-safe metrics of the last
    iteration of the last completed scenario (or None)."""
    with redirect_stdout(StringIO()):
        result = run_optimization(
            llm_model="mock",
            run_id=1,
            seed=seed,
            system_name=system_name,
            max_scenarios=MAX_SCENARIOS,
            max_iter=MAX_ITER,
            controllers=CONTROLLERS,
            custom_scenarios=[MEDIUM_SCENARIO],
            dt=DT,
            max_time=MAX_TIME,
            target=TARGET,
        )
    scenario_history = result.get("all_scenario_history", [])
    if not scenario_history:
        return None
    history = scenario_history[-1].get("history", [])
    if not history:
        return None
    return _json_safe(history[-1].get("metrics", {}))


def _finite_values(metrics_list: list, key: str) -> list:
    """All finite numeric values of `key` across a list of metrics dicts."""
    return [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]


def _aggregate(metrics_list: list) -> dict:
    """Mean/variance of mse, rmse and ss_error over a list of metrics dicts.
    Non-finite entries are skipped."""
    def stats(values):
        if not values:
            return None, None
        return round(float(np.mean(values)), 6), round(float(np.var(values)), 6)

    mse_mean, mse_var = stats(_finite_values(metrics_list, "mse"))
    rmse_mean, rmse_var = stats(_finite_values(metrics_list, "rmse"))
    sse_mean, sse_var = stats(_finite_values(metrics_list, "ss_error"))

    return {
        "n_samples": len(metrics_list),
        "mse_mean": mse_mean,
        "mse_variance": mse_var,
        "rmse_mean": rmse_mean,
        "rmse_variance": rmse_var,
        "ss_error_mean": sse_mean,
        "ss_error_variance": sse_var,
        "initial_error": INITIAL_ERROR,
    }


def _shared_config(extra: dict) -> dict:
    config = {
        "scenario": MEDIUM_SCENARIO,
        "controllers": CONTROLLERS,
        "target": TARGET,
        "max_iter": MAX_ITER,
        "max_scenarios": MAX_SCENARIOS,
        "dt": DT,
        "max_time": MAX_TIME,
    }
    config.update(extra)
    return config


# --- 1) Same config, DIFFERENT seeds --------------------------------------

def test_randomness_same_config_diff_seeds(record_report):
    plants = {}
    all_metrics = []

    for plant in PLANTS:
        per_seed = {}
        for seed in DIFF_SEEDS:
            metrics = _last_iteration_metrics(plant, seed)
            assert metrics is not None, f"{plant}/seed={seed}: no history"
            assert "mse" in metrics, f"{plant}/seed={seed}: metrics missing mse"
            per_seed[f"seed={seed}"] = metrics
            all_metrics.append(metrics)
        plants[plant] = per_seed

    record_report(REPORT_NAME, {
        "same_config_diff_seeds": {
            "title": "Same config, different seeds",
            "description": (
                "Drives run_optimization on each built-in plant under a fixed "
                "'medium' scenario (IC=[0.5, 0.5], randomness_level=0.5, "
                "disturbance_level=0.2) with PID and max_iter=5, over seeds "
                f"{DIFF_SEEDS}. Last-iteration metrics are aggregated over all "
                "plants x seeds."
            ),
            "config": _shared_config({"seeds": DIFF_SEEDS}),
            "aggregate": _aggregate(all_metrics),
            "plants": plants,
        }
    })


# --- 2) Same config, SAME seed, repeated runs (record LAST run only) -------

def test_randomness_same_seed_repeated_runs(record_report):
    plants = {}
    all_metrics = []

    for plant in PLANTS:
        last_metrics = None
        for _ in range(N_REPEATS):
            last_metrics = _last_iteration_metrics(plant, REPEAT_SEED)
            assert last_metrics is not None, f"{plant}: no history"
        assert "mse" in last_metrics, f"{plant}: metrics missing mse"
        plants[plant] = last_metrics          # only the LAST run is recorded
        all_metrics.append(last_metrics)

    record_report(REPORT_NAME, {
        "same_seed_repeated_runs": {
            "title": "Repeated runs (same seed)",
            "description": (
                "Same 'medium' scenario, PID and max_iter=5, but a single fixed "
                f"seed ({REPEAT_SEED}) run {N_REPEATS} times per plant; only the "
                "last run's metrics are recorded per plant (the design 'seed' "
                "does not reseed the mock agents' gain RNG, so repeats differ). "
                "Values aggregate the recorded last runs across the plants."
            ),
            "config": _shared_config({
                "seed": REPEAT_SEED,
                "repeats": N_REPEATS,
                "recorded": "last run per plant only",
            }),
            "aggregate": _aggregate(all_metrics),
            "plants": plants,
        }
    })
