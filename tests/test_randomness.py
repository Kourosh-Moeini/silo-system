"""
tests/test_randomness.py

Randomness CHARACTERIZATION on the DESIGN path (run_optimization). Both tests
drive the full mock design pipeline and persist a structured block into
reports/latest_report.json via the `record_report_section` fixture. Each block
carries the raw per-run metrics AND an `aggregate` summary (mean/variance of
mse, rmse and steady-state error, plus a success rate), which conftest renders
under the report's "Randomness Tests" section.

  * test_randomness_same_config_diff_seeds
        Fixed "medium" scenario, PID, 5 seeds, 3 plants. Records the
        LAST-ITERATION metrics of every run (plant -> seed) and aggregates over
        all plants x seeds.

  * test_randomness_same_seed_repeated_runs
        Same config, a SINGLE seed, run N times per plant, but ONLY the LAST
        run's metrics are kept per plant. Aggregates over the plants.

WHY RECORDS, NOT EQUALITY ASSERTIONS
------------------------------------
The design `seed` reseeds only np.random (IC + process noise) via
set_global_seed(); the mock agents draw gains from Python's `random`, seeded
once at import and never per job. So different seeds vary, and even the same
seed repeated varies (gains not reseeded). We assert only that runs produced
well-formed metrics and save the values for the report.
"""

import math
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from src.controllers_mock import run_optimization


# --- Fixed "medium" scenario + shared design config -----------------------

PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

# ASSUMPTION: "0/5, 0/5" -> initial_condition_range = [0.5, 0.5] (fixed IC).
MEDIUM_SCENARIO = {
    "id": "medium",
    "initial_condition_range": [0.5, 0.5],
    "randomness_level": 0.5,
    "disturbance_level": 0.2,
    "param_uncertainty": 0.0,
}

DIFF_SEEDS = [1, 7, 42, 100, 2026]
REPEAT_SEED = 42
N_REPEATS = 5
CONTROLLERS = ["PID"]
MAX_ITER = 5
MAX_SCENARIOS = 1
DT = 0.01
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


def _last_iteration_metrics(system_name: str, seed: int):
    """Run one full mock design job; return JSON-safe metrics of the LAST
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
        )
    scenario_history = result.get("all_scenario_history", [])
    if not scenario_history:
        return None
    history = scenario_history[-1].get("history", [])
    if not history:
        return None
    return _json_safe(history[-1].get("metrics", {}))


def _finite_col(metrics_list, key):
    """All finite numeric values of `key` across a list of metrics dicts."""
    return [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]


def _aggregate(metrics_list: list) -> dict:
    """Mean/variance of mse, rmse, ss_error over a list of metrics dicts, plus
    success_rate = exp(-mean ss_error). Non-finite entries are skipped."""
    mse = _finite_col(metrics_list, "mse")
    rmse = _finite_col(metrics_list, "rmse")
    sse = _finite_col(metrics_list, "ss_error")

    ss_mean = float(np.mean(sse)) if sse else None
    success = float(np.exp(-ss_mean)) if (ss_mean is not None and math.isfinite(ss_mean)) else 0.0

    def _r(vals, fn):
        return round(float(fn(vals)), 6) if vals else None

    return {
        "n_samples": len(metrics_list),
        "mse_mean": _r(mse, np.mean),
        "mse_variance": _r(mse, np.var),
        "rmse_mean": _r(rmse, np.mean),
        "rmse_variance": _r(rmse, np.var),
        "ss_error_mean": None if ss_mean is None else round(ss_mean, 6),
        "ss_error_variance": _r(sse, np.var),
        "success_rate": round(success, 6),
    }


def _config(extra: dict) -> dict:
    cfg = {
        "scenario": MEDIUM_SCENARIO,
        "controllers": CONTROLLERS,
        "max_iter": MAX_ITER,
        "max_scenarios": MAX_SCENARIOS,
        "dt": DT,
        "max_time": MAX_TIME,
    }
    cfg.update(extra)
    return cfg


# --- 1) Same config, DIFFERENT seeds --------------------------------------

def test_randomness_same_config_diff_seeds(record_report_section):
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

    record_report_section("test_randomness_same_config_diff_seeds", {
        "title": "Same config, different seeds",
        "description": (
            "Drives run_optimization on each built-in plant under a fixed "
            "'medium' scenario (IC=[0.5, 0.5], randomness_level=0.5, "
            "disturbance_level=0.2) with PID, max_iter=5, over seeds "
            f"{DIFF_SEEDS}. Last-iteration metrics are aggregated over all "
            "plants x seeds. success_rate = exp(-mean steady-state error) in "
            "(0, 1]; 1 means zero average steady-state error."
        ),
        "config": _config({"seeds": DIFF_SEEDS}),
        "aggregate": _aggregate(all_metrics),
        "plants": plants,
    })


# --- 2) Same config, SAME seed, repeated runs (record LAST run only) -------

def test_randomness_same_seed_repeated_runs(record_report_section):
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

    record_report_section("test_randomness_same_seed_repeated_runs", {
        "title": "Repeated runs (same seed)",
        "description": (
            "Same 'medium' scenario, PID and max_iter=5, but a single fixed "
            f"seed ({REPEAT_SEED}) run {N_REPEATS} times per plant; only the "
            "last run's metrics are recorded per plant (the design 'seed' does "
            "not reseed the mock agents' gain RNG, so repeats differ). Values "
            "aggregate the recorded last runs across the plants; statistics and "
            "success_rate are defined as in the different-seeds subsection."
        ),
        "config": _config({"seed": REPEAT_SEED, "repeats": N_REPEATS,
                           "recorded": "last run per plant only"}),
        "aggregate": _aggregate(all_metrics),
        "plants": plants,
    })