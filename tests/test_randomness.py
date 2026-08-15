"""
tests/test_randomness.py

Randomness / reproducibility tests. The engine has TWO independent RNG regimes,
and this file covers both — because they behave very differently:

  1. SIMULATE PATH (np.random). The initial condition is drawn with
     np.random.uniform (src/systems.py) and process noise scales with
     randomness_level. This stream IS controllable: seeding np.random makes a
     re-simulation bit-for-bit identical, and different seeds spread the metric
     out. Both facts are asserted below as PASSING tests.

  2. DESIGN PATH (Python's `random`). The mock LLM agents draw gains with
     random.uniform (src/llm_agents_mock.py, seeded once at import), but the
     job's `seed` is applied through set_global_seed(), which reseeds ONLY
     np.random and explicitly NOT `random` (see the comment in src/utils.py).
     So two design jobs with the SAME seed propose DIFFERENT gains. That is
     arguably a product bug (a "seed" that doesn't make the design reproducible),
     so it is encoded as xfail(strict): the day someone adds random.seed(seed)
     to set_global_seed, this test XPASSes and the strict marker turns that into
     a failure prompting its removal.

Simulate tests use the `client` fixture; the design reproducibility test calls
run_optimization directly so it can capture the proposed gains.
"""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest

from src.controllers_mock import run_optimization


# ---------------------------------------------------------------------------
# 1) SIMULATE PATH — np.random is controllable.
# ---------------------------------------------------------------------------

def _simulate_metrics(client, seed):
    """Reseed np.random, then run one simulation. Reseeding immediately before
    the call fixes the entire np.random stream (IC + noise) for that run."""
    np.random.seed(seed)
    body = {
        "system_name": "ball_beam",
        "controller_type": "PID",
        "gains": {"Kp": 5.0, "Ki": 0.1, "Kd": 1.0},
        "scenario": {"initial_condition_range": [-0.5, 0.5],
                     "randomness_level": 0.1, "disturbance_level": 0.0},
        "dt": 0.01,
        "max_time": 5.0,
    }
    r = client.post("/silo/simulate", json=body)
    assert r.status_code == 200
    return r.json()["metrics"]


def test_simulate_same_seed_is_deterministic(client):
    """Same seed -> bit-identical metrics across repeated runs.

    We reseed to the SAME value before each of three runs and require exact
    equality of both mse and rmse. This proves the simulate path is fully
    reproducible when the np.random stream is pinned.
    """
    runs = [_simulate_metrics(client, seed=12345) for _ in range(3)]
    mses = [m["mse"] for m in runs]
    rmses = [m["rmse"] for m in runs]

    assert mses[0] == mses[1] == mses[2], f"mse not reproducible: {mses}"
    assert rmses[0] == rmses[1] == rmses[2], f"rmse not reproducible: {rmses}"


def test_simulate_variance_across_seeds(client, record_property):
    """Different seeds -> genuinely different (but sane) outcomes.

    Asserts every mse is finite and non-negative, that the spread is non-zero
    (proving randomness is actually exercised, not silently ignored), and a
    generous sanity ceiling. Records each mse/stable so the Phase-3 report
    picks up the distribution.
    """
    seeds = list(range(10))
    values = []
    stable_count = 0
    for s in seeds:
        m = _simulate_metrics(client, seed=s)
        mse = m["mse"]
        values.append(mse)
        stable = bool(m.get("stable", False))
        if stable:
            stable_count += 1
        record_property("mse", mse)
        record_property("stable", stable)

    assert all(np.isfinite(v) for v in values), f"non-finite mse present: {values}"
    assert all(v >= 0 for v in values), "mse is never negative"
    assert all(v < 1e6 for v in values), f"mse exploded past sanity ceiling: {values}"
    assert float(np.std(values)) > 0.0, (
        "different seeds produced identical mse -> randomness not exercised"
    )

    # Visible in `pytest -rP`/`-s`; documents the observed variance.
    print(
        f"\n[variance] seeds={len(seeds)} "
        f"mean={np.mean(values):.6f} std={np.std(values):.6f} "
        f"min={np.min(values):.6f} max={np.max(values):.6f} "
        f"stable={stable_count}/{len(seeds)}"
    )


# ---------------------------------------------------------------------------
# 2) DESIGN PATH — `seed` does NOT make the mock agents reproducible.
# ---------------------------------------------------------------------------

def _design_proposed_gains(seed):
    """Run the mock optimization once and return the flat list of proposed
    gain dicts (numeric only). stdout is redirected to swallow the graph's
    emoji prints (and keep the test quiet)."""
    with redirect_stdout(StringIO()):
        result = run_optimization(
            llm_model="mock",
            run_id=1,
            seed=seed,
            system_name="ball_beam",
            max_scenarios=1,
            max_iter=3,
            controllers=["PID"],
        )
    gains = []
    for scen in result.get("all_scenario_history", []):
        for entry in scen.get("history", []):
            params = {k: v for k, v in entry.get("params", {}).items() if k != "reasoning"}
            gains.append(params)
    return gains


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known bug: the design 'seed' is applied via set_global_seed(), which "
        "reseeds only np.random. The mock LLM agents draw gains from Python's "
        "`random` (seeded once at import, never per-job), so two same-seed runs "
        "propose different gains. This xfail flips to a FAILURE (prompting its "
        "removal) once set_global_seed also calls random.seed(seed)."
    ),
)
def test_design_same_seed_is_reproducible():
    """Two design runs with the SAME seed should propose identical gains."""
    first = _design_proposed_gains(seed=20260815)
    second = _design_proposed_gains(seed=20260815)

    assert first and second, "the design run proposed no gains at all"
    assert first == second, (
        "same-seed design runs diverged; "
        f"first proposal={first[0]} vs second proposal={second[0]}"
    )


@pytest.mark.parametrize("seed", [1, 7, 42, 100, 2026])
def test_design_completes_across_seeds(client, poll_job, seed):
    """Whatever the seed, a design job still completes and yields the stable
    result_summary structure. (Documents that seed variation doesn't break the
    lifecycle, even though it doesn't make the gains reproducible.)"""
    r = client.post("/silo/start", json={
        "config": {"system_name": "ball_beam", "seed": seed,
                   "max_scenarios": 1, "max_iter": 3, "controllers": ["PID"]},
        "control_objective": "stabilize the plant output at 0",
    })
    assert r.status_code == 200
    data = poll_job(client, r.json()["job_id"])

    assert data["status"] == "completed", f"seed={seed} -> {data.get('error')!r}"
    summary = data["result_summary"]
    assert isinstance(summary, dict)
    assert isinstance(summary.get("scenario_metrics"), list)
