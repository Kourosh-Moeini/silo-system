"""
tests/test_design.py

Design-path tests: exercise the async LLM-agent optimization workflow
(POST /silo/start -> poll GET /silo/{job_id}) rather than the one-shot
simulate endpoint.

WHAT WE ACTUALLY VERIFIED (not assumed) drives the assertions here:

  * Every plant, with either the output-feedback (PID) or full-state (FSF)
    controller family, runs the mock graph to `status == "completed"`.
  * A bad custom_dynamics_path makes the *job* fail cleanly: the worker
    catches the FileNotFoundError, sets status="failed" with that error, and
    never publishes a result_summary.
  * `result_summary` has a STABLE STRUCTURE but is always empty of activity:
    {"progress_count": 0, "llm_call_count": 0, "scenario_metrics": []}. The
    DesignMonitor is created and passed into run_optimization, but no graph
    node ever calls it, and run_optimization returns keys that
    _summarize_result doesn't copy (see api/silo_service.py). So we split this
    into two tests:
        - a PASSING regression guard on the structural contract, and
        - an xfail(strict) that says "a real design run should record LLM
          activity" — it will start passing (and flag itself) the day the
          monitor gets wired.

All tests use the `client` and `poll_job` fixtures from tests/conftest.py.
"""

import pytest


BUILTIN_PLANTS = ["ball_beam", "dc_motor", "inverted_pendulum"]

# Controller *family* is selected by the gain keys the mock proposes, which is
# driven by `controllers`: "PID" -> Kp/Ki/Kd, "FSF" -> K1..K4.
CONTROLLER_FAMILIES = ["PID", "FSF"]


def _start_design_job(client, **config_overrides):
    """Start a small, fast design job and return its job_id."""
    config = {
        "system_name": "ball_beam",
        "seed": 42,
        "max_scenarios": 1,
        "max_iter": 3,
        "controllers": ["PID"],
    }
    config.update(config_overrides)
    r = client.post("/silo/start", json={
        "config": config,
        "control_objective": "stabilize the plant output at 0",
    })
    assert r.status_code == 200
    return r.json()["job_id"]


# ---------------------------------------------------------------------------
# 1) COMPLETION — the whole start -> poll -> completed lifecycle works.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plant", BUILTIN_PLANTS)
@pytest.mark.parametrize("controller", CONTROLLER_FAMILIES)
def test_design_job_completes(client, poll_job, plant, controller):
    """Every plant x {PID, FSF} runs the mock optimization graph to completion.

    (3 plants x 2 controller families = 6 cases.) A completed job must carry no
    error and must publish a result_summary object.
    """
    job_id = _start_design_job(client, system_name=plant, controllers=[controller])
    data = poll_job(client, job_id)

    assert data["status"] == "completed", (
        f"{plant}/{controller} ended as {data['status']!r}, error={data.get('error')!r}"
    )
    assert data["error"] is None
    assert data["result_summary"] is not None


# ---------------------------------------------------------------------------
# 2) RESULT-SUMMARY STRUCTURE — pin the stable shape of the summary object.
# ---------------------------------------------------------------------------

def test_result_summary_structure(client, poll_job):
    """A completed design job's result_summary has the documented shape.

    We assert TYPES (the structural contract), not values: progress_count and
    llm_call_count are ints; scenario_metrics is a list. This is the part of
    the summary that is correct-by-design, so it's a passing regression guard.
    """
    job_id = _start_design_job(client)
    data = poll_job(client, job_id)
    assert data["status"] == "completed"

    summary = data["result_summary"]
    assert isinstance(summary, dict)
    assert isinstance(summary.get("progress_count"), int)
    assert isinstance(summary.get("llm_call_count"), int)
    assert isinstance(summary.get("scenario_metrics"), list)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: the DesignMonitor is passed into run_optimization but no "
        "graph node ever calls add_llm_response/add_progress, and "
        "_summarize_result copies keys run_optimization doesn't return. So a "
        "real multi-iteration design run reports llm_call_count == 0. This "
        "xfail flips to a FAILURE (prompting its own removal) once the monitor "
        "is wired up."
    ),
)
def test_design_summary_records_activity(client, poll_job):
    """A real design run calls the mock LLM several times; the summary should
    reflect that. Today it reports zero activity -> expected failure."""
    job_id = _start_design_job(client, max_iter=3, max_scenarios=1)
    data = poll_job(client, job_id)
    assert data["status"] == "completed"
    assert data["result_summary"]["llm_call_count"] > 0


# ---------------------------------------------------------------------------
# 3) FAILURE BRANCH — a bad custom dynamics path fails the job cleanly.
# ---------------------------------------------------------------------------

def test_design_job_fails_on_bad_custom_path(client, poll_job):
    """system_name='custom' with a nonexistent file makes the JOB fail.

    Unlike the simulate endpoint (which returns 200 + success=False), the
    design worker raises FileNotFoundError inside initialize_state, the thread
    catches it, and the job goes to status="failed" with that error text and
    NO result_summary. We pin all three facts.
    """
    job_id = _start_design_job(
        client,
        system_name="custom",
        custom_dynamics_path="this_design_file_does_not_exist.py",
    )
    data = poll_job(client, job_id)

    assert data["status"] == "failed"
    assert data["result_summary"] is None
    assert "FileNotFoundError" in (data["error"] or "")
