"""Shared pytest fixtures and the raw-artifact recorder.

`pytest tests/ -q` writes these files into reports/latest_results/ — RAW facts
only, no analysis and no figures:

    sanity_check.json             pass/fail counts per category + pass rate
    test_scenario_report.json     the scenario-matrix runs
    test_randomness_report.json   the randomness characterization data

reports/latest_results_baseline/ holds a blessed mirror of the same files, used
for the pass-rate trend. It is seeded on the first run and refreshed only with
`--bless-baseline` (or SILO_BLESS_BASELINE=1), so trends stay meaningful.

`python reports/make_report.py` then turns reports/latest_results/ into
reports/report.md and the plots under reports/plots/.

A test contributes its own artifact through the `record_report` fixture:

    record_report("test_scenario_report", {"runs": [...]})
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORTS_DIR = "reports"
LATEST_DIR = os.path.join(REPORTS_DIR, "latest_results")
BASELINE_DIR = os.path.join(REPORTS_DIR, "latest_results_baseline")
SANITY_FILE = "sanity_check.json"

# ---------------------------------------------------------------------------
# Category taxonomy — one bucket per test module
# ---------------------------------------------------------------------------

CATEGORY_BY_MODULE = {
    "test_unit": "unit",
    "test_simulate": "simulate",
    "test_scenarios": "scenario",
    "test_design": "design",
    "test_randomness": "randomness",
    "test_failures": "failure",
    "test_user_fixtures": "fixtures",
}

# Display order; any unlisted module is appended alphabetically.
CATEGORY_ORDER = [
    "unit", "simulate", "scenario", "design", "randomness", "failure", "fixtures",
]

OUTCOMES = ("passed", "failed", "skipped", "xfailed", "xpassed")


def _category_of(item) -> str:
    """Map a pytest item to its report category (deterministic, no 'other')."""
    stem = os.path.splitext(os.path.basename(str(item.location[0]).replace("\\", "/")))[0]
    if stem in CATEGORY_BY_MODULE:
        return CATEGORY_BY_MODULE[stem]
    return stem[len("test_"):] if stem.startswith("test_") else (stem or "uncategorized")


# ---------------------------------------------------------------------------
# Collected state (module level so the session hooks can reach it)
# ---------------------------------------------------------------------------

_summary = dict.fromkeys(OUTCOMES, 0)
_per_category = defaultdict(lambda: dict.fromkeys(OUTCOMES, 0))
_recorded_reports: dict[str, dict] = {}
_start_time = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """In-process API client (no live server needed)."""
    return TestClient(app)


@pytest.fixture
def poll_job():
    """Return a helper that polls a design job until it reaches a terminal state.

        data = poll_job(client, job_id)   # blocks until completed/failed/cancelled
    """
    def _poll(client, job_id, timeout=30.0, interval=0.05):
        deadline = time.time() + timeout
        data = None
        while time.time() < deadline:
            data = client.get(f"/silo/{job_id}").json()
            if data.get("status") in ("completed", "failed", "cancelled"):
                return data
            time.sleep(interval)
        raise AssertionError(
            f"Job {job_id} did not reach a terminal state within {timeout}s; "
            f"last status={(data or {}).get('status')!r}"
        )
    return _poll


@pytest.fixture
def record_report():
    """Merge a JSON-serialisable block into reports/<name>.json.

    Characterization tests use this to save their raw results; the block is
    written once at the end of the session. Calling it twice with the same
    `name` merges the top-level keys, so several tests can share one artifact.
    Pure instrumentation — it never affects a test's pass/fail outcome.
    """
    def _record(name: str, payload: dict) -> None:
        _recorded_reports.setdefault(name, {}).update(payload)
    return _record


@pytest.fixture(scope="session", autouse=True)
def _force_utf8_stdout():
    """Make stdout/stderr UTF-8 for the whole session.

    The mock design graph prints emoji from a worker thread; on a Windows cp1252
    console under `pytest -s` those writes would raise UnicodeEncodeError and
    fail the design job. Reconfigure once, best-effort (no-op under capture).
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    yield


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.getgroup("silo-report").addoption(
        "--bless-baseline",
        action="store_true",
        default=False,
        help="After the run, overwrite reports/baseline.json with this run.",
    )


# ---------------------------------------------------------------------------
# Outcome accumulation
# ---------------------------------------------------------------------------

def _outcome_of(report) -> str | None:
    """Classify one completed test phase into exactly one outcome bucket."""
    expected_failure = hasattr(report, "wasxfail")

    if report.when == "call":
        if expected_failure:
            return "xfailed" if report.skipped else "xpassed" if report.passed else "failed"
        if report.passed:
            return "passed"
        if report.failed:
            return "failed"
        if report.skipped:
            return "skipped"
    elif report.when in ("setup", "teardown"):
        if report.failed:
            return "failed"            # a setup/collection error is a real failure
        if report.skipped and not expected_failure:
            return "skipped"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    bucket = _outcome_of(outcome.get_result())
    if bucket:
        _summary[bucket] += 1
        _per_category[_category_of(item)][bucket] += 1


# ---------------------------------------------------------------------------
# Sanity-check snapshot
# ---------------------------------------------------------------------------

def _pass_rate(summary: dict) -> float:
    executed = summary["passed"] + summary["failed"]
    return round(100.0 * summary["passed"] / executed, 2) if executed else 100.0


def _ordered_categories() -> dict:
    ordered = {name: dict(_per_category[name])
               for name in CATEGORY_ORDER if name in _per_category}
    for name in sorted(_per_category):
        ordered.setdefault(name, dict(_per_category[name]))
    return ordered


def _build_sanity_check(wall_time: float) -> dict:
    summary = dict(_summary)
    summary["total"] = sum(summary.values())
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": wall_time,
        "summary": summary,
        "pass_rate_percent": _pass_rate(summary),
        "categories": _ordered_categories(),
    }


def _compare_to_baseline(current: dict, baseline: dict) -> dict:
    current_rate = current.get("pass_rate_percent")
    baseline_rate = baseline.get("pass_rate_percent")
    delta = (None if current_rate is None or baseline_rate is None
             else round(current_rate - baseline_rate, 2))
    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "pass_rate_delta": delta,
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ---------------------------------------------------------------------------
# Session hooks
# ---------------------------------------------------------------------------

def pytest_sessionstart(session):
    global _start_time
    _start_time = time.time()
    os.makedirs(LATEST_DIR, exist_ok=True)


def _write_all(directory: str, artifacts: dict) -> None:
    """Write every {filename: data} artifact into `directory`."""
    os.makedirs(directory, exist_ok=True)
    for filename, data in artifacts.items():
        _write_json(os.path.join(directory, filename), data)


def pytest_sessionfinish(session, exitstatus):
    sanity = _build_sanity_check(round(time.time() - _start_time, 2))

    baseline = _read_json(os.path.join(BASELINE_DIR, SANITY_FILE))
    sanity["baseline_comparison"] = (
        {"status": "seeded"} if baseline is None
        else _compare_to_baseline(sanity, baseline)
    )

    artifacts = {SANITY_FILE: sanity}
    for name, payload in _recorded_reports.items():
        artifacts[f"{name}.json"] = payload

    _write_all(LATEST_DIR, artifacts)

    bless = (session.config.getoption("--bless-baseline")
             or os.getenv("SILO_BLESS_BASELINE") == "1")
    if baseline is None or bless:
        _write_all(BASELINE_DIR, artifacts)
