"""Shared pytest fixtures + the Phase-3 progress-report generator.

Running the suite with a single command (``pytest tests/ -q``) produces, as a
side effect of the hooks below, three artifacts under ``reports/``:

  * ``reports/latest_report.json`` — machine-readable snapshot of THIS run.
  * ``reports/report.md``          — the same snapshot, human-readable.
  * ``reports/baseline.json``      — a *blessed* reference the report is
                                     compared against (see "Baseline" below).

Design goals (mapped to ASSIGNMENT.md §3):

  * Pass/fail **by category** — categories are derived deterministically from
    the test module (see ``CATEGORY_BY_MODULE`` / ``_category_for``) so each
    Phase-1/Phase-2 area maps to exactly one honest bucket. No silent "other".
  * **Metric distributions** (mse / settling_time / overshoot / stable-rate)
    are aggregated across every simulate-path test that records them via the
    ``record_sim`` fixture — i.e. across plants x scenarios x controllers x
    seeds x fixtures. Non-finite values (e.g. ``settling_time == inf`` for an
    unstable run) are counted separately instead of poisoning the mean.
  * **Baseline comparison** is against a *blessed* ``baseline.json`` that does
    NOT change run-to-run, so the trend is meaningful. Re-bless it explicitly
    with ``--bless-baseline`` (or ``SILO_BLESS_BASELINE=1``).
  * **Regression gate** (opt-in) fails the session if the pass rate drops or the
    median mse worsens beyond a threshold: ``--regression-gate`` /
    ``SILO_REGRESSION_GATE=1``.
  * **Stable, diffable format** — categories and metrics are emitted in a fixed
    order; only genuinely volatile fields (wall time, timestamp) vary.

The report treats the engine as a black box: it only aggregates what tests pass
to ``record_property`` and the pass/fail outcomes pytest reports.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

REPORTS_DIR = "reports"
LATEST_PATH = os.path.join(REPORTS_DIR, "latest_report.json")
BASELINE_PATH = os.path.join(REPORTS_DIR, "baseline.json")
MARKDOWN_PATH = os.path.join(REPORTS_DIR, "report.md")

# Deterministic category taxonomy. Each test module maps to exactly one area
# from the Phase-1/Phase-2 brief. Unknown future modules (e.g. test_mpc.py)
# auto-categorize from their filename so the report keeps working as MPC /
# SysID / Neuroadaptive land.
CATEGORY_ORDER = [
    "unit",
    "simulate",
    "scenario",
    "design",
    "randomness",
    "failure",
    "fixtures",
]
CATEGORY_BY_MODULE = {
    "test_unit": "unit",
    "test_simulate": "simulate",
    "test_scenarios": "scenario",
    "test_design": "design",
    "test_randomness": "randomness",
    "test_failures": "failure",
    "test_user_fixtures": "fixtures",
}

# Metrics we aggregate into distributions, in a fixed display order.
METRIC_KEYS = ["mse", "settling_time", "overshoot"]

# Regression-gate thresholds (overridable via env).
DEFAULT_MAX_PASS_DROP_PCT = float(os.getenv("SILO_MAX_PASS_DROP_PCT", "1.0"))
DEFAULT_MAX_MSE_WORSEN_PCT = float(os.getenv("SILO_MAX_MSE_WORSEN_PCT", "10.0"))


def _category_for(item) -> str:
    """Map a pytest item to its report category (deterministic, no 'other')."""
    path = str(item.location[0]).replace("\\", "/")
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in CATEGORY_BY_MODULE:
        return CATEGORY_BY_MODULE[stem]
    if stem.startswith("test_"):
        return stem[len("test_"):]
    return stem or "uncategorized"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """A test client that can call the API in-process (no live server needed)."""
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _force_utf8_stdout():
    """Make stdout/stderr UTF-8 for the whole session.

    The mock design graph prints emoji (e.g. '🏁', '🎛️') from a worker thread.
    Under `pytest -s` on a Windows cp1252 console those writes raise
    UnicodeEncodeError, which bubbles up and marks the design *job* FAILED — so
    the design tests would fail only when run with capture disabled. We
    neutralise it test-side (no src edits) by reconfiguring the real streams
    once, before any job runs. Best-effort: under capture the stream has no
    `reconfigure` and we simply no-op.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    yield


@pytest.fixture
def poll_job():
    """Return a helper that polls a design job to completion.

    Usage:
        data = poll_job(client, job_id)          # blocks until terminal
        assert data["status"] == "completed"

    Polls GET /silo/{job_id} until the job reaches a terminal state
    (completed / failed / cancelled) and returns the final status JSON. Raises
    AssertionError on timeout so a stuck job fails loudly instead of hanging.
    """
    def _poll_until_terminal(client, job_id, timeout=30.0, interval=0.05):
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
    return _poll_until_terminal


@pytest.fixture
def record_sim(record_property):
    """Feed a simulate ``metrics`` dict into the Phase-3 progress report.

    Any simulate-path test can call ``record_sim(data["metrics"])`` to add its
    mse / settling_time / overshoot / stable to the aggregated distributions.
    Recording is pure instrumentation: it never changes a test's pass/fail
    outcome. Non-numeric / missing values are skipped silently; non-finite
    numbers are still recorded and bucketed as "non-finite" by the reporter.
    """
    def _record(metrics) -> None:
        if not isinstance(metrics, dict):
            return
        for key in METRIC_KEYS + ["rmse"]:
            if key in metrics:
                try:
                    record_property(key, float(metrics[key]))
                except (TypeError, ValueError):
                    pass
        if "stable" in metrics:
            record_property("stable", bool(metrics["stable"]))
    return _record


# ---------------------------------------------------------------------------
# CLI options for baseline / regression gate
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    group = parser.getgroup("silo-report")
    group.addoption(
        "--bless-baseline",
        action="store_true",
        default=False,
        help="After the run, overwrite reports/baseline.json with this run "
             "(promote it to the blessed regression reference).",
    )
    group.addoption(
        "--regression-gate",
        action="store_true",
        default=False,
        help="Fail the session if pass rate drops or median mse worsens beyond "
             "threshold vs reports/baseline.json.",
    )


# ---------------------------------------------------------------------------
# Report accumulation
# ---------------------------------------------------------------------------

report_data = {
    "summary": {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0},
    "categories": defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    ),
    "metrics": {key: [] for key in METRIC_KEYS},
    "stable_count": 0,
    "sim_count": 0,
    "outcomes_by_nodeid": defaultdict(list),
    "wall_time": 0.0,
}
start_time = 0.0


def pytest_sessionstart(session):
    """Fired at the very beginning of the test run."""
    global start_time
    start_time = time.time()
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _bucket_for(report) -> str | None:
    """Classify a completed test phase into exactly one summary bucket."""
    wasxfail = hasattr(report, "wasxfail")
    if report.when == "call":
        if wasxfail and report.skipped:
            return "xfailed"          # expected failure occurred
        if wasxfail and report.passed:
            return "xpassed"          # xfail marker but it passed (non-strict)
        if report.passed:
            return "passed"
        if report.failed:
            return "failed"           # incl. strict XPASS, which pytest fails
        if report.skipped:
            return "skipped"
    elif report.when in ("setup", "teardown"):
        # A collection/setup error is a real failure; a setup-time skip is a skip.
        if report.failed:
            return "failed"
        if report.skipped and not wasxfail:
            return "skipped"
    return None


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fired for every test phase (setup, call, teardown)."""
    outcome = yield
    report = outcome.get_result()

    # Track raw outcomes per nodeid for honest flaky detection (a test that
    # both fails/reruns and passes within one session is flaky). With no rerun
    # plugin installed this stays empty, which is the truthful answer.
    if report.when == "call":
        report_data["outcomes_by_nodeid"][item.nodeid].append(report.outcome)

    bucket = _bucket_for(report)
    if bucket is None:
        return

    category = _category_for(item)
    report_data["summary"][bucket] += 1
    report_data["categories"][category][bucket] += 1

    # Aggregate recorded simulation metrics (only meaningful on the call phase).
    if report.when == "call":
        for prop_name, prop_value in report.user_properties:
            if prop_name in report_data["metrics"] and isinstance(prop_value, (int, float)):
                report_data["metrics"][prop_name].append(float(prop_value))
            elif prop_name == "stable":
                report_data["sim_count"] += 1
                if prop_value:
                    report_data["stable_count"] += 1


# ---------------------------------------------------------------------------
# Distribution / comparison helpers
# ---------------------------------------------------------------------------

def _distribution(values) -> dict:
    """Finite-aware summary of a metric list.

    ``settling_time`` is ``inf`` for non-converging runs; folding that into a
    mean would make the whole distribution meaningless. So we split finite from
    non-finite, summarise the finite part, and report the non-finite count.
    """
    arr = [float(v) for v in values]
    finite = [v for v in arr if math.isfinite(v)]
    dist = {
        "count": len(arr),
        "finite_count": len(finite),
        "non_finite_count": len(arr) - len(finite),
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
    }
    if finite:
        a = np.asarray(finite, dtype=float)
        dist.update(
            mean=round(float(np.mean(a)), 4),
            median=round(float(np.median(a)), 4),
            std=round(float(np.std(a)), 4),
            min=round(float(np.min(a)), 4),
            max=round(float(np.max(a)), 4),
        )
    return dist


def _pass_rate(summary) -> float:
    executed = summary["passed"] + summary["failed"]
    return round(100.0 * summary["passed"] / executed, 2) if executed else 100.0


def _compare_to_baseline(current, baseline) -> dict:
    """Deltas of the headline indicators vs the blessed baseline."""
    def _median(rep, metric):
        return (rep.get("metrics", {}).get("distributions", {})
                   .get(metric, {}) or {}).get("median")

    cur_mse = _median(current, "mse")
    base_mse = _median(baseline, "mse")
    cur_ts = _median(current, "settling_time")
    base_ts = _median(baseline, "settling_time")
    cur_pass = current.get("pass_rate_percent")
    base_pass = baseline.get("pass_rate_percent")
    cur_stable = current.get("metrics", {}).get("stable_rate_percent")
    base_stable = baseline.get("metrics", {}).get("stable_rate_percent")

    def _delta(cur, base):
        if cur is None or base is None:
            return None
        return round(cur - base, 4)

    mse_delta = _delta(cur_mse, base_mse)
    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "pass_rate_delta": _delta(cur_pass, base_pass),
        "mse_median_delta": mse_delta,
        "mse_trend": (
            "n/a" if mse_delta is None
            else "degraded" if mse_delta > 0
            else "improved or stable"
        ),
        "settling_time_median_delta": _delta(cur_ts, base_ts),
        "stable_rate_delta": _delta(cur_stable, base_stable),
    }


def _evaluate_gate(current, baseline) -> dict:
    """Regression gate: pass rate must not drop, median mse must not worsen
    beyond configured thresholds."""
    reasons = []
    cur_pass = current.get("pass_rate_percent", 100.0)
    base_pass = baseline.get("pass_rate_percent", 100.0)
    if base_pass - cur_pass > DEFAULT_MAX_PASS_DROP_PCT:
        reasons.append(
            f"pass rate dropped {base_pass - cur_pass:.2f}% "
            f"(> {DEFAULT_MAX_PASS_DROP_PCT}%): {base_pass} -> {cur_pass}"
        )

    cur_mse = (current.get("metrics", {}).get("distributions", {})
                      .get("mse", {}) or {}).get("median")
    base_mse = (baseline.get("metrics", {}).get("distributions", {})
                       .get("mse", {}) or {}).get("median")
    if cur_mse is not None and base_mse:
        worsen_pct = (cur_mse - base_mse) / base_mse * 100.0
        if worsen_pct > DEFAULT_MAX_MSE_WORSEN_PCT:
            reasons.append(
                f"median mse worsened {worsen_pct:.1f}% "
                f"(> {DEFAULT_MAX_MSE_WORSEN_PCT}%): {base_mse} -> {cur_mse}"
            )

    return {"enabled": True, "passed": not reasons, "reasons": reasons}


# ---------------------------------------------------------------------------
# Session finish: build JSON + Markdown, compare, optionally gate
# ---------------------------------------------------------------------------

def _ordered_categories() -> dict:
    """Emit categories in a fixed order (known first, then any extras sorted)."""
    cats = report_data["categories"]
    ordered = {}
    for name in CATEGORY_ORDER:
        if name in cats:
            ordered[name] = dict(cats[name])
    for name in sorted(cats):
        if name not in ordered:
            ordered[name] = dict(cats[name])
    return ordered


def _build_snapshot() -> dict:
    summary = dict(report_data["summary"])
    summary["total"] = sum(summary.values())

    distributions = {m: _distribution(report_data["metrics"][m]) for m in METRIC_KEYS}
    stable_rate = (
        round(100.0 * report_data["stable_count"] / report_data["sim_count"], 2)
        if report_data["sim_count"] else 0.0
    )

    # Honest flaky detection: a nodeid that shows >1 distinct outcome (e.g. a
    # rerun then a pass) within the session. Empty without a rerun plugin.
    flaky = sorted(
        nid for nid, outs in report_data["outcomes_by_nodeid"].items()
        if len(set(outs)) > 1 or "rerun" in outs
    )

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": report_data["wall_time"],
        "summary": summary,
        "pass_rate_percent": _pass_rate(summary),
        "categories": _ordered_categories(),
        "metrics": {
            "sim_count": report_data["sim_count"],
            "stable_count": report_data["stable_count"],
            "stable_rate_percent": stable_rate,
            "distributions": distributions,
        },
        "flaky_tests": flaky,
    }


def _render_markdown(snap: dict) -> str:
    s = snap["summary"]
    lines = [
        "# Test Suite Progress Report",
        f"_Generated: {snap['generated_at']} · Wall time: {snap['wall_time_seconds']}s_",
        "",
        "## 📊 Summary",
        f"- **Passed:** {s['passed']}",
        f"- **Failed:** {s['failed']}",
        f"- **Skipped:** {s['skipped']}",
        f"- **XFailed (known gaps):** {s['xfailed']}",
        f"- **XPassed:** {s['xpassed']}",
        f"- **Pass rate:** {snap['pass_rate_percent']}%",
        "",
        "## 📁 Results by Category",
        "| Category | Passed | Failed | Skipped | XFail |",
        "|---|---|---|---|---|",
    ]
    for name, c in snap["categories"].items():
        lines.append(
            f"| {name} | {c['passed']} | {c['failed']} | {c['skipped']} | {c['xfailed']} |"
        )

    m = snap["metrics"]
    lines += [
        "",
        "## 📈 Metric Distributions",
        f"- **Stable rate:** {m['stable_rate_percent']}% "
        f"({m['stable_count']}/{m['sim_count']} simulations)",
        "",
        "| Metric | n (finite/∞) | Mean | Median | Min | Max | Std |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in METRIC_KEYS:
        d = m["distributions"][key]
        n = f"{d['finite_count']}/{d['non_finite_count']}"
        lines.append(
            f"| {key} | {n} | {d['mean']} | {d['median']} | "
            f"{d['min']} | {d['max']} | {d['std']} |"
        )

    cmp_ = snap.get("baseline_comparison") or {}
    lines += ["", "## 🔄 Baseline Comparison"]
    if cmp_.get("status") == "seeded":
        lines.append("- Baseline seeded from this run (no prior reference).")
    elif cmp_:
        lines += [
            f"- **Baseline:** {cmp_.get('baseline_generated_at')}",
            f"- **Pass-rate delta:** {cmp_.get('pass_rate_delta')}%",
            f"- **MSE median trend:** {cmp_.get('mse_trend')} "
            f"(delta: {cmp_.get('mse_median_delta')})",
            f"- **Settling-time median delta:** {cmp_.get('settling_time_median_delta')}",
            f"- **Stable-rate delta:** {cmp_.get('stable_rate_delta')}%",
        ]
    else:
        lines.append("- No baseline available.")

    gate = snap.get("regression_gate")
    if gate and gate.get("enabled"):
        lines += ["", "## 🚦 Regression Gate",
                  f"- **Status:** {'PASS' if gate['passed'] else 'FAIL'}"]
        for reason in gate.get("reasons", []):
            lines.append(f"  - {reason}")

    lines += ["", "## ⚠️ Flaky Tests"]
    if snap["flaky_tests"]:
        lines += [f"- `{nid}`" for nid in snap["flaky_tests"]]
    else:
        lines.append("- None detected.")

    return "\n".join(lines) + "\n"


def pytest_sessionfinish(session, exitstatus):
    """Build the JSON + Markdown report, compare to the blessed baseline, and
    optionally enforce the regression gate."""
    report_data["wall_time"] = round(time.time() - start_time, 2)
    snap = _build_snapshot()

    bless = session.config.getoption("--bless-baseline") or os.getenv("SILO_BLESS_BASELINE") == "1"
    gate_on = session.config.getoption("--regression-gate") or os.getenv("SILO_REGRESSION_GATE") == "1"

    baseline = None
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH, "r", encoding="utf-8") as fh:
                baseline = json.load(fh)
        except (json.JSONDecodeError, OSError):
            baseline = None

    if baseline is None:
        # First run (or corrupt baseline): seed it so future runs have a
        # stable reference. This run's comparison is marked "seeded".
        snap["baseline_comparison"] = {"status": "seeded"}
    else:
        snap["baseline_comparison"] = _compare_to_baseline(snap, baseline)

    if gate_on:
        snap["regression_gate"] = (
            _evaluate_gate(snap, baseline) if baseline
            else {"enabled": True, "passed": True, "reasons": [], "note": "no baseline"}
        )

    # Write artifacts (stable schema, deterministic ordering).
    with open(LATEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(snap))

    # Seed or (re-)bless the baseline as requested.
    if baseline is None or bless:
        with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2)

    # Enforce the gate: turn a regression into a non-zero exit code.
    gate = snap.get("regression_gate")
    if gate_on and gate and not gate["passed"]:
        print("\n[regression-gate] FAILED:")
        for reason in gate["reasons"]:
            print(f"  - {reason}")
        session.exitstatus = 1
