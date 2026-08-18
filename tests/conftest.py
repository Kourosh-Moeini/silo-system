"""Shared pytest fixtures + the progress-report generator.

Running `pytest tests/ -q` writes three artifacts under reports/ as a side
effect of the session hooks below:

  * reports/latest_report.json — machine-readable snapshot of THIS run.
  * reports/report.md          — the same snapshot, human-readable.
  * reports/baseline.json      — a blessed reference the report is compared
                                 against (pass-rate trend). Re-bless with
                                 `--bless-baseline` (or SILO_BLESS_BASELINE=1).

The report contains:
  * pass/fail counts by category (one bucket per test module),
  * wall time,
  * a pass-rate delta vs the blessed baseline,
  * any custom characterization blocks a test attaches via the
    `record_report_section` fixture (e.g. the randomness metrics).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
import math

import pytest
from fastapi.testclient import TestClient

from api.main import app

# ---------------------------------------------------------------------------
# Paths & category taxonomy
# ---------------------------------------------------------------------------

REPORTS_DIR = "reports"
LATEST_PATH = os.path.join(REPORTS_DIR, "latest_report.json")
BASELINE_PATH = os.path.join(REPORTS_DIR, "baseline.json")
MARKDOWN_PATH = os.path.join(REPORTS_DIR, "report.md")

# Display order for known categories; unknown modules auto-categorize by name.
CATEGORY_ORDER = ["unit", "simulate", "design", "randomness", "failure", "fixtures"]
CATEGORY_BY_MODULE = {
    "test_unit": "unit",
    "test_simulate": "simulate",
    "test_scenarios": "scenario",
    "test_design": "design",
    "test_randomness": "randomness",
    "test_failures": "failure",
    "test_user_fixtures": "fixtures",
}

# Top-level snapshot keys that are NOT custom characterization blocks.
_RESERVED_KEYS = {
    "generated_at", "wall_time_seconds", "summary",
    "pass_rate_percent", "categories", "baseline_comparison",
}

# ---------------------------------------------------------------------------
# Scenario matrix: complexity rate, success rate, and report plots
# ---------------------------------------------------------------------------

PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Matrix axes, ordered EASY -> HARD (must match the keys test_scenarios records).
SCEN_SCENARIOS   = ["easy", "medium", "hard"]      # list of 3
SCEN_CONTROLLERS = ["P", "PI", "PID", "FSF"]        # list of 4

# gain-range label -> exact key stored in the report JSON.
SCEN_GAIN_KEY = {
    "[0, 1]":  "gains[0.0,1.0]",
    "[1, 10]": "gains[1.0,10.0]",
    "[-5, 0]": "gains[-5.0,0.0]",
}
# Gains in the COMPLEXITY / SUCCESS analysis: size 2, [-5,0] excluded (it's a different thing).
SCEN_COMPLEXITY_GAINS = ["[0, 1]", "[1, 10]"]       # easy -> hard
# All three gain ranges are used only for the control-effort plot.
SCEN_ALL_GAINS = ["[0, 1]", "[1, 10]", "[-5, 0]"]

# Weights (sum to 1.0): scenario 0.5, controller 0.2, gain 0.3.
SCEN_WEIGHTS = {"scenario": 0.50, "controller": 0.20, "gain": 0.30}


def _scen_complexity(si: int, ci: int, gi: int, n_gain: int) -> float:
    """Index-normalize each axis to 0..1, weight, and sum."""
    norm_s = si / (len(SCEN_SCENARIOS) - 1)     # 3 items -> denom 2
    norm_c = ci / (len(SCEN_CONTROLLERS) - 1)   # 4 items -> denom 3
    norm_g = gi / (n_gain - 1)                  # 2 items -> denom 1
    score = (norm_s * SCEN_WEIGHTS["scenario"]
             + norm_c * SCEN_WEIGHTS["controller"]
             + norm_g * SCEN_WEIGHTS["gain"])
    return round(score, 4)


def _analyze_scenarios_and_plot(snap: dict):
    """Augment the test_scenarios cells with complexity + success_rate, write
    PNG plots under reports/plots/, and stash a summary at
    snap['test_scenarios']['analysis']. No-op if the section is absent."""
    section = snap.get("test_scenarios")
    if not isinstance(section, dict) or "plants" not in section:
        return None

    import matplotlib
    matplotlib.use("Agg")               # headless: write files, never open a window
    import matplotlib.pyplot as plt

    plants = section["plants"]
    plant_names = list(plants.keys())
    os.makedirs(PLOTS_DIR, exist_ok=True)

    per_plant = {p: {"x": [], "y": []} for p in plant_names}
    combo_success = defaultdict(list)     # (scenario, controller, gain) -> [success per plant]
    combo_complexity = {}                 # (scenario, controller, gain) -> complexity
    effort_by_gain = defaultdict(list)    # gain label -> [control_effort over everything]

    n_gain = len(SCEN_COMPLEXITY_GAINS)
    for p in plant_names:
        for si, s in enumerate(SCEN_SCENARIOS):
            for ci, c in enumerate(SCEN_CONTROLLERS):
                # complexity + success: only the two participating gain ranges
                for gi, glabel in enumerate(SCEN_COMPLEXITY_GAINS):
                    cell = plants.get(p, {}).get(s, {}).get(c, {}).get(SCEN_GAIN_KEY[glabel])
                    if not isinstance(cell, dict):
                        continue
                    comp = _scen_complexity(si, ci, gi, n_gain)
                    cell["complexity"] = comp
                    ss = cell.get("ss_error")
                    if ss is None:
                        continue
                    success = math.exp(-ss)
                    cell["success_rate"] = round(success, 6)
                    per_plant[p]["x"].append(comp)
                    per_plant[p]["y"].append(success)
                    combo = (s, c, glabel)
                    combo_success[combo].append(success)
                    combo_complexity[combo] = comp
                # control effort: all three gain ranges (incl. [-5,0])
                for glabel in SCEN_ALL_GAINS:
                    cell = plants.get(p, {}).get(s, {}).get(c, {}).get(SCEN_GAIN_KEY[glabel])
                    if isinstance(cell, dict) and cell.get("control_effort") is not None:
                        effort_by_gain[glabel].append(cell["control_effort"])

    plots = {}

    # 1) per-plant: complexity vs success rate
    per_plant_paths = {}
    for p in plant_names:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(per_plant[p]["x"], per_plant[p]["y"], s=45, alpha=0.8)
        ax.set_xlabel("Complexity rate")
        ax.set_ylabel("Success rate = exp(-ss_error)")
        ax.set_title(f"{p}: complexity vs success rate")
        ax.grid(True, alpha=0.3)
        fname = f"scenario_{p}_complexity_vs_success.png"
        fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150); plt.close(fig)
        per_plant_paths[p] = f"plots/{fname}"
    plots["per_plant"] = per_plant_paths

    # 2) all plants: complexity vs MEAN success rate (mean over plants, per combination)
    combos_sorted = sorted(combo_complexity, key=lambda k: combo_complexity[k])
    xs = [combo_complexity[k] for k in combos_sorted]
    ys = [sum(combo_success[k]) / len(combo_success[k]) for k in combos_sorted]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=45, alpha=0.8, color="tab:red")
    ax.set_xlabel("Complexity rate")
    ax.set_ylabel("Mean success rate (over all plants)")
    ax.set_title("Complexity vs mean success rate — all plants")
    ax.grid(True, alpha=0.3)
    fname = "scenario_all_plants_complexity_vs_mean_success.png"
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150); plt.close(fig)
    plots["all_plants_mean"] = f"plots/{fname}"

    # 3) mean control effort by gain range (incl. [-5,0])
    labels = [g for g in SCEN_ALL_GAINS if effort_by_gain[g]]
    means = [sum(effort_by_gain[g]) / len(effort_by_gain[g]) for g in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, color=["tab:blue", "tab:orange", "tab:green"][:len(labels)])
    ax.bar_label(bars, fmt="%.4f")
    ax.set_xlabel("Gain range")
    ax.set_ylabel("Mean control effort (all plants/scenarios/controllers)")
    ax.set_title("Mean control effort by gain range")
    ax.grid(True, axis="y", alpha=0.3)
    fname = "scenario_control_effort_by_gain.png"
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150); plt.close(fig)
    plots["control_effort"] = f"plots/{fname}"

    section["analysis"] = {
        "weights": SCEN_WEIGHTS,
        "complexity_gains": SCEN_COMPLEXITY_GAINS,
        "combinations_sorted": [
            {"scenario": s, "controller": c, "gain": g,
             "complexity": combo_complexity[(s, c, g)],
             "mean_success": round(sum(combo_success[(s, c, g)]) / len(combo_success[(s, c, g)]), 6)}
            for (s, c, g) in combos_sorted
        ],
        "mean_control_effort_by_gain": {
            g: round(sum(effort_by_gain[g]) / len(effort_by_gain[g]), 6) for g in labels
        },
        "plots": plots,
    }
    return section["analysis"]


def _render_scenario_section(snap: dict, lines: list) -> None:
    """Embed the scenario plots + a small table into report.md (alongside the
    randomness section). No-op unless _analyze_scenarios_and_plot ran."""
    analysis = (snap.get("test_scenarios") or {}).get("analysis")
    if not analysis:
        return
    lines += [
        "", "## Scenario Complexity vs Success",
        "Complexity rate = index-normalized, weighted blend of scenario (0.5), "
        "controller (0.2) and gain-range (0.3) difficulty; the [-5,0] gain range is "
        "excluded here (treated separately). Success rate = exp(-steady-state error).",
    ]
    for p, path in analysis["plots"].get("per_plant", {}).items():
        lines += ["", f"### {p}: complexity vs success rate", "", f"![{p}]({path})"]
    if analysis["plots"].get("all_plants_mean"):
        lines += ["", "### All plants: complexity vs mean success rate", "",
                  f"![all plants]({analysis['plots']['all_plants_mean']})"]
    if analysis["plots"].get("control_effort"):
        lines += ["", "### Mean control effort by gain range", "",
                  f"![control effort]({analysis['plots']['control_effort']})"]
    ce = analysis.get("mean_control_effort_by_gain", {})
    if ce:
        lines += ["", "| Gain range | Mean control effort |", "|---|---|"]
        lines += [f"| {g} | {v} |" for g, v in ce.items()]

def _category_for(item) -> str:
    """Map a pytest item to its report category (deterministic, no 'other')."""
    path = str(item.location[0]).replace("\\", "/")
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem in CATEGORY_BY_MODULE:
        return CATEGORY_BY_MODULE[stem]
    return stem[len("test_"):] if stem.startswith("test_") else (stem or "uncategorized")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """In-process API client (no live server needed)."""
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def _force_utf8_stdout():
    """Make stdout/stderr UTF-8 for the whole session.

    The mock design graph prints emoji from a worker thread; on a Windows
    cp1252 console under `pytest -s` those writes would raise UnicodeEncodeError
    and fail the design job. Reconfigure once, best-effort (no-op under capture).
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
    """Return a helper that polls a design job until a terminal state.

        data = poll_job(client, job_id)   # blocks until completed/failed/cancelled
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
def record_report_section():
    """Persist an arbitrary JSON-serialisable block into the final report.

    Characterization tests (e.g. randomness) use this to save structured
    config + metrics under a named top-level key in reports/latest_report.json.
    Pure instrumentation: it never affects a test's pass/fail outcome.

        record_report_section("my_block", {"config": {...}, "plants": {...}})
    """
    def _record(name: str, payload: dict) -> None:
        report_data["custom_sections"][name] = payload
    return _record


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
# Accumulation
# ---------------------------------------------------------------------------

report_data = {
    "summary": {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0},
    "categories": defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    ),
    "custom_sections": {},
    "wall_time": 0.0,
}
start_time = 0.0


def pytest_sessionstart(session):
    global start_time
    start_time = time.time()
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _bucket_for(report) -> str | None:
    """Classify one completed test phase into exactly one summary bucket."""
    wasxfail = hasattr(report, "wasxfail")
    if report.when == "call":
        if wasxfail and report.skipped:
            return "xfailed"       # expected failure occurred
        if wasxfail and report.passed:
            return "xpassed"       # xfail marker but it passed
        if report.passed:
            return "passed"
        if report.failed:
            return "failed"        # incl. strict XPASS
        if report.skipped:
            return "skipped"
    elif report.when in ("setup", "teardown"):
        if report.failed:
            return "failed"        # a setup/collection error is a real failure
        if report.skipped and not wasxfail:
            return "skipped"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    bucket = _bucket_for(report)
    if bucket is None:
        return
    report_data["summary"][bucket] += 1
    report_data["categories"][_category_for(item)][bucket] += 1


# ---------------------------------------------------------------------------
# Snapshot / report building
# ---------------------------------------------------------------------------

def _pass_rate(summary) -> float:
    executed = summary["passed"] + summary["failed"]
    return round(100.0 * summary["passed"] / executed, 2) if executed else 100.0


def _ordered_categories() -> dict:
    cats = report_data["categories"]
    ordered = {name: dict(cats[name]) for name in CATEGORY_ORDER if name in cats}
    for name in sorted(cats):
        ordered.setdefault(name, dict(cats[name]))
    return ordered


def _build_snapshot() -> dict:
    summary = dict(report_data["summary"])
    summary["total"] = sum(summary.values())

    snapshot = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": report_data["wall_time"],
        "summary": summary,
        "pass_rate_percent": _pass_rate(summary),
        "categories": _ordered_categories(),
    }
    # Merge custom characterization blocks (e.g. randomness) as top-level keys.
    for name, payload in report_data["custom_sections"].items():
        if name not in snapshot:
            snapshot[name] = payload
    return snapshot


def _compare_to_baseline(current, baseline) -> dict:
    cur = current.get("pass_rate_percent")
    base = baseline.get("pass_rate_percent")
    delta = None if cur is None or base is None else round(cur - base, 2)
    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "pass_rate_delta": delta,
    }


def _render_markdown(snap: dict) -> str:
    s = snap["summary"]
    lines = [
        "# Test Suite Progress Report",
        f"_Generated: {snap['generated_at']} · Wall time: {snap['wall_time_seconds']}s_",
        "",
        "## Summary",
        f"- Passed: {s['passed']}",
        f"- Failed: {s['failed']}",
        f"- Skipped: {s['skipped']}",
        f"- XFailed (known gaps): {s['xfailed']}",
        f"- XPassed: {s['xpassed']}",
        f"- Pass rate: {snap['pass_rate_percent']}%",
        "",
        "## Results by Category",
        "| Category | Passed | Failed | Skipped | XFail |",
        "|---|---|---|---|---|",
    ]
    for name, c in snap["categories"].items():
        lines.append(f"| {name} | {c['passed']} | {c['failed']} | {c['skipped']} | {c['xfailed']} |")

    cmp_ = snap.get("baseline_comparison") or {}
    lines += ["", "## Baseline Comparison"]
    if cmp_.get("status") == "seeded":
        lines.append("- Baseline seeded from this run (no prior reference).")
    elif cmp_:
        lines += [
            f"- Baseline: {cmp_.get('baseline_generated_at')}",
            f"- Pass-rate delta: {cmp_.get('pass_rate_delta')}%",
        ]
    else:
        lines.append("- No baseline available.")

    _render_custom_sections(snap, lines)
    _render_scenario_section(snap, lines)       # <-- add this line
    return "\n".join(lines) + "\n"


def _render_custom_sections(snap: dict, lines: list) -> None:
    """Render blocks attached via record_report_section.

    Blocks that carry an 'aggregate' summary (the randomness tests) are shown
    under 'Randomness Tests' as subsections with a stats table; the raw
    per-plant/seed data stays in latest_report.json. Any other custom blocks
    are listed as pointers.
    """
    customs = [(k, v) for k, v in snap.items()
               if k not in _RESERVED_KEYS and isinstance(v, dict)]
    aggregated = [(k, v) for k, v in customs if "aggregate" in v]
    plain = [k for k, v in customs if "aggregate" not in v]

    if aggregated:
        lines += ["", "## Randomness Tests"]
        for key, sec in aggregated:
            lines += ["", f"### {sec.get('title', key)}"]
            if sec.get("description"):
                lines += [sec["description"], ""]
            agg = sec["aggregate"]
            lines += [
                "| Statistic | Value |",
                "|---|---|",
                f"| Samples | {agg.get('n_samples')} |",
                f"| MSE mean | {agg.get('mse_mean')} |",
                f"| MSE variance | {agg.get('mse_variance')} |",
                f"| RMSE mean | {agg.get('rmse_mean')} |",
                f"| RMSE variance | {agg.get('rmse_variance')} |",
                f"| Steady-state error mean | {agg.get('ss_error_mean')} |",
                f"| Steady-state error variance | {agg.get('ss_error_variance')} |",
                f"| Success rate = exp(-mean SSE) | {agg.get('success_rate')} |",
                "",
                "_Per-plant/seed raw metrics are in latest_report.json._",
            ]

    if plain:
        lines += ["", "## Characterization Data (see latest_report.json)"]
        lines += [f"- `{k}`" for k in plain]


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _dump_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def pytest_sessionfinish(session, exitstatus):
    report_data["wall_time"] = round(time.time() - start_time, 2)
    snap = _build_snapshot()

    baseline = _load_json(BASELINE_PATH)
    if baseline is None:
        snap["baseline_comparison"] = {"status": "seeded"}
    else:
        snap["baseline_comparison"] = _compare_to_baseline(snap, baseline)

    try:                                        # <-- add these 4 lines
        _analyze_scenarios_and_plot(snap)
    except Exception as exc:                    # never let reporting break the run
        print(f"[scenario analysis] skipped: {exc}")

    _dump_json(LATEST_PATH, snap)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(snap))

    bless = (session.config.getoption("--bless-baseline")
             or os.getenv("SILO_BLESS_BASELINE") == "1")
    if baseline is None or bless:
        _dump_json(BASELINE_PATH, snap)