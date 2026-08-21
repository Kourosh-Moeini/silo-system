"""
reports/make_report.py

STANDALONE report + plot generator.

`pytest` records only RAW facts into reports/latest_results/; every derived
quantity and every figure is produced here, so the analysis can be re-tuned
without re-running the suite.

    pytest tests/ -q                  # step 1: record  -> latest_results/*.json
    python reports/make_report.py      # step 2: analyse -> report.md + plots/

INPUTS  (reports/latest_results/)
    sanity_check.json             pass/fail counts per category + pass rate
    test_randomness_report.json   randomness characterization data
    test_scenario_report.json     the scenario-matrix runs

OUTPUTS
    reports/report.md             three sections: Sanity Check, Randomness
                                  Tests, Scenario Tests (the last with
                                  plant-complexity and per-controller
                                  subsections, each with its plot)
    reports/plots/*.png

DERIVED QUANTITIES
------------------
1. PLANT complexity (0..1) — the three open-loop probes in
   reports/plant_complexity.py (target-hold, nonlinearity, swing), min-max
   normalized across the plants and blended. Echoed to stdout on every run.

2. PARAMETER complexity (0..1) — from the user-input parameters of each run:
   randomness_level, disturbance_level and num_states. Each term is mapped to
   0..1 against a FIXED reference scale (below) rather than min-max over the
   current matrix, so the value stays comparable between archived reports.

3. TOTAL complexity = W_PLANT * plant + W_PARAM * parameter (both weights 1.0
   by default, i.e. the literal sum; use 0.5/0.5 for a 0..1 blend).

4. SUCCESS RATE — one bounded score in (0, 1], shared by the scenario and
   randomness sections, built from the relative residual error
   r = ss_error / initial_error:

       success = 1 / (1 + log10(1 + r))        (SUCCESS_MODE = "log", default)

   The logarithm is the point: a plain exp(-r) underflows to exactly 0.0 once r
   is large (the saturating ball_beam runs reach r ~ 87, i.e. exp(-87) ~ 1e-38),
   which makes all bad runs indistinguishable. On a log scale the whole range
   stays ordered and readable:

       r = 0     -> 1.000     r = 87    -> 0.340
       r = 1     -> 0.768     r = 1e44  -> 0.022
       r = 9     -> 0.500

   Alternatives without touching code paths:
       SUCCESS_MODE = "ratio"  ->  1 / (1 + r)
       SUCCESS_MODE = "exp"    ->  exp(-r)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

# --- make the repo root AND this directory importable ----------------------
_here = Path(__file__).resolve().parent
_root = _here
while _root != _root.parent and not (_root / "api").is_dir():
    _root = _root.parent
for _path in (str(_root), str(_here)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib
matplotlib.use("Agg")               # headless: write files, never open a window
import matplotlib.pyplot as plt

from plant_complexity import compute_plant_complexity, print_plant_complexity


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPORTS_DIR = str(_here)
LATEST_DIR = os.path.join(REPORTS_DIR, "latest_results")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
MARKDOWN_PATH = os.path.join(REPORTS_DIR, "report.md")

SANITY_FILE = "sanity_check.json"
RANDOMNESS_FILE = "test_randomness_report.json"
SCENARIO_FILE = "test_scenario_report.json"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fixed reference scales: the value that maps a parameter term to 1.0 ("hardest").
RANDOMNESS_REF = 1.0      # randomness_level upper bound used by the suite
DISTURBANCE_REF = 1.0     # disturbance_level upper bound used by the suite
NUM_STATES_REF = 10       # CustomDynamicalSystem auto-detects up to 10 states

# Weights of the parameter terms (must sum to 1.0).
PARAM_WEIGHTS = {"randomness": 1 / 3, "disturbance": 1 / 3, "num_states": 1 / 3}

# Total complexity = W_PLANT * plant + W_PARAM * parameter.
W_PLANT = 1.0
W_PARAM = 1.0

# Success-rate definition: "log" (default), "ratio" or "exp".
SUCCESS_MODE = "log"

CONTROLLER_ORDER = ["P", "PI", "PID", "FSF"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(filename: str) -> dict:
    """Read one artifact from reports/latest_results/; {} when absent."""
    path = os.path.join(LATEST_DIR, filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def _unit(value: float, reference: float) -> float:
    """Map a non-negative value to 0..1 against a reference scale."""
    if reference <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(reference)))


def _dimension_unit(count: int, reference: int) -> float:
    """Map a dimension (1 = simplest) to 0..1 against a reference maximum."""
    if reference <= 1:
        return 0.0
    return max(0.0, min(1.0, (float(count) - 1.0) / (float(reference) - 1.0)))


def parameter_complexity(run: dict) -> tuple[dict, float]:
    """Parameter (user-input) complexity of one run.

    Terms: randomness_level, disturbance_level, num_states.
    Returns (per-term 0..1 values, weighted total 0..1)."""
    terms = {
        "randomness": _unit(run["randomness_level"], RANDOMNESS_REF),
        "disturbance": _unit(run["disturbance_level"], DISTURBANCE_REF),
        "num_states": _dimension_unit(run["num_states"], NUM_STATES_REF),
    }
    return terms, sum(PARAM_WEIGHTS[k] * v for k, v in terms.items())


def success_score(ss_error, initial_error) -> float:
    """Bounded success score in (0, 1] from r = ss_error / initial_error.

    Shared by the scenario and randomness sections so the report uses exactly one
    definition. A null (non-finite / diverged) ss_error scores 0.0."""
    if ss_error is None or not initial_error:
        return 0.0

    r = abs(float(ss_error)) / float(initial_error)
    if SUCCESS_MODE == "log":
        return 1.0 / (1.0 + math.log10(1.0 + r))
    if SUCCESS_MODE == "ratio":
        return 1.0 / (1.0 + r)
    if SUCCESS_MODE == "exp":
        return 0.0 if r > 700.0 else math.exp(-r)
    raise ValueError(f"unknown SUCCESS_MODE={SUCCESS_MODE!r}")


def build_records(runs: list, plant_cx: dict) -> list:
    """Attach plant / parameter / total complexity and the success rate to each run."""
    records = []
    for run in runs:
        terms, param_cx = parameter_complexity(run)
        plant = float(plant_cx["complexity"].get(run["plant"], 0.0))
        records.append({
            **run,
            "parameter_terms": terms,
            "parameter_complexity": round(param_cx, 6),
            "plant_complexity": round(plant, 6),
            "total_complexity": round(W_PLANT * plant + W_PARAM * param_cx, 6),
            "success_rate": round(
                success_score(run.get("metrics", {}).get("ss_error"),
                              run.get("initial_error")), 6),
        })
    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_plant_complexity(plant_cx: dict) -> str:
    """Bar chart: each plant vs its complexity rate."""
    plants = sorted(plant_cx["complexity"], key=lambda p: plant_cx["complexity"][p])
    values = [plant_cx["complexity"][p] for p in plants]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(plants, values, color="tab:blue", width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Plant")
    ax.set_ylabel("Plant complexity rate (0 = easiest, 1 = hardest)")
    ax.set_title("Plant complexity\n(target-hold + nonlinearity + swing probes)")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    filename = "plant_complexity.png"
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)
    return f"plots/{filename}"


def plot_success_vs_complexity(records: list, controller: str) -> str | None:
    """Scatter: success rate vs total complexity for one controller (plants pooled)."""
    subset = [r for r in records if r["controller"] == controller]
    if not subset:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter([r["total_complexity"] for r in subset],
               [r["success_rate"] for r in subset],
               s=55, alpha=0.85, color="tab:blue", edgecolors="none")
    ax.set_xlabel(f"Total complexity  ({W_PLANT:g}*plant + {W_PARAM:g}*parameter)")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{controller}: success rate vs total complexity")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    filename = f"success_vs_complexity_{controller}.png"
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close(fig)
    return f"plots/{filename}"


# ---------------------------------------------------------------------------
# Section 1 — Sanity Check
# ---------------------------------------------------------------------------

def render_sanity_check(sanity: dict) -> list:
    lines = ["## 1. Sanity Check", ""]
    if not sanity:
        return lines + ["- No `sanity_check.json`; run `pytest tests/ -q` first."]

    summary = sanity.get("summary", {})
    lines += [
        f"_Generated: {sanity.get('generated_at')} · "
        f"Wall time: {sanity.get('wall_time_seconds')}s_",
        "",
        f"**Pass rate: {sanity.get('pass_rate_percent')}%** — "
        f"{summary.get('passed', 0)} passed, {summary.get('failed', 0)} failed, "
        f"{summary.get('skipped', 0)} skipped, "
        f"{summary.get('xfailed', 0)} xfailed (known gaps), "
        f"{summary.get('xpassed', 0)} xpassed, "
        f"{summary.get('total', 0)} total.",
        "",
        "| Category | Passed | Failed | Skipped | XFail | XPass |",
        "|---|---|---|---|---|---|",
    ]
    for name, counts in (sanity.get("categories") or {}).items():
        lines.append(
            f"| {name} | {counts.get('passed', 0)} | {counts.get('failed', 0)} "
            f"| {counts.get('skipped', 0)} | {counts.get('xfailed', 0)} "
            f"| {counts.get('xpassed', 0)} |"
        )

    comparison = sanity.get("baseline_comparison") or {}
    lines += ["", "**Baseline** — "]
    if comparison.get("status") == "seeded":
        lines[-1] += "seeded from this run (no prior reference)."
    elif comparison:
        lines[-1] += (f"{comparison.get('baseline_generated_at')} · "
                      f"pass-rate delta {comparison.get('pass_rate_delta')}%.")
    else:
        lines[-1] += "not available."
    return lines


# ---------------------------------------------------------------------------
# Section 2 — Randomness Tests
# ---------------------------------------------------------------------------

def render_randomness(randomness: dict) -> list:
    lines = ["", "## 2. Randomness Tests", ""]
    if not randomness:
        return lines + ["- No `test_randomness_report.json`; "
                        "run `pytest tests/test_randomness.py -q` first."]

    lines += [
        "Both experiments drive the full mock design pipeline and record the "
        "last-iteration metrics. Mean/variance are taken over the samples; the "
        "success rate uses the shared definition applied to the mean "
        "steady-state error.",
        "",
        "| Experiment | Samples | MSE mean | MSE var | RMSE mean | RMSE var | "
        "SSE mean | SSE var | **success rate** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value, digits=4):
        return "-" if value is None else f"{value:.{digits}f}"

    for block in randomness.values():
        if not isinstance(block, dict) or "aggregate" not in block:
            continue
        aggregate = block["aggregate"]
        success = success_score(aggregate.get("ss_error_mean"),
                                aggregate.get("initial_error"))
        lines.append(
            f"| {block.get('title', '?')} | {aggregate.get('n_samples', '-')} "
            f"| {fmt(aggregate.get('mse_mean'))} | {fmt(aggregate.get('mse_variance'))} "
            f"| {fmt(aggregate.get('rmse_mean'))} | {fmt(aggregate.get('rmse_variance'))} "
            f"| {fmt(aggregate.get('ss_error_mean'))} "
            f"| {fmt(aggregate.get('ss_error_variance'))} "
            f"| **{success:.4f}** |"
        )

    lines.append("")
    for block in randomness.values():
        if isinstance(block, dict) and block.get("description"):
            lines.append(f"- **{block.get('title', '?')}** — {block['description']}")
    lines += ["", "_Per-plant / per-seed raw metrics stay in "
              "`latest_results/test_randomness_report.json`._"]
    return lines


# ---------------------------------------------------------------------------
# Section 3 — Scenario Tests
# ---------------------------------------------------------------------------

def render_methodology() -> list:
    weights = PARAM_WEIGHTS
    return [
        "",
        "### 3.1 Methodology",
        "",
        "**Plant complexity (0..1)** — three open-loop probes "
        "(`reports/plant_complexity.py`), min-max normalized across the plants "
        "and blended with equal weights:",
        "",
        "1. *target-hold* — start on a non-zero target with the controller off; "
        "score = mean normalized distance drifted away from it.",
        "2. *nonlinearity* — apply a constant control and test homogeneity: for a "
        "linear plant `sim(a*x0, a*u) == a*sim(x0, u)`; score = mean relative "
        "deviation.",
        "3. *swing* — sit on the target, apply a small control action, and count "
        "the output's direction reversals (oscillations).",
        "",
        "**Parameter complexity (0..1)** — weighted blend of each run's "
        "user-input parameters, mapped to 0..1 against fixed reference scales so "
        "the value stays comparable between archived reports:",
        "",
        "| Term | Weight | Reference (maps to 1.0) |",
        "|---|---|---|",
        f"| randomness_level | {weights['randomness']:.4f} | {RANDOMNESS_REF} |",
        f"| disturbance_level | {weights['disturbance']:.4f} | {DISTURBANCE_REF} |",
        f"| num_states | {weights['num_states']:.4f} | {NUM_STATES_REF} |",
        "",
        f"**Total complexity** = {W_PLANT:g} x plant + {W_PARAM:g} x parameter.",
        "",
        f"**Success rate** (`SUCCESS_MODE = \"{SUCCESS_MODE}\"`) — bounded score in "
        "(0, 1] from `r = ss_error / initial_error`:",
        "",
        "```",
        "success = 1 / (1 + log10(1 + r))",
        "```",
        "",
        "The log scale is deliberate: a plain `exp(-r)` underflows to exactly 0 "
        "once `r` is large (the saturating ball_beam runs reach `r ~ 87`), which "
        "makes all bad runs indistinguishable. Reference points: `r=0 -> 1.000`, "
        "`r=1 -> 0.768`, `r=9 -> 0.500`, `r=87 -> 0.340`, `r=1e44 -> 0.022`.",
    ]


def render_plant_complexity(plant_cx: dict, plot_path: str) -> list:
    raw, norm = plant_cx["raw"], plant_cx["normalized"]
    complexity = plant_cx["complexity"]

    lines = [
        "",
        "### 3.2 Plant Complexity",
        "",
        "| Plant | hold (raw) | nonlin. (raw) | swing (raw) | hold | nonlin. | "
        "swing | #states | **complexity** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for plant in sorted(complexity, key=lambda p: complexity[p]):
        lines.append(
            f"| {plant} | {raw['target_hold'][plant]:.4f} "
            f"| {raw['nonlinearity'][plant]:.4f} | {raw['swing'][plant]:.1f} "
            f"| {norm['target_hold'][plant]:.2f} | {norm['nonlinearity'][plant]:.2f} "
            f"| {norm['swing'][plant]:.2f} | {plant_cx['num_states'][plant]} "
            f"| **{complexity[plant]:.3f}** |"
        )
    return lines + ["", f"![plant complexity]({plot_path})"]


def render_parameter_complexity(records: list) -> list:
    lines = [
        "",
        "### 3.3 Parameter Complexity",
        "",
        "`num_states` is a plant property, so the score is listed per "
        "(plant, scenario) pair.",
        "",
        "| Plant | Scenario | randomness | disturbance | #states | "
        "**parameter complexity** |",
        "|---|---|---|---|---|---|",
    ]
    unique = {}
    for record in records:
        unique.setdefault((record["plant"], record["scenario"]), record)
    for (plant, scenario), record in sorted(
            unique.items(), key=lambda item: item[1]["parameter_complexity"]):
        lines.append(
            f"| {plant} | {scenario} | {record['randomness_level']:.2f} "
            f"| {record['disturbance_level']:.2f} | {record['num_states']} "
            f"| **{record['parameter_complexity']:.4f}** |"
        )
    return lines


def render_controller(records: list, controller: str, plot_path: str | None,
                     number: str) -> list:
    subset = sorted((r for r in records if r["controller"] == controller),
                    key=lambda r: r["total_complexity"])
    lines = [
        "",
        f"### {number} Controller {controller}",
        "",
        "| Plant | Scenario | plant cx | param cx | **total cx** | ss_error | "
        "initial error | r | **success rate** | stable |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in subset:
        ss_error = record["metrics"].get("ss_error")
        if ss_error is None:
            ss_text, r_text = "null (diverged)", "-"
        else:
            ss_text = f"{ss_error:.4f}"
            r_text = f"{abs(ss_error) / record['initial_error']:.2f}"
        lines.append(
            f"| {record['plant']} | {record['scenario']} "
            f"| {record['plant_complexity']:.3f} "
            f"| {record['parameter_complexity']:.4f} "
            f"| **{record['total_complexity']:.4f}** | {ss_text} "
            f"| {record['initial_error']:.2f} | {r_text} "
            f"| **{record['success_rate']:.4f}** | {record['metrics'].get('stable')} |"
        )
    if plot_path:
        lines += ["", f"![{controller}]({plot_path})"]
    return lines


def render_scenarios(scenario_report: dict, records: list, plant_cx: dict,
                     plots: dict) -> list:
    lines = ["", "## 3. Scenario Tests", ""]
    if not records:
        return lines + ["- No `test_scenario_report.json`; "
                        "run `pytest tests/test_scenarios.py -q` first."]

    if scenario_report.get("description"):
        lines += [scenario_report["description"]]

    lines += render_methodology()
    lines += render_plant_complexity(plant_cx, plots["plant"])
    lines += render_parameter_complexity(records)

    controllers = [c for c in CONTROLLER_ORDER
                   if any(r["controller"] == c for r in records)]
    controllers += sorted({r["controller"] for r in records} - set(controllers))
    for index, controller in enumerate(controllers, start=4):
        lines += render_controller(records, controller, plots.get(controller),
                                   f"3.{index}")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sanity = load_json(SANITY_FILE)
    randomness = load_json(RANDOMNESS_FILE)
    scenario_report = load_json(SCENARIO_FILE)

    if not (sanity or randomness or scenario_report):
        raise SystemExit(
            f"No artifacts in {LATEST_DIR} — run `pytest tests/ -q` first."
        )

    runs = scenario_report.get("runs") or []
    plants = sorted({r["plant"] for r in runs})

    plant_cx = compute_plant_complexity(plants) if plants else {
        "plants": [], "complexity": {}, "raw": {}, "normalized": {},
        "num_states": {}, "num_actions": {},
    }
    if plants:
        print_plant_complexity(plant_cx)

    records = build_records(runs, plant_cx) if runs else []

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plots = {}
    if records:
        plots["plant"] = plot_plant_complexity(plant_cx)
        for controller in CONTROLLER_ORDER:
            path = plot_success_vs_complexity(records, controller)
            if path:
                plots[controller] = path

    lines = ["# SiLo Benchmark Progress Report", ""]
    lines += render_sanity_check(sanity)
    lines += render_randomness(randomness)
    lines += render_scenarios(scenario_report, records, plant_cx, plots)

    with open(MARKDOWN_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nsanity check  : {'yes' if sanity else 'missing'}")
    print(f"randomness    : {len([b for b in randomness.values() if isinstance(b, dict)])} experiments")
    print(f"scenario runs : {len(records)}  (success mode = {SUCCESS_MODE!r})")
    print(f"report        : {MARKDOWN_PATH}")
    for key, path in plots.items():
        print(f"plot [{key:<5}] : {os.path.join(REPORTS_DIR, path)}")


if __name__ == "__main__":
    main()
