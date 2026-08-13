import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    """A test client that can call the API in-process (no live server needed)."""
    return TestClient(app)

import json
import os
import time
import pytest
import numpy as np
from collections import defaultdict

# Global state to accumulate results during the Pytest run
report_data = {
    "summary": {"passed": 0, "failed": 0, "skipped": 0},
    "categories": defaultdict(lambda: {"passed": 0, "failed": 0}),
    "metrics": {"mse": [], "settling_time": [], "stable_count": 0, "total_sims": 0},
    "flaky_tests": [],
    "wall_time": 0.0
}
start_time = 0.0

def pytest_sessionstart(session):
    """Fired at the very beginning of the test run."""
    global start_time
    start_time = time.time()
    os.makedirs("reports", exist_ok=True)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fired for every test phase (setup, call, teardown)."""
    outcome = yield
    report = outcome.get_result()
    
    # We only care about the actual test execution ("call"), not setup/teardown
    if report.when == "call":
        # Categorize by filename (e.g., test_unit.py -> unit)
        filename = item.location[0]
        category = "other"
        if "unit" in filename: category = "unit"
        elif "simulate" in filename: category = "simulate"
        elif "scenarios" in filename: category = "design"
        elif "fixtures" in filename: category = "fixtures"
        
        # Track pass/fail
        if report.passed:
            report_data["summary"]["passed"] += 1
            report_data["categories"][category]["passed"] += 1
        elif report.failed:
            report_data["summary"]["failed"] += 1
            report_data["categories"][category]["failed"] += 1
        elif report.skipped:
            report_data["summary"]["skipped"] += 1

        # Check for flaky tests (tests that pass but threw warnings or needed retries if using pytest-rerunfailures)
        if report.failed and "randomness" in filename:
             report_data["flaky_tests"].append(item.nodeid)

        # Extract recorded metrics (from record_property)
        for prop_name, prop_value in report.user_properties:
            if prop_name in ["mse", "settling_time"] and isinstance(prop_value, (int, float)):
                report_data["metrics"][prop_name].append(prop_value)
            elif prop_name == "stable":
                report_data["metrics"]["total_sims"] += 1
                if prop_value:
                    report_data["metrics"]["stable_count"] += 1

def pytest_sessionfinish(session, exitstatus):
    """Fired at the very end of the test run. Time to build the reports."""
    report_data["wall_time"] = round(time.time() - start_time, 2)
    
    # 1. Calculate Metric Distributions
    distributions = {}
    for metric in ["mse", "settling_time"]:
        vals = report_data["metrics"][metric]
        if vals:
            distributions[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "min": round(float(np.min(vals)), 4)
            }
    
    stable_rate = 0.0
    if report_data["metrics"]["total_sims"] > 0:
        stable_rate = round((report_data["metrics"]["stable_count"] / report_data["metrics"]["total_sims"]) * 100, 2)
    distributions["stable_rate_percent"] = stable_rate

    # 2. Handle Baseline Comparison
    baseline_path = "reports/baseline.json"
    latest_path = "reports/latest_report.json"
    comparison = {}
    
    if os.path.exists(latest_path):
        # The previous 'latest' becomes the new 'baseline'
        os.replace(latest_path, baseline_path)
        
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
            prev_mse = baseline.get("distributions", {}).get("mse", {}).get("mean", 0)
            curr_mse = distributions.get("mse", {}).get("mean", 0)
            mse_diff = curr_mse - prev_mse
            comparison["mse_trend"] = "degraded" if mse_diff > 0 else "improved or stable"
            comparison["mse_diff"] = round(mse_diff, 4)

    # 3. Compile Final JSON
    final_output = {
        "summary": report_data["summary"],
        "categories": dict(report_data["categories"]),
        "distributions": distributions,
        "flaky_tests": report_data["flaky_tests"],
        "wall_time": report_data["wall_time"],
        "comparison_vs_baseline": comparison
    }

    # Save JSON
    with open(latest_path, "w") as f:
        json.dump(final_output, f, indent=4)
        
    # 4. Generate Markdown Artifact
    md_lines = [
        "# Test Suite Progress Report",
        f"**Wall Time:** {final_output['wall_time']} seconds",
        "",
        "## 📊 Summary",
        f"- **Passed:** {final_output['summary']['passed']}",
        f"- **Failed:** {final_output['summary']['failed']}",
        f"- **Skipped:** {final_output['summary']['skipped']}",
        "",
        "## 📁 Results by Category",
        "| Category | Passed | Failed |",
        "|---|---|---|"
    ]
    for cat, stats in final_output["categories"].items():
        md_lines.append(f"| {cat.capitalize()} | {stats['passed']} | {stats['failed']} |")
        
    md_lines.extend([
        "",
        "## 📈 Metric Distributions",
        f"- **Stability Rate:** {distributions.get('stable_rate_percent', 0)}%"
    ])
    
    for m, stats in distributions.items():
        if isinstance(stats, dict):
            md_lines.append(f"- **{m.upper()}**: Mean: {stats['mean']} | Min: {stats['min']} | Max: {stats['max']}")
            
    if comparison:
        md_lines.extend([
            "",
            "## 🔄 Baseline Comparison",
            f"- **MSE Trend:** {comparison.get('mse_trend', 'N/A')} (Diff: {comparison.get('mse_diff', 0)})"
        ])
        
    if final_output["flaky_tests"]:
        md_lines.extend(["", "## ⚠️ Flaky Tests Detected"])
        for ft in final_output["flaky_tests"]:
            md_lines.append(f"- `{ft}`")
            
    with open("reports/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))