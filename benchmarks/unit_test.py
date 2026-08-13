"""
benchmarks/test_unit.py
=======================
 
A STANDALONE unit-check script for the metrics calculation --- NO pytest.
 
Why this exists
---------------
Instead of letting pytest discover and run tests, we run everything ourselves
with `python benchmarks/test_unit.py` and PRINT the results. This makes the
whole idea of testing transparent: a "test" is just (1) run some code, (2) check
the answer against what we expect, (3) count pass/fail and report.
 
We start with the METRICS CALCULATION (`SimulationRunner.calculate_metrics` in
src/simulation.py) because it is pure math on arrays, so we can feed it inputs
whose correct answers we worked out BY HAND.
 
How to run
----------
From the project root (the folder containing `src/`, `api/`, `benchmarks/`):
 
    python benchmarks/test_unit.py
 
You'll see a line per check (PASS/FAIL) and a summary at the end. The script
exits with code 0 if everything passed, 1 otherwise (handy for CI later).
"""
 
# --- Make sure Python can find the `src` package when we run this file directly.
#     benchmarks/ lives next to src/, so we add the project root to the path. ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
 
import numpy as np
from src.simulation import SimulationRunner
 
 
# ===========================================================================
# A tiny home-grown test harness (this is the part pytest normally does for us)
# ===========================================================================
 
class Harness:
    """Collects checks, prints each result, and tallies pass/fail."""
 
    def __init__(self):
        self.passed = 0
        self.failed = 0
 
    def check(self, name, condition, detail=""):
        """Record one check. `condition` should be True to pass."""
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name}   <-- {detail}")
 
    def close(self, approx):
        """Return True if a value is close enough to expected (floats are fuzzy)."""
        return approx  # placeholder to keep API tidy; see approx_equal below
 
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"RESULTS: {self.passed}/{total} checks passed, {self.failed} failed")
        print("=" * 60)
        return self.failed == 0
 
 
def approx_equal(a, b, tol=1e-9):
    """True if two numbers are equal within a tiny tolerance.
 
    Floating-point math is not exact (0.1 + 0.2 != 0.3 on a computer), so we
    never compare floats with '=='. We check they are *close enough* instead.
    """
    return abs(float(a) - float(b)) <= tol
 
 
# ===========================================================================
# A fake "system" object.
#
# calculate_metrics() only reads a few attributes off self.system:
#   dt, max_time, min_control, max_control.
# It does NOT need a real plant or a simulation. So we hand it a tiny stand-in
# with just those fields. This is what "testing in isolation" looks like ---
# we strip away everything the function doesn't actually need.
# ===========================================================================
 
class FakeSystem:
    def __init__(self, dt=0.1, max_time=1.0, min_control=-10.0, max_control=10.0):
        self.dt = dt
        self.max_time = max_time
        self.min_control = min_control
        self.max_control = max_control
 
def make_runner(system):
    """Build a SimulationRunner and attach our fake system to it."""
    runner = SimulationRunner(type(system))  # the class is required by __init__
    runner.system = system                   # but we override with our instance
    return runner
 
 
# ===========================================================================
# The actual checks
# ===========================================================================
 
def test_mse_and_rmse_are_correct(h):
    """MSE = mean of squared errors; RMSE = sqrt of that. Hand-checkable.
 
    errors = [3, 4]  ->  mse = (3^2 + 4^2) / 2 = (9 + 16) / 2 = 12.5
                          rmse = sqrt(12.5)
    (These exact numbers were confirmed by running the real code.)
    """
    print("\n[Test] MSE / RMSE on errors = [3, 4]")
    runner = make_runner(FakeSystem(dt=0.1, max_time=0.2))
    errors = np.array([3.0, 4.0])
    controls = np.array([1.0, -1.0])
 
    m = runner.calculate_metrics(errors, controls)
 
    h.check("mse == 12.5", approx_equal(m["mse"], 12.5), f"got {m['mse']}")
    h.check("rmse == sqrt(12.5)", approx_equal(m["rmse"], np.sqrt(12.5)), f"got {m['rmse']}")
    h.check("rmse == sqrt(mse)", approx_equal(m["rmse"], np.sqrt(m["mse"])), "rmse/mse mismatch")
 
 
def test_expected_keys_exist(h):
    """The scorecard must always contain the metrics the rest of the app relies on."""
    print("\n[Test] all expected metric keys are present")
    runner = make_runner(FakeSystem(dt=0.1, max_time=0.2))
    m = runner.calculate_metrics(np.array([1.0, 0.5, 0.2]), np.array([0.0, 0.0, 0.0]))
 
    expected = [
        "mse", "rmse", "settling_time", "overshoot", "stable",
        "rise_time", "zero_crossings", "control_effort",
        "control_zero_crossings", "ss_error", "stability_margin",
    ]
    for key in expected:
        h.check(f"'{key}' present", key in m, f"missing from {sorted(m.keys())}")
 
 
def test_types_and_signs(h):
    """mse/rmse are never negative, and 'stable' is a real boolean (not a string)."""
    print("\n[Test] value types and signs are sane")
    runner = make_runner(FakeSystem(dt=0.1, max_time=0.2))
    m = runner.calculate_metrics(np.array([2.0, -1.0, 0.5]), np.array([1.0, -1.0, 0.0]))
 
    h.check("mse >= 0", m["mse"] >= 0, f"got {m['mse']}")
    h.check("rmse >= 0", m["rmse"] >= 0, f"got {m['rmse']}")
    h.check("'stable' is a bool", isinstance(m["stable"], (bool, np.bool_)), type(m["stable"]).__name__)
 
 
def test_zero_error_is_perfect(h):
    """If the error is zero the whole time, the system is perfectly at target:
    mse = 0, it counts as stable, and steady-state error = 0.
    (Confirmed against the real code: mse=0.0, stable=True, ss_error=0.0.)
    """
    print("\n[Test] all-zero error => perfect, stable result")
    runner = make_runner(FakeSystem(dt=0.1, max_time=1.0))
    zeros = np.zeros(11)
    m = runner.calculate_metrics(zeros, zeros)
 
    h.check("mse == 0", approx_equal(m["mse"], 0.0), f"got {m['mse']}")
    h.check("ss_error == 0", approx_equal(m["ss_error"], 0.0), f"got {m['ss_error']}")
    h.check("stable is True", bool(m["stable"]) is True, f"got {m['stable']}")
 
 
def test_decaying_error_settles_and_is_stable(h):
    """A textbook decaying response (error shrinks toward 0) should be marked
    stable and have a FINITE settling time (it does eventually settle).
    (Confirmed against the real code: stable=True, settling_time=3.0, finite.)
    """
    print("\n[Test] decaying error => stable with finite settling time")
    runner = make_runner(FakeSystem(dt=0.1, max_time=5.0))
    t = np.arange(0, 5.0, 0.1)
    errors = np.exp(-t)                 # 1.0 down to ~0
    controls = np.zeros_like(errors)
 
    m = runner.calculate_metrics(errors, controls)
 
    h.check("stable is True", bool(m["stable"]) is True, f"got {m['stable']}")
    h.check("settling_time is finite", np.isfinite(m["settling_time"]), f"got {m['settling_time']}")
    h.check("mse > 0 (error was nonzero early)", m["mse"] > 0, f"got {m['mse']}")
 
 
# ===========================================================================
# Run everything
# ===========================================================================
 
def main():
    print("=" * 60)
    print("UNIT CHECKS: metrics calculation (calculate_metrics)")
    print("=" * 60)
 
    h = Harness()
    test_mse_and_rmse_are_correct(h)
    test_expected_keys_exist(h)
    test_types_and_signs(h)
    test_zero_error_is_perfect(h)
    test_decaying_error_settles_and_is_stable(h)
 
    ok = h.summary()
    # Exit code 0 = all good, 1 = something failed. Lets CI (and you) detect failures.
    sys.exit(0 if ok else 1)
 
 
if __name__ == "__main__":
    main()