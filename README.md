# Automated Benchmark Testing — SiLo Module

Automated benchmark test suite and progress-reporting workflow for the **SiLo**
(single-loop) AI control-design pipeline. LLM agents are **mocked** — no API
keys, no `.env`, no database, no network access in any test.

See **ASSIGNMENT.md** for the original brief.

## Layout

| Path | Role |
|------|------|
| `api/` | Provided FastAPI surface (`/silo/start`, `/silo/simulate`, …) |
| `src/` | Provided engine: mock LangGraph workflow, plants, simulation, metrics |
| `case_studies/` | Provided reference dynamics (ball_beam, dc_motor, inverted_pendulum) |
| `tests/` | The benchmark suite (this work) |
| `tests/fixtures/user_dynamics/` | User-dynamics registry + sample plants |
| `reports/` | Reporting tools + generated report, plots and raw results |

`api/`, `src/` and `case_studies/` are **unmodified** — no bug fixes were needed.

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Optional sanity check of the provided API (run from the repo root):

```bash
uvicorn api.main:app --port 8000      # OpenAPI at http://localhost:8000/docs
```

## Run the suite and generate the report

Two explicit steps. `pytest` only **records raw facts**; all analysis and every
figure is produced by the report tool, so the analysis can be re-tuned without
re-running the suite.

```bash
pytest tests/ -q                  # 1. record  -> reports/latest_results/*.json
python reports/make_report.py     # 2. analyse -> reports/report.md + plots/
```

### What gets produced

| File | Role |
|------|------|
| `reports/latest_results/sanity_check.json` | Pass/fail counts per category + pass rate |
| `reports/latest_results/test_scenario_report.json` | The 36 scenario-matrix runs (raw) |
| `reports/latest_results/test_randomness_report.json` | Randomness characterization data (raw) |
| `reports/latest_results_baseline/` | Blessed mirror of the three files above |
| `reports/report.md` | Human-readable progress report |
| `reports/plots/*.png` | Plant complexity + one scatter per controller |

The formats are stable so reports can be diffed or archived over time.

`reports/plant_complexity.py` is also runnable on its own to print just the
plant-complexity table:

```bash
python reports/plant_complexity.py
```

## Test suite

One module per area; each maps to exactly one report category.

| Module | Category | What it exercises |
|--------|----------|-------------------|
| `test_unit.py` | `unit` | Metrics math, parameter validation (422), schema defaults, job-store lifecycle |
| `test_simulate.py` | `simulate` | `POST /silo/simulate` across the built-in plants |
| `test_design.py` | `design` | `POST /silo/start` → poll `GET /silo/{job_id}`; `result_summary` contract; clean job failure |
| `test_scenarios.py` | `scenario` | 3 plants × 3 scenarios × 4 controllers = 36 design runs |
| `test_randomness.py` | `randomness` | Same config/different seeds; same seed/repeated runs |
| `test_failures.py` | `failure` | Bad paths, unknown job 404s, malformed bodies, cancel race, documented quirks |
| `test_user_fixtures.py` | `fixtures` | Registry-driven user-dynamics smoke + state-dimension check |

A new module is categorized automatically: `test_mpc.py` → `mpc`.

`tests/probe_complexity.py` is a **sandbox** used to develop the plant-complexity
methodology. It is not collected by pytest (the name does not start with
`test_`); the production version is `reports/plant_complexity.py`.

## Report structure

```
1. Sanity Check      pass rate, per-category table, baseline delta
2. Randomness Tests  aggregate table (mean/variance of mse, rmse, ss_error)
3. Scenario Tests
   3.1 Methodology
   3.2 Plant Complexity      + plot
   3.3 Parameter Complexity
   3.4 Controller P          + plot
   3.5 Controller PI         + plot
   3.6 Controller PID        + plot
   3.7 Controller FSF        + plot
```

## Methodology

### Plant complexity (0..1)

Three **open-loop** probes per plant (`reports/plant_complexity.py`), each
min-max normalized across the plants and blended with equal weights:

1. **target-hold** — start exactly on a non-zero target with the controller off
   (`u = 0`); score = mean normalized distance it drifts away. A plant that holds
   an arbitrary setpoint scores ~0; one that is pulled off it scores higher.
2. **nonlinearity** — apply a constant control and test homogeneity: a linear
   plant satisfies `sim(a*x0, a*u) == a*sim(x0, u)`; score = mean relative
   deviation from that.
3. **swing** — sit on the target, apply a *small* control action, and count the
   output's direction reversals. Monotonic → 0; oscillatory → higher.

The probes drive `system_dynamics(x, u)` directly rather than
`POST /silo/simulate`, because that endpoint is closed-loop only and cannot
inject a known open-loop `u`.

### Parameter complexity (0..1)

Weighted blend of each run's user-input parameters, each mapped to 0..1 against a
**fixed reference scale** (not min-max over the current matrix) so the value stays
comparable between archived reports:

| Term | Weight | Reference → 1.0 |
|------|--------|-----------------|
| `randomness_level` | 1/3 | 1.0 |
| `disturbance_level` | 1/3 | 1.0 |
| `num_states` | 1/3 | 10 |

### Total complexity

`total = W_PLANT * plant + W_PARAM * parameter`, both weights `1.0` by default
(the literal sum). Set both to `0.5` in `reports/make_report.py` for a 0..1 blend.

### Success rate

Bounded score in (0, 1] from the relative residual error
`r = ss_error / initial_error`:

```
success = 1 / (1 + log10(1 + r))
```

| r | 0 | 1 | 9 | 87 | 1e44 |
|---|---|---|---|---|---|
| success | 1.000 | 0.768 | 0.500 | 0.340 | 0.022 |

The logarithm is deliberate. A plain `exp(-r)` underflows to exactly `0.0` as
soon as `r` is large — the saturating `ball_beam` runs reach `r ≈ 87`, i.e.
`exp(-87) ≈ 1e-38` — which makes every bad run indistinguishable. On a log scale
the whole range stays ordered and readable, and even a numerically diverged run
keeps a small non-zero score. Set `SUCCESS_MODE` to `"ratio"` (`1/(1+r)`) or
`"exp"` in `reports/make_report.py` to compare definitions.

The scenario and randomness sections share this one definition.

## Baseline / regression trend

`reports/latest_results_baseline/` is a blessed mirror of the raw results. It is
seeded automatically on the first run and then left alone, so the pass-rate trend
stays meaningful. Promote the current run explicitly:

```bash
pytest tests/ -q --bless-baseline        # or: SILO_BLESS_BASELINE=1 pytest tests/ -q
```

## Adding a new user-dynamics fixture

A new plant becomes a regression fixture **without touching the harness**: drop a
file in and add one registry entry.

### 1. Drop the dynamics file in

Put the model in `tests/fixtures/user_dynamics/`, e.g. `magnetic_robot.py`. It
must define `dynamics(t, x, u)` returning the state derivatives (SISO — `u` is a
scalar):

```python
import numpy as np

def dynamics(t, x, u):
    # states: x[0] = position, x[1] = velocity ; input: u = force
    force = u[0] if isinstance(u, (list, np.ndarray)) else u
    return np.array([x[1], force - 0.5 * x[1] - 2.0 * x[0]])
```

### 2. Register it

Append one object to `tests/fixtures/user_dynamics/registry.json`:

```json
{
  "name": "Magnetic Robot",
  "path": "magnetic_robot.py",
  "num_states": 2,
  "smoke_gains": { "Kp": 5.0, "Kd": 0.1 },
  "default_scenario": {
    "initial_condition_range": [-0.5, 0.5],
    "randomness_level": 0.0,
    "disturbance_level": 0.0
  }
}
```

| Field | Required | Meaning |
|-------|----------|---------|
| `name` | yes | Human-readable label; becomes the pytest case id |
| `path` | yes | File name, relative to `tests/fixtures/user_dynamics/` |
| `num_states` | optional | Expected state dimension. If given it is **validated** against what the engine auto-detects — a mismatch fails CI |
| `smoke_gains` | yes | Gains that should at least run. The keys pick the controller: `Kp/Ki/Kd` → PID family, `K1..Kn` → full-state feedback |
| `default_scenario` | optional | `initial_condition_range`, `randomness_level`, `disturbance_level` |

### 3. Run the suite — that is the CI path

```bash
pytest tests/ -q
```

`tests/test_user_fixtures.py` discovers every registry entry and generates two
parametrized cases per fixture, with no harness edits:

- `test_user_provided_dynamics[<name>]` — simulates via `POST /silo/simulate` and
  checks the metrics are well-formed.
- `test_user_dynamics_num_states[<name>]` — asserts the engine-detected state
  dimension matches the declared `num_states` (skipped when not declared).

## Assumptions

- **Scenarios.** `easy / mid / hard` are `(initial position, disturbance_level,
  randomness_level)` = `(0.5, 0.0, 0.0)`, `(0.5, 0.5, 0.5)`, `(0.5, 1.0, 1.0)`.
  The initial position is pinned at 0.5 and the target is 0.0, so every run starts
  from the same `initial_error` of 0.5 and the scenarios differ only in noise and
  disturbance. Because that distance never varies it carries no information and is
  deliberately **not** part of the parameter-complexity score.
- **Gain range** is `[0, 100]` for every gain of every controller.
- **Controller families** are selected by the gain *keys* handed to the mock agent
  (`P→Kp`, `PI→Kp/Ki`, `PID→Kp/Ki/Kd`, `FSF→K1..Kn`), so all four are genuinely
  exercised rather than collapsing to FSF.
- **`dt = 0.01`** (the engine/API default). At `dt = 0.1` the stiff DC-motor loop
  diverges under the engine's forward-Euler integrator (mse ~1e89), which would
  drive every `dc_motor` success rate to 0 and hide the real trend.
- Design jobs are polled to a terminal state with a timeout rather than slept on.

## Known limitations and findings

These are properties of the provided engine, verified by running it rather than
assumed. Several are pinned as regression guards so a future change trips a test.

1. **Results depend on test order.** `src/llm_agents_mock.py` calls
   `random.seed(42)` once at *import* and never per job, so the gains proposed
   depend on how many draws happened earlier in the session. Verified:
   `ball_beam/easy/P` yields `ss_error = 15.93` when running
   `pytest tests/test_scenarios.py` alone but `43.50` in a full-suite run. **Run
   the full suite for any report you intend to archive.** A per-run reseed in the
   mock agent would fix this.
2. **Running a subset overwrites `sanity_check.json`** with only that module's
   counts, since it reflects the session that produced it.
3. **`ball_beam` saturates under this gain range.** Its `u` is a beam *angle in
   radians* clipped to ±10, so any gain ≥ 20 saturates immediately
   (`|Kp · 0.5| ≥ 10`) and the ball accelerates away identically regardless of the
   gains — reproduced analytically at `ss_error ≈ 43.3` vs the recorded `43.50`.
   With gains sampled from `[0, 100]` most draws saturate, so `ball_beam` scores
   ~0.34 for every controller. The gain range, not the metric, is the mismatch.
4. **Plant complexity does not predict closed-loop success here.** `ball_beam` is
   rated easiest yet scores worst (saturation, above), while `inverted_pendulum` is
   rated hardest yet scores well — with the coded dynamics its `-mgl·sin θ` term is
   restoring, so it is self-stabilizing at `θ = 0`. The probes measure *intrinsic
   plant properties*, not controllability under a random-gain search.
5. **`stable` is almost always `False` and `settling_time` often infinite**,
   because the mock agents propose random gains rather than designed ones.
6. **The `DesignMonitor` is never wired up.** It is created and passed into
   `run_optimization`, but no graph node calls it, and `_summarize_result` copies
   keys `run_optimization` does not return — so `result_summary` is always
   `{progress_count: 0, llm_call_count: 0, scenario_metrics: []}`. Pinned by
   `xfail(strict=True)` in `test_design.py`, which will flip to a failure (asking
   for its own removal) the day the monitor is connected.
7. **An unknown `system_name` silently falls back to `ball_beam`** instead of
   raising (`create_system` in `src/systems.py`), and **missing gains still run** a
   zero controller. Both are pinned as documented-quirk regression tests.
8. **Cancelling is inherently racy** — mock jobs finish in milliseconds, so
   `test_failures.py` accepts either `200` (accepted) or `400` (already terminal)
   rather than asserting a timing-dependent outcome.
9. `src/utils.py` contains module-level duplicates of
   `calculate_performance_score` / `get_best_entries` that take `self` but are not
   methods — dead code in the provided engine, left untouched.
10. **Parallel execution is not enabled.** The job store is in-memory module-level
    state, and the mock agents share one global RNG, so `-n auto` would make the
    scenario and randomness numbers non-deterministic.
