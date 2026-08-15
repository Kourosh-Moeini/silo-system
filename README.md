# Single-Loop (SiLo) Tuner Module

Introductory single-loop (SiLo) design module for take-home work. LLM agents are **mocked** (no API keys, no `.env`, no database).

## Layout

| Path | Role |
|------|------|
| `api/` | Minimal FastAPI surface (`/silo/start`, `/silo/simulate`, …) |
| `src/` | Mock LangGraph workflow, plants, simulation, metrics |
| `case_studies/` | Reference dynamics (root + `py/` extras) |
| `ASSIGNMENT.md` | Take-home brief: automated benchmark testing |

## Quick start

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Always start uvicorn from the project root so `api` and `src` import cleanly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes for candidates

- Prefer calling `api.silo_service` or the HTTP API from tests; treat `src/` as the engine under test.
- Mock agents are seeded (`random.seed(42)` in `llm_agents_mock`); still vary seeds in benchmarks.
- Custom plants: point `custom_dynamics_path` at a `.py` file that defines `dynamics(t, x, u)`.

See **ASSIGNMENT.md** for the full exercise.

## Running the tests & progress report

The whole suite runs from a single command, and the progress report is written
as a side effect of that same run (no separate step):

```bash
pytest tests/ -q
```

This produces three artifacts under `reports/` (stable, diffable formats):

| File | Role |
|------|------|
| `reports/report.md` | Human-readable progress report for this run |
| `reports/latest_report.json` | Machine-readable snapshot of this run |
| `reports/baseline.json` | **Blessed** reference the report is compared against |

The report contains:

- **Pass/fail by category** — each test module maps to exactly one area
  (`unit`, `simulate`, `scenario`, `design`, `randomness`, `failure`,
  `fixtures`); a new module such as `test_mpc.py` auto-categorizes as `mpc`.
  `xfail` known-gap guards are counted as **XFailed**, not skipped.
- **Metric distributions** — `mse`, `settling_time`, `overshoot` and the
  stable rate, aggregated across every simulate-path test (plants × scenarios ×
  controllers × seeds × fixtures). Both **mean and median** are reported;
  non-converging runs (`settling_time == inf`, nulled to non-finite by the API)
  are counted separately instead of poisoning the mean.
- **Baseline comparison** — deltas of pass rate, median mse, median settling
  time, and stable rate vs `baseline.json`.
- **Wall time and flaky tests** — flaky detection is honest: it reports a test
  only if it shows more than one outcome in a run (requires a rerun plugin;
  otherwise empty).

### Baseline (blessing a reference)

`baseline.json` does **not** change run-to-run, so trends are meaningful. The
first run seeds it; promote a later run to the new reference explicitly:

```bash
pytest tests/ -q --bless-baseline       # or: SILO_BLESS_BASELINE=1 pytest tests/ -q
```

### Regression gate (opt-in)

Fail the run (non-zero exit) if quality regresses vs the baseline:

```bash
pytest tests/ -q --regression-gate      # or: SILO_REGRESSION_GATE=1 pytest tests/ -q
```

Thresholds are configurable via env vars (defaults shown):
`SILO_MAX_PASS_DROP_PCT=1.0` (max allowed pass-rate drop, %),
`SILO_MAX_MSE_WORSEN_PCT=10.0` (max allowed median-mse worsening, %).
The gate is off by default, so ordinary runs stay green.

## Adding a new user-dynamics fixture

Any uploaded `.py` plant can become a regression fixture **without touching the
test harness** — you add a file and one registry entry, and pytest discovers it
automatically.

### 1. Drop your dynamics file in

Put the model in `tests/fixtures/user_dynamics/`, e.g.
`tests/fixtures/user_dynamics/magnetic_robot.py`. It must define a
`dynamics(t, x, u)` function returning the state derivatives as a NumPy array
(SISO — `u` is a scalar / single input):

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
| `num_states` | optional | Expected state dimension. If given, it is **validated** against the dimension the engine auto-detects — a mismatch fails CI |
| `smoke_gains` | yes | Gains that should at least run to completion. The keys pick the controller: `Kp/Ki/Kd` → PID family, `K1..Kn` → full-state feedback (FSF) |
| `default_scenario` | optional | `initial_condition_range`, `randomness_level`, `disturbance_level` |

### 3. Run the suite — that is the CI path

```bash
pytest tests/ -q
```

`tests/test_user_fixtures.py` discovers every registry entry and generates two
parametrized cases per fixture — no harness edits required:

- `test_user_provided_dynamics[<name>]` — smoke test: the plant simulates via
  `POST /silo/simulate` and returns well-formed metrics.
- `test_user_dynamics_num_states[<name>]` — asserts the engine-detected state
  dimension matches the declared `num_states` (skipped when not declared).

The same run feeds the progress report's `fixtures` category.
