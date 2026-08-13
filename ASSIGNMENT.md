# Automated Benchmark Testing — Take-Home Exercise

**Project:** SiLo (single-loop) module  
**Time we expect:** about 6–10 hours  
**Please send it back within:** 72 hours of receiving it  
**Stack:** Python 3.11+ · FastAPI · pytest  

---

## Why this exercise

We are developing an AI-assisted **control-system design** platform. Users upload plant dynamics (Python or MATLAB), then run design pipelines driven by LLM agents that propose controller structures and gains, simulate closed-loop behaviour, and iterate.

**SiLo** (single-loop) is the first pipeline we ship. Later we will add MPC, system identification, neuroadaptive control, and multi-loop (MuLo) modules — all of which also use LLM agents and accept arbitrary user dynamics. The person who owns automated testing will own those modules as well.

Because:

1. LLM agent actions are stochastic,
2. users can upload arbitrary dynamics and scenario settings,

we need a **thorough, repeatable automated benchmark suite** that covers randomness, edge cases, and regression over time — plus a lightweight way to **report progress** (pass rates, metric distributions, regressions) as the product evolves.

This exercise is intentionally **introductory**: LLM calls are mocked (no API keys, no `.env`), the API is a thin in-memory FastAPI surface, and the physics/engine code is provided. Your job is the testing layer and the reporting workflow, not re-implementing control theory or agents.

---

## What you’ll receive

```
silo-intro/
|-- api/                     # thin FastAPI surface over the mock SiLo graph
|   |-- __init__.py
|   |-- main.py              # uvicorn entry: api.main:app
|   |-- schemas.py           # Pydantic request/response models
|   |-- job_store.py         # in-memory jobs (no DB)
|   `-- silo_service.py      # start job / status / simulate adapters
|
|-- src/                     # SiLo core (treat as read-only unless a bug blocks you)
|   |-- controllers_mock.py  # LangGraph workflow with mock LLM agents
|   |-- llm_agents_mock.py   # deterministic-ish random gains (seeded)
|   |-- simulation.py        # SimulationRunner + metrics
|   |-- systems.py           # GeneralDynamicalSystem, CustomDynamicalSystem, create_system factory
|   `-- utils.py
|
|-- case_studies/            # plant models for end-to-end tests
|   |-- BallBeam.py
|   |-- DCMotor.py
|   |-- InvPendulum.py
|   `-- py/                  # additional plants (active_suspension, cstr, …)
|
|-- requirements.txt
`-- README.md
```

**Quick orientation**

1. Install and run the API (no secrets required):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

   OpenAPI: http://localhost:8000/docs  

2. Core library imports (same ones the API uses):

   ```python
   from src.controllers_mock import run_optimization, initialize_state
   from src.simulation import SimulationRunner
   from src.systems import create_system, CustomDynamicalSystem
   from api.silo_service import start_silo_job, get_job_status, simulate_silo_response
   ```

---

## What we’d like you to build

An **automated benchmark testing** layer for the SiLo module, plus a small **progress-reporting** workflow. Aim for something the rest of the team can run in CI and that you can extend when MPC / SysID / Neuroadaptive modules arrive.

### 1. Test scripts (core)

Create a `tests/` (or `benchmarks/`) package that covers at least:

| Area | What to exercise |
|------|------------------|
| **Unit / pure** | Metrics calculation, parameter validation, schema defaults, job store lifecycle |
| **Simulate path** | `POST /silo/simulate` (or `simulate_silo_response`) for built-in plants (`ball_beam`, `dc_motor`, `inverted_pendulum`) and at least 2–3 files from `case_studies/py/` via `custom_dynamics_path` |
| **Design path** | `POST /silo/start` → poll `GET /silo/{job_id}` until `completed` / `failed`; assert structure of `result_summary` |
| **Randomness** | Same config, different seeds; same seed, repeated runs — document variance of metrics / success rates |
| **Scenario / user input variability** | Vary `initial_condition_range`, `randomness_level`, `disturbance_level`, controller type (`P`/`PI`/`PD`/`PID`/`FSF` where supported), gain ranges, `dt` / `max_time` |
| **Failure modes** | Unknown system name, missing gains, path that doesn’t exist, cancel mid-job |

Prefer **pytest**. Parametrize over case studies and scenario matrices rather than one giant script. Keep tests hermetic (no network, no real LLM keys).

### 2. Workflow to add user inputs to the suite

Users will upload arbitrary `.py` dynamics. We need a clear path so a new plant can become a regression fixture without rewriting the harness.

Suggested shape (you may refine):

- A directory such as `tests/fixtures/user_dynamics/` (or extend `case_studies/`).
- A small registry (YAML/JSON/Python) listing: path, expected `num_states` (if known), default scenario, “smoke” gains that should at least run.
- A helper that discovers fixtures and parametrizes pytest cases.
- Document how a teammate drops a new file in and gets it into CI.

### 3. Performing the tests & progress report

- A single entry command, e.g. `pytest tests/ -q` or `python -m benchmarks run`.
- A **progress report** artifact (Markdown and/or JSON) that summarizes:
  - pass / fail counts by category (unit, simulate, design, fixtures),
  - metric distributions (mse, settling time, stable rate, …) across plants and seeds,
  - comparison vs a previous run if a baseline file is present (optional but valued),
  - wall time and any flaky tests.

This report is what we will look at periodically to see whether the website/product is improving as modules and agents change. Keep the format stable so it can be diffed or archived.

### 4. Layout we expect after your work

```
silo-intro/
|-- api/                 # provided
|-- src/                 # provided (prefer not to rewrite)
|-- case_studies/        # provided
|-- tests/               # or benchmarks/ — your suite
|   |-- ...
|   `-- fixtures/        # user-dynamics registry + samples
|-- reports/             # generated progress reports (gitignored except examples)
|-- requirements.txt     # add any extra test deps you need
|-- README.md            # how to run tests + generate a report
`-- ASSIGNMENT.md        # this file
```

You may add a thin `Makefile` or `scripts/run_benchmarks.sh` if it helps.

---

## API surface (for orientation)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Liveness |
| `GET` | `/case-studies` | List bundled plants |
| `POST` | `/silo/start` | Start mock design job → `{job_id, module, status}` |
| `GET` | `/silo/{job_id}` | Status + optional `result_summary` |
| `GET` | `/silo/{job_id}/detail` | Fuller job snapshot |
| `POST` | `/silo/{job_id}/cancel` | Request cancel |
| `POST` | `/silo/simulate` | Fixed-gain closed-loop sim (no LLM) |

Illustrative bodies:

`POST /silo/start`

```json
{
  "config": {
    "system_name": "ball_beam",
    "seed": 42,
    "max_scenarios": 2,
    "max_iter": 5,
    "controllers": ["PID"],
    "dt": 0.01,
    "max_time": 5.0
  },
  "control_objective": "stabilize ball position at 0"
}
```

`POST /silo/simulate`

```json
{
  "system_name": "ball_beam",
  "controller_type": "PID",
  "gains": { "Kp": 5.0, "Ki": 0.1, "Kd": 1.0 },
  "scenario": {
    "initial_condition_range": [-0.5, 0.5],
    "randomness_level": 0.0,
    "disturbance_level": 0.0
  },
  "dt": 0.01,
  "max_time": 5.0
}
```

You can also call `src` / `api.silo_service` directly from tests without HTTP if you prefer; both are acceptable. HTTP tests (e.g. via `httpx.AsyncClient` + FastAPI `TestClient`) demonstrate the surface the website uses.

---

## What we’re looking for

| Area | We care about |
|------|----------------|
| Coverage design | Thoughtful matrix over plants, seeds, scenarios, controller types — not only the happy path |
| Randomness handling | Explicit strategy for LLM/mock and IC randomness (seeds, statistical assertions, or distributional checks) |
| Extensibility | Clear path to register new user dynamics and, later, new modules (MPC, SysID, …) |
| Reporting | Stable, human-readable progress report that can be produced on every run |
| Code quality | Readable pytest layout, minimal duplication, no hard-coded secrets |
| Judgment | Reasonable defaults where the brief is ambiguous; note assumptions in the README |

You do **not** need control-theory depth beyond reading the returned metrics. Treat the engine as a black box that maps (plant, gains, scenario) → trajectory + metrics, and the design graph as a black box that maps config → job result.

---

## Optional extras (only if you have time left)

- Baseline file + simple regression gate (fail CI if pass rate drops > X% or median mse worsens beyond a threshold).
- Property-based tests (e.g. Hypothesis) for metric invariants.
- A tiny HTML or Markdown dashboard generated from the report JSON.
- Parallel pytest (`-n auto`) with notes on what is safe to parallelize given in-memory job state.

None of these are required for a solid submission.

---

## How to send it back

Push the work to a GitHub repo (public or private with access for us). Keep the provided `api/`, `src/`, and `case_studies/` behaviour intact unless you fix a clear bug (call it out in the README). Add your `tests/` (or `benchmarks/`), report tooling, and an updated `README.md` with:

- how to install,
- how to run the full suite,
- how to generate the progress report,
- how to add a new user dynamics fixture,
- any assumptions or known limitations.

We’ll review by:

```bash
pip install -r requirements.txt
uvicorn api.main:app --port 8000   # sanity
pytest tests/ -q                   # or your documented command
# and by reading the generated report
```

---

## Context for the broader role

Once this introductory suite is in place, the same owner will:

- extend coverage as **MPC, SysID, Neuroadaptive, MuLo**, etc. land,
- keep the suite honest against real (non-mock) LLM providers in staging,
- produce **periodic progress reports** that product and research can use to see whether the website’s design quality is improving.

This exercise is the foundation of that workflow.

---

Thanks for taking the time. Looking forward to seeing how you’d approach automated benchmarking with us.
