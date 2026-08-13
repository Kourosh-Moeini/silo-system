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
