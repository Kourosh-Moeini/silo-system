"""Introductory SiLo API — FastAPI entrypoint (no auth, no DB, no .env).

Run from repo root:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

# LangGraph pulls LangChain serializers that emit a pending-deprecation warning
# about `allowed_objects` default. Harmless for this intro; silence it.
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
)
# LangChain uses a custom warning class in some versions
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.job_store import job_store
from api.schemas import (
    CaseStudyInfo,
    JobResponse,
    JobStatusResponse,
    SiloSimulateRequest,
    SiloStartRequest,
    SimulateResponse,
)
from api import silo_service

app = FastAPI(
    title="LabCD SiLo (introductory)",
    description=(
        "Minimal single-loop design API backed by mock LLM agents. "
        "No external API keys or database required."
    ),
    version="0.1.0-intro",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "module": "silo-intro"}


@app.get("/case-studies", response_model=list[CaseStudyInfo])
def case_studies() -> list:
    return silo_service.list_case_studies()


@app.post("/silo/start", response_model=JobResponse)
def start_silo(request: SiloStartRequest) -> JobResponse:
    job_id = silo_service.start_silo_job(
        request.config or {},
        control_objective=request.control_objective or "",
    )
    job = job_store.get(job_id)
    return JobResponse(job_id=job_id, module=job.module, status=job.status.value)


@app.get("/silo/{job_id}", response_model=JobStatusResponse)
def silo_status(job_id: str) -> JobStatusResponse:
    try:
        data = silo_service.get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JobStatusResponse(
        job_id=data["job_id"],
        module=data["module"],
        status=data["status"],
        error=data.get("error"),
        result_summary=data.get("result_summary"),
    )


@app.get("/silo/{job_id}/detail")
def silo_detail(job_id: str) -> Dict[str, Any]:
    try:
        return silo_service.get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.post("/silo/{job_id}/cancel", response_model=JobResponse)
def cancel_silo(job_id: str) -> JobResponse:
    try:
        job = job_store.request_cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JobResponse(job_id=job.id, module=job.module, status=job.status.value)


@app.post("/silo/simulate", response_model=SimulateResponse)
def simulate(request: SiloSimulateRequest) -> SimulateResponse:
    raw = silo_service.simulate_silo_response(request.model_dump())
    return SimulateResponse(
        success=bool(raw.get("success", False)),
        metrics=raw.get("metrics") or {},
        time_points=raw.get("time_points") or [],
        trajectory=raw.get("trajectory") or [],
        control_signals=raw.get("control_signals") or [],
        errors=raw.get("errors") or [],
        message=raw.get("message"),
    )
