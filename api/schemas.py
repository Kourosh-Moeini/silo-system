"""Pydantic request/response models for the introductory SiLo API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    module: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    module: str
    status: str
    error: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None


class SiloStartRequest(BaseModel):
    """Start a single-loop design job.

    `config` mirrors the keys accepted by `src.controllers_mock.initialize_state`
    / `run_optimization`. Extra keys are ignored by the mock path.
    """

    config: Dict[str, Any] = Field(default_factory=dict)
    control_objective: Optional[str] = None


class SiloSimulateRequest(BaseModel):
    """Manual gain re-simulation (time-response style)."""

    system_name: str = "ball_beam"
    gains: Dict[str, float] = Field(default_factory=dict)
    controller_type: str = "PID"
    scenario: Optional[Dict[str, Any]] = None
    custom_dynamics_path: Optional[str] = None
    dt: float = 0.01
    max_time: float = 5.0
    target: float = 0.0
    min_ctrl: float = -10.0
    max_ctrl: float = 10.0


class SimulateResponse(BaseModel):
    success: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    time_points: List[float] = Field(default_factory=list)
    trajectory: List[float] = Field(default_factory=list)
    control_signals: List[float] = Field(default_factory=list)
    errors: List[float] = Field(default_factory=list)
    message: Optional[str] = None


class CaseStudyInfo(BaseModel):
    name: str
    path: str
    num_states: Optional[int] = None
