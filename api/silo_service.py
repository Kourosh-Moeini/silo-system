"""SiLo HTTP service adapter over the introductory mock agents.

Uses `src.controllers_mock` and related modules. No LLM API keys, no database.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from api.job_store import JobStatus, job_store
from src.controllers_mock import initialize_state, run_optimization
from src.simulation import SimulationRunner
from src.systems import create_system, CustomDynamicalSystem


class DesignMonitor:
    """Lightweight monitor compatible with the mock graph progress hooks."""

    def __init__(self) -> None:
        self.progress_history: list = []
        self.state_history: list = []
        self.llm_responses: list = []
        self.current_state: Dict[str, Any] = {}
        self.is_running: bool = False
        self.revision: int = 0
        self.scenario_metrics_history: list = []
        self.final_report: Optional[Dict[str, Any]] = None

    def add_progress(self, message: str, data: Optional[Dict] = None) -> None:
        self.progress_history.append({"message": message, "data": data or {}})
        self.revision += 1

    def add_llm_response(self, agent_name: str, prompt: str, response: str) -> None:
        self.llm_responses.append(
            {"agent": agent_name, "prompt": prompt[:200], "response": response}
        )
        self.revision += 1

    def update_state(self, update: Dict) -> None:
        self.current_state.update(update)
        self.revision += 1

    def add_scenario_metrics(self, scenario_level: int, metrics: Dict) -> None:
        self.scenario_metrics_history.append(
            {"scenario_level": scenario_level, "metrics": metrics}
        )
        self.revision += 1


def _default_config() -> Dict[str, Any]:
    return {
        "llm_model": "mock",
        "seed": 42,
        "system_name": "ball_beam",
        "max_scenarios": 2,
        "max_iter": 5,
        "controllers": ["PID"],
        "dt": 0.01,
        "max_time": 5.0,
        "target": 0.0,
        "min_ctrl": -10.0,
        "max_ctrl": 10.0,
        "num_inputs": 1,
        "input_channel": 0,
        "output_channel": 0,
    }


def start_silo_job(config: Dict[str, Any], control_objective: str = "") -> str:
    """Create an in-memory job and run the mock optimization graph in a thread."""
    cfg = _default_config()
    cfg.update(config or {})
    if control_objective:
        cfg["control_objective"] = control_objective

    monitor = DesignMonitor()
    job = job_store.create(module="silo", metadata={"config": cfg, "monitor": monitor})

    def _worker() -> None:
        job.touch(JobStatus.RUNNING)
        monitor.is_running = True
        try:
            result = run_optimization(
                llm_model=cfg.get("llm_model", "mock"),
                run_id=1,
                seed=int(cfg.get("seed", 42)),
                system_name=cfg.get("system_name", "ball_beam"),
                max_scenarios=int(cfg.get("max_scenarios", 2)),
                max_iter=int(cfg.get("max_iter", 5)),
                controllers=cfg.get("controllers"),
                custom_scenarios=cfg.get("custom_scenarios"),
                param_ranges=cfg.get("param_ranges"),
                target_metrics=cfg.get("target_metrics"),
                custom_dynamics_path=cfg.get("custom_dynamics_path"),
                file_type=cfg.get("file_type", "Python (.py)"),
                dt=float(cfg.get("dt", 0.01)),
                max_time=float(cfg.get("max_time", 5.0)),
                target=float(cfg.get("target", 0.0)),
                num_inputs=int(cfg.get("num_inputs", 1)),
                input_channel=int(cfg.get("input_channel", 0)),
                output_channel=int(cfg.get("output_channel", 0)),
                min_ctrl=float(cfg.get("min_ctrl", -10.0)),
                max_ctrl=float(cfg.get("max_ctrl", 10.0)),
                monitor=monitor,
            )
            job.metadata["result"] = _summarize_result(result, monitor)
            job.touch(JobStatus.COMPLETED)
        except Exception as exc:  # noqa: BLE001 — surface full error to job
            job.error = f"{type(exc).__name__}: {exc}"
            job.metadata["traceback"] = traceback.format_exc()
            job.touch(JobStatus.FAILED)
        finally:
            monitor.is_running = False

    t = threading.Thread(target=_worker, daemon=True)
    job.thread = t
    t.start()
    return job.id


def _summarize_result(result: Any, monitor: DesignMonitor) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "progress_count": len(monitor.progress_history),
        "llm_call_count": len(monitor.llm_responses),
        "scenario_metrics": monitor.scenario_metrics_history,
    }
    if isinstance(result, dict):
        # Keep only JSON-friendly bits
        for key in ("best_params", "best_metrics", "controller_type", "success", "report"):
            if key in result:
                val = result[key]
                if isinstance(val, (np.ndarray,)):
                    summary[key] = val.tolist()
                else:
                    summary[key] = val
    if monitor.final_report:
        summary["final_report"] = monitor.final_report
    return summary


def get_job_status(job_id: str) -> Dict[str, Any]:
    job = job_store.get(job_id)
    out: Dict[str, Any] = {
        "job_id": job.id,
        "module": job.module,
        "status": job.status.value,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if "result" in job.metadata:
        out["result_summary"] = job.metadata["result"]
    monitor = job.metadata.get("monitor")
    if monitor is not None:
        out["progress_tail"] = monitor.progress_history[-5:]
    return out


def _apply_runtime_config(system, dt, max_time, target, min_ctrl, max_ctrl) -> None:
    """Ensure control limits and sim settings exist even if a plant skips super().__init__."""
    system.dt = dt
    system.max_time = max_time
    system.target = target
    system.min_control = min_ctrl
    system.max_control = max_ctrl
    if not hasattr(system, "num_inputs"):
        system.num_inputs = 1
    if not hasattr(system, "output_channel"):
        system.output_channel = 0


def simulate_silo_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single closed-loop simulation with fixed gains (no LLM)."""
    system_name = body.get("system_name", "ball_beam")
    gains = body.get("gains") or {}
    custom_path = body.get("custom_dynamics_path")
    dt = float(body.get("dt", 0.01))
    max_time = float(body.get("max_time", 5.0))
    target = float(body.get("target", 0.0))
    min_ctrl = float(body.get("min_ctrl", -10.0))
    max_ctrl = float(body.get("max_ctrl", 10.0))
    scenario = body.get("scenario") or {
        "initial_condition_range": (-1.0, 1.0),
        "randomness_level": 0.0,
        "disturbance_level": 0.0,
    }

    try:
        if custom_path:
            system = CustomDynamicalSystem(custom_path, scenario)
        else:
            system = create_system(system_name or "ball_beam", scenario=scenario)

        runner = SimulationRunner(type(system))
        runner.system = system
        _apply_runtime_config(system, dt, max_time, target, min_ctrl, max_ctrl)
        result = runner.evaluate_parameters(gains)

        if not isinstance(result, dict):
            result = {
                "success": False,
                "metrics": {},
                "trajectory": [],
                "control_signals": [],
                "errors": [],
                "message": "Unexpected simulation return type.",
            }
        for key in ("trajectory", "control_signals", "errors", "time_points"):
            if key in result and hasattr(result[key], "tolist"):
                result[key] = result[key].tolist()
        if "error" in result and not result.get("success", True):
            result["message"] = result.pop("error")
        result.setdefault("success", bool(result.get("metrics")))
        if result.get("trajectory") and not result.get("time_points"):
            n = len(result["trajectory"])
            result["time_points"] = (np.arange(n) * dt).tolist()
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "metrics": {},
            "time_points": [],
            "trajectory": [],
            "control_signals": [],
            "errors": [],
            "message": f"{type(exc).__name__}: {exc}",
        }


def list_case_studies(base: Optional[Path] = None) -> list:
    root = base or Path(__file__).resolve().parents[1] / "case_studies"
    items = []
    for p in sorted(root.rglob("*.py")):
        if p.name.startswith("_"):
            continue
        items.append(
            {
                "name": p.stem,
                "path": str(p.relative_to(root.parent)),
            }
        )
    return items
