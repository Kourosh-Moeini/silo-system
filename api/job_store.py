"""In-memory job tracking for long-running SiLo workflows (introductory, no DB)."""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    module: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cancel_requested: bool = False
    thread: Optional[threading.Thread] = field(default=None, repr=False)

    def touch(self, status: Optional[JobStatus] = None) -> None:
        if status is not None:
            self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, module: str, metadata: Optional[Dict[str, Any]] = None) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, module=module, metadata=metadata or {})
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def request_cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            raise ValueError(f"Job {job_id} cannot be cancelled (status: {job.status.value})")
        job.cancel_requested = True
        monitor = job.metadata.get("monitor")
        if monitor is not None and hasattr(monitor, "is_running"):
            monitor.is_running = False
        job.touch()
        return job


job_store = JobStore()
