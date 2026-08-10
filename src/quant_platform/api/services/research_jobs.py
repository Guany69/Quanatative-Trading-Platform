"""Durable FIFO research queue with exactly one active persistence owner."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from quant_platform.api.config import ApiSettings
from quant_platform.research.results import ResultsStore
from quant_platform.research.runner import (
    ExperimentRunOrchestrator,
    ResearchRunRequest,
)

logger = logging.getLogger("quant_platform.api.jobs")

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED"})
JOB_STATES = frozenset({"QUEUED", "RUNNING", *TERMINAL_STATES})


class OrchestratorProtocol(Protocol):
    def run(self, request: ResearchRunRequest, **kwargs: Any) -> Any: ...


@dataclass
class ResearchJob:
    run_id: str
    state: str
    request: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    stage: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None


class ResearchRunJobManager:
    """Own one worker thread and serialize every research-store mutation."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        orchestrator: OrchestratorProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.orchestrator = orchestrator or ExperimentRunOrchestrator(
            results_db=settings.results_db,
            snapshot_root=settings.snapshot_root,
            cache_root=settings.cache_root,
            artifact_root=settings.artifact_root,
            report_root=settings.report_root,
            registry_path=settings.registry_path,
        )
        self.writer_lock = threading.RLock()
        self._jobs: dict[str, ResearchJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._state_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        """Recover stale state once at the application lifecycle boundary."""
        with self._state_lock:
            if self._started:
                return
            self._load()
            now = datetime.now(UTC).isoformat()
            for job in self._jobs.values():
                if job.state == "RUNNING":
                    job.state = "FAILED"
                    job.completed_at = now
                    job.updated_at = now
                    job.failure_code = "STALE_RUN_RECOVERY"
                    job.failure_message = "API process exited before the research run completed."
                elif job.state == "QUEUED":
                    self._queue.put(job.run_id)
            self._persist()
            with self.writer_lock, ResultsStore(self.settings.results_db) as store:
                store.recover_stale_runs("API process exited before completion")
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="quant-platform-research-writer",
                daemon=True,
            )
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def submit(self, request: ResearchRunRequest) -> ResearchJob:
        if not self._started:
            raise RuntimeError("research job manager has not started")
        now = datetime.now(UTC).isoformat()
        run_id = uuid.uuid4().hex[:16]
        job = ResearchJob(
            run_id=run_id,
            state="QUEUED",
            request=self._serialize_request(request),
            created_at=now,
            updated_at=now,
        )
        with self._state_lock:
            self._jobs[run_id] = job
            self._persist()
            self._queue.put(run_id)
        return ResearchJob(**asdict(job))

    def get(self, run_id: str) -> ResearchJob | None:
        with self._state_lock:
            job = self._jobs.get(run_id)
            return ResearchJob(**asdict(job)) if job else None

    def list(self) -> list[ResearchJob]:
        with self._state_lock:
            return [ResearchJob(**asdict(job)) for job in self._jobs.values()]

    def run_writer_operation(self, operation: Any) -> Any:
        """Serialize other audited DuckDB mutations (currently holdout consumption)."""
        with self.writer_lock:
            return operation()

    def wait_for_idle(self, timeout: float = 10.0) -> bool:
        """Test/maintenance helper; never used to block an HTTP research request."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._state_lock:
                active = any(job.state in {"QUEUED", "RUNNING"} for job in self._jobs.values())
            if not active:
                return True
            time.sleep(0.01)
        return False

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                run_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            job = self.get(run_id)
            if job is None or job.state != "QUEUED":
                self._queue.task_done()
                continue
            self._update(run_id, state="RUNNING", started_at=datetime.now(UTC).isoformat())

            def report_progress(stage: str, job_id: str = run_id) -> None:
                self._update(job_id, stage=stage)

            try:
                with self.writer_lock:
                    self.orchestrator.run(
                        self._deserialize_request(job.request),
                        run_id=run_id,
                        created_at=datetime.fromisoformat(job.created_at),
                        progress=report_progress,
                        recover_stale=False,
                    )
                self._update(
                    run_id,
                    state="COMPLETED",
                    completed_at=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                logger.exception("research job %s failed", run_id)
                self._update(
                    run_id,
                    state="FAILED",
                    completed_at=datetime.now(UTC).isoformat(),
                    failure_code=type(exc).__name__.upper(),
                    failure_message=str(exc),
                )
            finally:
                self._queue.task_done()

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._state_lock:
            job = self._jobs[run_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(UTC).isoformat()
            self._persist()

    def _load(self) -> None:
        path = self.settings.job_state_path
        if not path.exists():
            self._jobs = {}
            return
        try:
            payload = json.loads(path.read_text())
            jobs = [ResearchJob(**item) for item in payload.get("jobs", [])]
            self._jobs = {job.run_id: job for job in jobs if job.state in JOB_STATES}
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"invalid research job journal at {path}: {exc}") from exc

    def _persist(self) -> None:
        path = self.settings.job_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {"version": 1, "jobs": [asdict(job) for job in self._jobs.values()]}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    @staticmethod
    def _serialize_request(request: ResearchRunRequest) -> dict[str, Any]:
        return {
            "charter_path": str(request.charter_path),
            "snapshot_id": request.snapshot_id,
            "models": list(request.models),
            "strategies": list(request.strategies),
            "scenarios": list(request.scenarios),
            "portfolio_model": request.portfolio_model,
            "fixture_securities": request.fixture_securities,
            "max_workers": request.max_workers,
        }

    @staticmethod
    def _deserialize_request(payload: dict[str, Any]) -> ResearchRunRequest:
        return ResearchRunRequest(
            charter_path=payload["charter_path"],
            snapshot_id=payload.get("snapshot_id"),
            models=tuple(payload["models"]),
            strategies=tuple(payload["strategies"]),
            scenarios=tuple(payload["scenarios"]),
            portfolio_model=payload["portfolio_model"],
            fixture_securities=int(payload["fixture_securities"]),
            max_workers=payload.get("max_workers"),
        )
