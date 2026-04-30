"""In-memory job registry with progress tracking and background execution."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class JobProgress:
    stage: str = "queued"          # "queued" | "running" | "done" | "error"
    message: str = ""              # human-readable status line (e.g. "Transcribing…")
    current: int = 0               # progress counter (e.g. current scene index)
    total: int = 0                 # total expected work units (0 if unknown)


@dataclass
class Job:
    id: str
    kind: str                      # "prompts" | "video"
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "stage": self.progress.stage,
                "message": self.progress.message,
                "current": self.progress.current,
                "total": self.progress.total,
                "result": self.result,
                "error": self.error,
            }

    def append_log(self, line: str) -> None:
        text = str(line).rstrip()
        if not text:
            return
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {text}"
        with self._lock:
            self.logs.append(entry)
            # Keep memory bounded for long-running ffmpeg output.
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def logs_since(self, offset: int = 0) -> tuple[list[str], int]:
        with self._lock:
            safe_offset = max(0, min(int(offset), len(self.logs)))
            return list(self.logs[safe_offset:]), len(self.logs)

    def request_cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] Cancellation requested")
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self.cancel_requested

    def raise_if_cancelled(self) -> None:
        if self.is_cancel_requested():
            raise RuntimeError("Job cancelled by user")

    def update(
        self,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        with self._lock:
            if stage is not None:
                self.progress.stage = stage
            if message is not None:
                self.progress.message = message
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
                if len(self.logs) > 2000:
                    self.logs = self.logs[-2000:]
            if current is not None:
                self.progress.current = current
            if total is not None:
                self.progress.total = total


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._cleanup_tasks: Dict[str, Dict[str, Any]] = {}
        self._cleanup_lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def create(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def run(
        self,
        job: Job,
        target: Callable[[Job], Dict[str, Any]],
        on_finished: Optional[Callable[[Job], None]] = None,
    ) -> None:
        """Run ``target(job)`` in a background thread. ``target`` should
        return a dict that becomes ``job.result``. Any raised exception is
        captured into ``job.error`` with the stage set to ``error``."""

        def _worker() -> None:
            job.update(stage="running", message="Starting…")
            try:
                result = target(job)
                if job.is_cancel_requested():
                    with job._lock:
                        job.progress.stage = "cancelled"
                        job.progress.message = "Cancelled"
                        job.logs.append(f"[{time.strftime('%H:%M:%S')}] Cancelled")
                        if len(job.logs) > 2000:
                            job.logs = job.logs[-2000:]
                    return
                with job._lock:
                    job.result = result or {}
                    job.progress.stage = "done"
                    if not job.progress.message or job.progress.stage == "running":
                        job.progress.message = "Complete"
                    job.logs.append(f"[{time.strftime('%H:%M:%S')}] Complete")
                    if len(job.logs) > 2000:
                        job.logs = job.logs[-2000:]
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                tb = traceback.format_exc()
                if job.is_cancel_requested():
                    with job._lock:
                        job.progress.stage = "cancelled"
                        job.progress.message = "Cancelled"
                        job.logs.append(f"[{time.strftime('%H:%M:%S')}] Cancelled")
                        if len(job.logs) > 2000:
                            job.logs = job.logs[-2000:]
                    return
                with job._lock:
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.progress.stage = "error"
                    job.progress.message = job.error
                    job.logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {job.error}")
                    for line in tb.strip().splitlines():
                        job.logs.append(f"[{time.strftime('%H:%M:%S')}] {line}")
                    if len(job.logs) > 2000:
                        job.logs = job.logs[-2000:]
            finally:
                if on_finished is not None:
                    try:
                        on_finished(job)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def schedule_cleanup(
        self,
        *,
        job_id: str,
        delay_seconds: int,
        paths: Optional[list[str]] = None,
        delete_job: bool = True,
    ) -> None:
        due_at = time.time() + max(1, int(delay_seconds))
        with self._cleanup_lock:
            self._cleanup_tasks[job_id] = {
                "due_at": due_at,
                "paths": list(paths or []),
                "delete_job": bool(delete_job),
            }

    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(5)
            now = time.time()
            due_ids: list[str] = []
            with self._cleanup_lock:
                for jid, task in self._cleanup_tasks.items():
                    if float(task.get("due_at", now + 1)) <= now:
                        due_ids.append(jid)

            for jid in due_ids:
                task: Optional[Dict[str, Any]] = None
                with self._cleanup_lock:
                    task = self._cleanup_tasks.pop(jid, None)
                if not task:
                    continue

                for p in task.get("paths", []):
                    try:
                        path = Path(str(p))
                        if path.is_dir():
                            shutil.rmtree(path, ignore_errors=True)
                        elif path.is_file():
                            path.unlink(missing_ok=True)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()

                if task.get("delete_job", True):
                    with self._lock:
                        self._jobs.pop(jid, None)


registry = JobRegistry()
