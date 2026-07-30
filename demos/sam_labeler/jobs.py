"""Background jobs with pollable progress.

Loading SAM downloads several gigabytes and CPU inference can take a minute, so
neither can happen inside a request: the browser would just sit there with no
feedback, which is exactly how it looked. Both now run on a worker thread and the
page polls :class:`Job` for a stage, an elapsed time and, while downloading, the
number of bytes that have landed in the Hugging Face cache.

There is no total to divide by — the hub does not tell us up front — so the bar is
honest about that: it reports megabytes and rate rather than a fake percentage.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

IDLE = "idle"
RUNNING = "running"
DONE = "done"
ERROR = "error"


def hf_cache_bytes(model_id: str) -> int:
    """Bytes currently cached for ``model_id``, including partial downloads.

    Counting the cache directory is a cheap way to show real progress without
    hooking into the hub's internals; ``*.incomplete`` blobs are included, which is
    what makes the number move while a download is in flight.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        root = Path(HF_HUB_CACHE)
    except Exception:
        root = Path.home() / ".cache" / "huggingface" / "hub"

    repo_dir = root / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return 0
    total = 0
    for path in repo_dir.rglob("*"):
        try:
            if path.is_file() or path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue  # a blob being written can vanish mid-walk
    return total


@dataclass
class Job:
    """One long-running operation, safe to read from a request thread."""

    kind: str = ""
    state: str = IDLE
    stage: str = ""
    detail: str = ""
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    bytes_done: int = 0
    bytes_rate: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.state == RUNNING

    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.monotonic()
        return max(0.0, end - self.started_at)

    def start(
        self,
        kind: str,
        work: Callable[["Job"], None],
        *,
        stage: str = "starting",
    ) -> bool:
        """Run ``work`` on a thread. Returns False if a job is already running."""
        with self._lock:
            if self.state == RUNNING:
                return False
            self.kind = kind
            self.state = RUNNING
            self.stage = stage
            self.detail = ""
            self.error = None
            self.started_at = time.monotonic()
            self.finished_at = 0.0
            self.bytes_done = 0
            self.bytes_rate = 0.0

        def runner() -> None:
            try:
                work(self)
            except Exception as failure:  # a worker must never take the server down
                self.fail(str(failure).splitlines()[0][:200] or type(failure).__name__)
            else:
                with self._lock:
                    if self.state == RUNNING:
                        self.state = DONE
                        self.stage = "done"
                        self.finished_at = time.monotonic()

        self._thread = threading.Thread(target=runner, name=f"sam-{kind}", daemon=True)
        self._thread.start()
        return True

    def set_stage(self, stage: str, detail: str = "") -> None:
        with self._lock:
            self.stage = stage
            if detail:
                self.detail = detail

    def note_bytes(self, total: int) -> None:
        """Record cache growth, deriving a rate for the UI."""
        with self._lock:
            elapsed = max(self.elapsed(), 1e-6)
            self.bytes_done = total
            self.bytes_rate = total / elapsed

    def fail(self, message: str) -> None:
        with self._lock:
            self.state = ERROR
            self.stage = "failed"
            self.error = message
            self.finished_at = time.monotonic()

    def to_json(self) -> dict[str, Any]:
        megabytes = self.bytes_done / 1_048_576
        return {
            "kind": self.kind,
            "state": self.state,
            "stage": self.stage,
            "detail": self.detail,
            "error": self.error,
            "running": self.running,
            "elapsed": round(self.elapsed(), 1),
            "megabytes": round(megabytes, 1),
            "rate": round(self.bytes_rate / 1_048_576, 2),
        }


def watch_cache(job: Job, model_id: str, stop: threading.Event, interval: float = 1.0) -> threading.Thread:
    """Poll the HF cache while a load runs, so the page can show real progress."""

    def poll() -> None:
        baseline = hf_cache_bytes(model_id)
        while not stop.wait(interval):
            total = hf_cache_bytes(model_id)
            job.note_bytes(max(0, total))
            if total > baseline:
                job.set_stage("downloading")

    thread = threading.Thread(target=poll, name="sam-cache-watch", daemon=True)
    thread.start()
    return thread
