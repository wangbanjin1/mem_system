from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.logging_utils import JsonLinesLogger
from ..core.models import now_utc_str
from ..storage.filestore import FileStore


class FileTaskQueue:
    """File-backed task queue with background worker and dead-letter logging (task_dead)."""

    def __init__(
        self,
        data_root: str | Path,
        handler: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_retries: int = 3,
        poll_interval_sec: float = 0.2,
        logger: Optional[JsonLinesLogger] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.queue_dir = self.data_root / "queue"
        self.pending_dir = self.queue_dir / "pending"
        self.done_dir = self.queue_dir / "done"
        self.dead_dir = self.queue_dir / "dead"
        for d in (self.pending_dir, self.done_dir, self.dead_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.handler = handler
        self.max_retries = int(max_retries)
        self.poll_interval_sec = float(poll_interval_sec)
        self.logger = logger or JsonLinesLogger()

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def enqueue(self, payload: Dict[str, Any], task_id: Optional[str] = None) -> str:
        tid = task_id or f"TASK_{uuid.uuid4().hex[:12].upper()}"
        task_obj = {
            "task_id": tid,
            "status": "pending",
            "attempts": 0,
            "created_at": now_utc_str(),
            "payload": payload,
        }
        with self._lock:
            FileStore.atomic_write_text(
                self.pending_dir / f"{tid}.json",
                json.dumps(task_obj, ensure_ascii=False, indent=2),
            )
        return tid

    def process_one(self, task_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if task_id:
                target = self.pending_dir / f"{task_id}.json"
                files = [target] if target.exists() else []
            else:
                files = sorted(self.pending_dir.glob("*.json"))
            if not files:
                return None
            task_file = files[0]
            try:
                task_obj = json.loads(task_file.read_text(encoding="utf-8"))
            except Exception:
                return None

        tid = task_obj["task_id"]
        task_obj["attempts"] = int(task_obj.get("attempts", 0)) + 1
        try:
            if self.handler is not None:
                self.handler(task_obj["payload"])
            task_obj["status"] = "done"
            task_obj["completed_at"] = now_utc_str()
            with self._lock:
                FileStore.atomic_write_text(
                    self.done_dir / f"{tid}.json",
                    json.dumps(task_obj, ensure_ascii=False, indent=2),
                )
                if task_file.exists():
                    task_file.unlink()
            return task_obj
        except Exception as exc:
            task_obj["last_error"] = str(exc)
            if task_obj["attempts"] >= self.max_retries:
                task_obj["status"] = "dead"
                task_obj["failed_at"] = now_utc_str()
                self.logger.log("task_dead", task_id=tid, attempts=task_obj["attempts"], error=str(exc))
                with self._lock:
                    FileStore.atomic_write_text(
                        self.dead_dir / f"{tid}.json",
                        json.dumps(task_obj, ensure_ascii=False, indent=2),
                    )
                    if task_file.exists():
                        task_file.unlink()
            else:
                with self._lock:
                    FileStore.atomic_write_text(
                        task_file,
                        json.dumps(task_obj, ensure_ascii=False, indent=2),
                    )
            return task_obj

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        for folder in (self.done_dir, self.pending_dir, self.dead_dir):
            f = folder / f"{task_id}.json"
            if f.exists():
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return {"task_id": task_id, "status": "not_found"}

    def start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="memory-queue-worker")
        self._worker_thread.start()

    def stop_worker(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            processed = self.process_one()
            if processed is None:
                time.sleep(self.poll_interval_sec)
