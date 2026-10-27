from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .models import now_utc_str

_LOG_LOCK = threading.RLock()


class JsonLinesLogger:
    """Append-only structured JSONL logger for observability and quality alerts."""

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path

    def log(self, event_name: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts": now_utc_str(),
            "event": event_name,
            **fields,
        }
        if self.log_path is not None:
            with _LOG_LOCK:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
