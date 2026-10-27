from __future__ import annotations

import copy
import re
from typing import Any, Dict, Set


class CaseSanitizer:
    """Strips user/session identity fields before writing Agent experience nodes (§4 Pitfall 11)."""

    STRIP_KEYS: Set[str] = {
        "tenant_id",
        "user_id",
        "session_id",
        "source_session_id",
        "owner_id",
        "raw",
        "raw_text",
        " raw_utterance",
        "phone",
        "email",
        "ip",
    }

    _USER_ID_TOKEN_RE = re.compile(r"\b(?:USER_\w+|U\d{3,}|tenant_\w+|sess_\w+|SESSION_\w+)\b", re.IGNORECASE)

    @classmethod
    def sanitize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            if k in cls.STRIP_KEYS:
                continue
            if isinstance(v, dict):
                cleaned[k] = cls.sanitize(v)
            elif isinstance(v, list):
                cleaned[k] = [
                    cls.sanitize(item)
                    if isinstance(item, dict)
                    else (cls._USER_ID_TOKEN_RE.sub("[REDACTED]", item) if isinstance(item, str) else copy.deepcopy(item))
                    for item in v
                ]
            elif isinstance(v, str):
                cleaned[k] = cls._USER_ID_TOKEN_RE.sub("[REDACTED]", v)
            else:
                cleaned[k] = copy.deepcopy(v)
        return cleaned
