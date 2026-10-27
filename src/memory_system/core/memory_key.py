from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Union


SEPARATOR = " | "
MISSING_TOKEN = "_"


def _extract_fields(spec_or_fields: Any) -> List[str]:
    if hasattr(spec_or_fields, "memory_key_fields"):
        return list(spec_or_fields.memory_key_fields)
    if isinstance(spec_or_fields, (list, tuple)):
        return [str(f) for f in spec_or_fields]
    raise TypeError(f"Unsupported spec_or_fields type: {type(spec_or_fields)}")


def build_memory_key(spec_or_fields: Any, data: Dict[str, Any]) -> str:
    """Build deterministic memory_key from fixed ordered fields joined by ' | '.
    Missing or empty values render as '_'.
    """
    fields = _extract_fields(spec_or_fields)
    parts: List[str] = []
    for f in fields:
        val = data.get(f)
        if val is None:
            parts.append(MISSING_TOKEN)
        else:
            s = str(val).strip()
            parts.append(s if s else MISSING_TOKEN)
    return SEPARATOR.join(parts)


def parse_memory_key(spec_or_fields: Any, memory_key: str) -> Dict[str, Optional[str]]:
    """Parse a memory_key back into its ordered field dictionary."""
    fields = _extract_fields(spec_or_fields)
    raw_parts = [p.strip() for p in (memory_key or "").split(SEPARATOR)]
    result: Dict[str, Any] = {}
    for idx, f in enumerate(fields):
        if idx < len(raw_parts):
            tok = raw_parts[idx]
            result[f] = None if tok == MISSING_TOKEN or tok == "" else tok
        else:
            result[f] = None
    return result
