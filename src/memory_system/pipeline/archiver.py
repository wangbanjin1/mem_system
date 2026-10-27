from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..storage.filestore import FileStore


class SessionArchiver:
    """Archives raw session turns and metadata to sessions/{sid}/history.jsonl + meta.json."""

    def __init__(self, filestore: FileStore) -> None:
        self.filestore = filestore

    def archive_session(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        turns: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        return self.filestore.save_session(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            turns=turns,
            metadata=metadata,
        )
