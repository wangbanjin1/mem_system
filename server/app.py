from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is on sys.path when executed directly as `python server/app.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dataclasses import dataclass
from pydantic import BaseModel, Field

from memory_system.api import MemorySystem

try:
    from fastapi import FastAPI, HTTPException
except Exception:

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, detail: str = "") -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    @dataclass
    class _Route:
        path: str
        method: str
        endpoint: Any

    class FastAPI:  # type: ignore[no-redef]
        def __init__(self, title: str = "", version: str = "") -> None:
            self.title = title
            self.version = version
            self.state = type("State", (), {})()
            self.routes: List[_Route] = []

        def get(self, path: str) -> Any:
            def _dec(fn: Any) -> Any:
                self.routes.append(_Route(path=path, method="GET", endpoint=fn))
                return fn

            return _dec

        def post(self, path: str) -> Any:
            def _dec(fn: Any) -> Any:
                self.routes.append(_Route(path=path, method="POST", endpoint=fn))
                return fn

            return _dec


class SessionRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    agent_id: str = "default_agent"
    session_id: str
    turns: List[Dict[str, Any]]
    async_mode: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    query: str
    top_k: int = 10
    levels: str = "L0+L1"
    subtypes: Optional[List[str]] = None
    memory_types: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    owner_type: str = "user"
    agent_id: str = ""
    follow_edges: int = 0
    use_router: bool = False
    bump_heat: bool = True


class WorkingRecordsIngestRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    records: List[Dict[str, Any]] = Field(default_factory=list)


class ExperienceCardsIngestRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    cards: List[Dict[str, Any]] = Field(default_factory=list)


class AgentExperienceRequest(BaseModel):
    tenant_id: str = "default"
    user_id: str
    agent_id: str
    subtype: str = "case"
    content: Dict[str, Any]
    memory_key_fields: Optional[Dict[str, Any]] = None
    confidence: float = 0.9


class MaintenanceRequest(BaseModel):
    tenant_id: str = "default"
    user_id: Optional[str] = None


def create_app(system: Optional[MemorySystem] = None) -> FastAPI:
    app = FastAPI(title="Memory System Service", version="0.2.0")
    mem_sys = system or MemorySystem()
    app.state.memory_system = mem_sys

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "data_root": str(mem_sys.data_root),
            "subtypes_count": len(mem_sys.registry),
        }

    @app.post("/sessions")
    def record_session(req: SessionRequest) -> Dict[str, Any]:
        return mem_sys.record_session(
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            agent_id=req.agent_id,
            session_id=req.session_id,
            turns=req.turns,
            async_mode=req.async_mode,
            metadata=req.metadata,
        )

    @app.post("/search")
    def search_memories(req: SearchRequest) -> Dict[str, Any]:
        hits = mem_sys.search(
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            query=req.query,
            top_k=req.top_k,
            levels=req.levels,
            subtypes=req.subtypes,
            memory_types=req.memory_types,
            statuses=req.statuses,
            owner_type=req.owner_type,
            agent_id=req.agent_id,
            follow_edges=req.follow_edges,
            use_router=req.use_router,
            bump_heat=req.bump_heat,
        )
        return {"results": hits, "count": len(hits)}

    @app.post("/ingest/working_records")
    def ingest_working_records(req: WorkingRecordsIngestRequest) -> Dict[str, Any]:
        nodes = mem_sys.ingest_working_records(
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            records=req.records,
        )
        return {
            "ingested": len(nodes),
            "memory_ids": [n.memory_id for n in nodes],
        }

    @app.post("/ingest/experience_cards")
    def ingest_experience_cards(req: ExperienceCardsIngestRequest) -> Dict[str, Any]:
        nodes = mem_sys.ingest_experience_cards(
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            cards=req.cards,
        )
        return {
            "ingested": len(nodes),
            "memory_ids": [n.memory_id for n in nodes],
        }

    @app.post("/agent_experience")
    def record_agent_experience(req: AgentExperienceRequest) -> Dict[str, Any]:
        node = mem_sys.record_agent_experience(
            tenant_id=req.tenant_id,
            user_id=req.user_id,
            agent_id=req.agent_id,
            subtype=req.subtype,
            content=req.content,
            memory_key_fields=req.memory_key_fields,
            confidence=req.confidence,
        )
        return node.to_dict()

    @app.post("/maintenance")
    def run_maintenance(req: MaintenanceRequest) -> Dict[str, Any]:
        return mem_sys.run_maintenance(tenant_id=req.tenant_id, user_id=req.user_id)

    @app.post("/rebuild_index")
    def rebuild_index() -> Dict[str, Any]:
        rebuilt = mem_sys.rebuild_index()
        return {"rebuilt_nodes": rebuilt}

    @app.get("/nodes/{user_id}/{memory_id}")
    def get_node(user_id: str, memory_id: str, tenant_id: str = "default") -> Dict[str, Any]:
        node = mem_sys.get_node(tenant_id=tenant_id, user_id=user_id, memory_id=memory_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Memory node not found")
        return node

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MEMORY_PORT", "8900"))
    host = os.environ.get("MEMORY_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
