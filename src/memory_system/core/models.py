from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


def now_utc_str() -> str:
    """Return local-compatible ISO timestamp YYYY-MM-DDTHH:MM:SSZ.
    Note: per guide §4 Pitfall 8, uses localtime so strptime + time.mktime round-trips cleanly.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime())


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REINFORCED = "reinforced"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class EventType(str, Enum):
    PREFERENCE_SET = "PREFERENCE_SET"
    PREFERENCE_CONFIRM = "PREFERENCE_CONFIRM"
    PREFERENCE_CORRECT = "PREFERENCE_CORRECT"
    PREFERENCE_CLEAR = "PREFERENCE_CLEAR"
    EXPRESSION_ALIAS = "EXPRESSION_ALIAS"
    WORKING_RECORD = "WORKING_RECORD"
    WORKING_RECORD_REUSE = "WORKING_RECORD_REUSE"
    WORKING_RECORD_RESCHEDULE = "WORKING_RECORD_RESCHEDULE"
    EXPERIENCE_CRYSTALLIZE = "EXPERIENCE_CRYSTALLIZE"
    EXPERIENCE_APPLY = "EXPERIENCE_APPLY"
    CASE_RECORD = "CASE_RECORD"
    TRAJECTORY_RECORD = "TRAJECTORY_RECORD"
    POLICY_RECORD = "POLICY_RECORD"


class EdgeType(str, Enum):
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    REINFORCES = "reinforces"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    SUMMARIZED_INTO = "summarized_into"
    ABSTRACTED_INTO = "abstracted_into"
    DERIVED_FROM_CASE = "derived_from_case"
    DERIVED_FROM_TRAJECTORY = "derived_from_trajectory"
    USES = "uses"


@dataclass
class Stats:
    votes: int = 0
    positive_votes: int = 0
    negative_votes: int = 0
    heat: float = 0.0
    last_heat_at: str = field(default_factory=now_utc_str)
    last_reinforced_at: str = field(default_factory=now_utc_str)
    reuse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "votes": self.votes,
            "positive_votes": self.positive_votes,
            "negative_votes": self.negative_votes,
            "heat": round(self.heat, 6),
            "last_heat_at": self.last_heat_at,
            "last_reinforced_at": self.last_reinforced_at,
            "reuse_count": self.reuse_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Stats:
        if not data:
            return cls()
        return cls(
            votes=int(data.get("votes", 0)),
            positive_votes=int(data.get("positive_votes", 0)),
            negative_votes=int(data.get("negative_votes", 0)),
            heat=float(data.get("heat", 0.0)),
            last_heat_at=str(data.get("last_heat_at") or now_utc_str()),
            last_reinforced_at=str(data.get("last_reinforced_at") or now_utc_str()),
            reuse_count=int(data.get("reuse_count", 0)),
        )


@dataclass
class ValidityInfo:
    type: str = "long_term"  # "long_term" | "temporary"
    valid_from: Optional[str] = field(default_factory=now_utc_str)
    valid_to: Optional[str] = None
    condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ValidityInfo:
        if isinstance(data, str):
            return cls(type=data)
        if not isinstance(data, dict):
            return cls()
        return cls(
            type=str(data.get("type", "long_term")),
            valid_from=data.get("valid_from", now_utc_str()),
            valid_to=data.get("valid_to"),
            condition=data.get("condition"),
        )


@dataclass
class MemoryNode:
    memory_id: str = field(default_factory=lambda: f"MEM_{uuid.uuid4().hex[:12].upper()}")
    memory_type: str = "preference"
    subtype: str = "application_business"
    memory_key: str = ""
    owner_type: str = "user"  # "user" | "agent"
    user_id: str = ""
    agent_id: str = ""
    tenant_id: str = "default"
    content: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    overview: str = ""
    details: str = ""
    status: str = MemoryStatus.CANDIDATE.value
    confidence: float = 0.85
    version: int = 1
    validity: ValidityInfo = field(default_factory=ValidityInfo)
    stats: Stats = field(default_factory=Stats)
    evidence: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_utc_str)
    updated_at: str = field(default_factory=now_utc_str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "subtype": self.subtype,
            "memory_key": self.memory_key,
            "owner": {
                "owner_type": self.owner_type,
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "tenant_id": self.tenant_id,
            },
            "owner_type": self.owner_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "content": dict(self.content),
            "description": self.description,
            "overview": self.overview,
            "details": self.details,
            "status": self.status if isinstance(self.status, str) else self.status.value,
            "confidence": round(float(self.confidence), 4),
            "version": int(self.version),
            "validity": self.validity.to_dict(),
            "stats": self.stats.to_dict(),
            "evidence": list(self.evidence),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryNode:
        owner = data.get("owner") or {}
        status_val = data.get("status", MemoryStatus.CANDIDATE.value)
        if isinstance(status_val, MemoryStatus):
            status_val = status_val.value
        return cls(
            memory_id=str(data.get("memory_id") or f"MEM_{uuid.uuid4().hex[:12].upper()}"),
            memory_type=str(data.get("memory_type", "preference")),
            subtype=str(data.get("subtype") or data.get("memory_subtype") or "application_business"),
            memory_key=str(data.get("memory_key", "")),
            owner_type=str(data.get("owner_type") or owner.get("owner_type") or "user"),
            user_id=str(data.get("user_id") or owner.get("user_id") or owner.get("owner_id") or ""),
            agent_id=str(data.get("agent_id") or owner.get("agent_id") or ""),
            tenant_id=str(data.get("tenant_id") or owner.get("tenant_id") or "default"),
            content=dict(data.get("content") or {}),
            description=str(data.get("description", "")),
            overview=str(data.get("overview", "")),
            details=str(data.get("details", "")),
            status=str(status_val),
            confidence=float(data.get("confidence", 0.85)),
            version=int(data.get("version", 1)),
            validity=ValidityInfo.from_dict(data.get("validity")),
            stats=Stats.from_dict(data.get("stats") or {}),
            evidence=list(data.get("evidence") or []),
            created_at=str(data.get("created_at") or now_utc_str()),
            updated_at=str(data.get("updated_at") or now_utc_str()),
        )


@dataclass
class Event:
    event_id: str = field(default_factory=lambda: f"EVT_{uuid.uuid4().hex[:12].upper()}")
    event_type: str = EventType.PREFERENCE_SET.value
    subtype: str = "application_business"
    memory_key_fields: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    validity: str = "long_term"
    confidence: float = 0.9
    source_session_id: str = ""
    evidence_turns: List[int] = field(default_factory=list)
    target_slots: List[str] = field(default_factory=list)
    is_concession: bool = False
    event_time: str = field(default_factory=now_utc_str)
    owner_type: str = "user"
    user_id: str = ""
    agent_id: str = ""
    tenant_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type if isinstance(self.event_type, str) else self.event_type.value,
            "subtype": self.subtype,
            "memory_key_fields": dict(self.memory_key_fields),
            "content": dict(self.content),
            "validity": self.validity,
            "confidence": round(float(self.confidence), 4),
            "source_session_id": self.source_session_id,
            "evidence_turns": list(self.evidence_turns),
            "target_slots": list(self.target_slots),
            "is_concession": self.is_concession,
            "event_time": self.event_time,
            "owner_type": self.owner_type,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        et = data.get("event_type", EventType.PREFERENCE_SET.value)
        if isinstance(et, EventType):
            et = et.value
        return cls(
            event_id=str(data.get("event_id") or f"EVT_{uuid.uuid4().hex[:12].upper()}"),
            event_type=str(et),
            subtype=str(data.get("subtype", "application_business")),
            memory_key_fields=dict(data.get("memory_key_fields") or {}),
            content=dict(data.get("content") or {}),
            validity=str(data.get("validity") or data.get("validity_type") or "long_term"),
            confidence=float(data.get("confidence", 0.9)),
            source_session_id=str(data.get("source_session_id", "")),
            evidence_turns=[int(t) for t in (data.get("evidence_turns") or [])],
            target_slots=list(data.get("target_slots") or []),
            is_concession=bool(data.get("is_concession", False)),
            event_time=str(data.get("event_time") or now_utc_str()),
            owner_type=str(data.get("owner_type", "user")),
            user_id=str(data.get("user_id", "")),
            agent_id=str(data.get("agent_id", "")),
            tenant_id=str(data.get("tenant_id", "default")),
        )


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: str = EdgeType.SUPPORTS.value
    user_id: str = ""
    owner_type: str = "user"
    agent_id: str = ""
    created_at: str = field(default_factory=now_utc_str)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type if isinstance(self.edge_type, str) else self.edge_type.value,
            "user_id": self.user_id,
            "owner_type": self.owner_type,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Edge:
        return cls(
            source_id=str(data.get("source_id", "")),
            target_id=str(data.get("target_id", "")),
            edge_type=str(data.get("edge_type", EdgeType.SUPPORTS.value)),
            user_id=str(data.get("user_id", "")),
            owner_type=str(data.get("owner_type", "user")),
            agent_id=str(data.get("agent_id", "")),
            created_at=str(data.get("created_at") or now_utc_str()),
            metadata=dict(data.get("metadata") or {}),
        )
