from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Namespace:
    """Isolation scope for memory nodes and SQL queries.
    Per §1.3: Isolation key is ONLY user_id (and owner_type + agent_id for agent experience).
    tenant_id is strictly audit metadata and is NOT part of the path or SQL isolation key.
    """
    user_id: str
    owner_type: str = "user"  # "user" | "agent"
    agent_id: str = ""
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        if not self.user_id or not str(self.user_id).strip():
            raise ValueError("user_id is required for Namespace isolation")
        if self.owner_type not in ("user", "agent"):
            raise ValueError(f"Invalid owner_type: {self.owner_type}")
        if self.owner_type == "agent" and not self.agent_id:
            raise ValueError("agent_id is required when owner_type == 'agent'")

    @property
    def scope_key(self) -> str:
        """Key used for per-namespace locking and index isolation (excludes tenant_id)."""
        if self.owner_type == "agent":
            return f"agent:{self.user_id}:{self.agent_id}"
        return f"user:{self.user_id}"


def user_namespace(user_id: str, tenant_id: str = "default") -> Namespace:
    return Namespace(user_id=user_id, owner_type="user", agent_id="", tenant_id=tenant_id or "default")


def agent_namespace(user_id: str, agent_id: str, tenant_id: str = "default") -> Namespace:
    return Namespace(user_id=user_id, owner_type="agent", agent_id=agent_id, tenant_id=tenant_id or "default")
