from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.models import MemoryStatus
from ..core.namespace import Namespace
from ..providers.llm import LLMProvider
from ..storage.filestore import FileStore


ROUTES = (
    "memory_search",
    "working_memory_search",
    "experience_search",
    "agent_experience_search",
    "hybrid_search",
)


class QueryRouter:
    """Query router supporting expression alias short-circuit + LLM 5-choose-1 routing."""

    def __init__(
        self,
        filestore: FileStore,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.filestore = filestore
        self.llm = llm_provider

    def route(self, ns: Namespace, query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"route": "memory_search", "expanded_query": q, "matched_aliases": []}

        # 1. Deterministic alias short-circuit (§3 Step 5 & §7)
        matched_aliases: List[Dict[str, Any]] = []
        expanded_q = q
        for node in self.filestore.iter_nodes(ns, memory_type="preference", subtype="expression"):
            if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value):
                continue
            alias = str(node.content.get("alias") or node.content.get("source_expression") or "").strip()
            target = str(node.content.get("target") or "").strip()
            if alias and alias in q:
                matched_aliases.append(
                    {
                        "memory_id": node.memory_id,
                        "alias": alias,
                        "target": target,
                        "content": dict(node.content),
                    }
                )
                if target and target not in expanded_q:
                    expanded_q = f"{expanded_q} ({target})"

        if matched_aliases:
            return {
                "route": "memory_search",
                "short_circuit": True,
                "expanded_query": expanded_q,
                "matched_aliases": matched_aliases,
                "memory_types": ["preference"],
            }

        # 2. Deterministic keyword heuristic for 3-tier cognitive routing
        if any(k in q for k in ("工单", "流水", "单据", "历史记录", "上次执行", "working_record", "REC_")):
            return {
                "route": "working_memory_search",
                "short_circuit": True,
                "expanded_query": expanded_q,
                "matched_aliases": [],
                "memory_types": ["working_record"],
            }
        if any(k in q for k in ("经验", "排障", "预案", "指南", "策略卡", "怎么办", "如何处理", "experience", "EXP_")):
            return {
                "route": "experience_search",
                "short_circuit": True,
                "expanded_query": expanded_q,
                "matched_aliases": [],
                "memory_types": ["experience"],
            }

        # 3. Optional LLM 5-choose-1 routing
        if self.llm is not None:
            try:
                resp = self.llm.chat_json(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是检索路由器。从以下5个路由中选1个返回JSON {\"route\": \"...\"}：\n"
                                "- memory_search：用户个人偏好、习惯、老规矩、应用配置\n"
                                "- working_memory_search：历史工单、流水单据\n"
                                "- experience_search：故障排查经验卡、领域处理策略\n"
                                "- agent_experience_search：Agent交互轨迹Case/Policy\n"
                                "- hybrid_search：需要同时查多层记忆"
                            ),
                        },
                        {"role": "user", "content": q},
                    ]
                )
                if isinstance(resp, dict) and resp.get("route") in ROUTES:
                    chosen = resp["route"]
                    mtypes = None
                    if chosen == "memory_search":
                        mtypes = ["preference"]
                    elif chosen == "working_memory_search":
                        mtypes = ["working_record"]
                    elif chosen == "experience_search":
                        mtypes = ["experience"]
                    return {
                        "route": chosen,
                        "short_circuit": False,
                        "expanded_query": expanded_q,
                        "matched_aliases": [],
                        "memory_types": mtypes,
                    }
            except Exception:
                pass

        return {
            "route": "memory_search",
            "short_circuit": False,
            "expanded_query": expanded_q,
            "matched_aliases": [],
            "memory_types": None,
        }
