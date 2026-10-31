"""Prompt constants for 5-choose-1 cognitive memory query routing."""
from __future__ import annotations

_ROUTER_SYSTEM_PROMPT: str = (
    '你是检索路由器。从以下5个路由中选1个返回JSON {"route": "..."}：\n'
    "- memory_search：用户个人偏好、习惯、老规矩、应用配置\n"
    "- working_memory_search：历史工单、流水单据\n"
    "- experience_search：故障排查经验卡、领域处理策略\n"
    "- agent_experience_search：Agent交互轨迹Case/Policy\n"
    "- hybrid_search：需要同时查多层记忆"
)

ROUTER_SYSTEM_PROMPT: str = _ROUTER_SYSTEM_PROMPT

__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "_ROUTER_SYSTEM_PROMPT",
]
