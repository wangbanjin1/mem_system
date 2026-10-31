"""Prompt constants for slow-path interaction induction and experience rule extraction."""
from __future__ import annotations

EXPERIENCE_EXTRACTION_SYSTEM_PROMPT: str = (
    "你是慢路径交互归纳与经验规则沉淀器。分析多轮澄清轨迹中的触发上下文 Trigger(C)、"
    "首轮默认参数缺陷 Defect(S_init) 与用户纠正后的参数调整动作 Rule(S_init -> S_revised)，"
    "生成可跨会话复用的结构化经验规则卡（temporal_calendar_adjustment / literal_trap_override / "
    "ledger_disambiguation_policy）。"
)

__all__ = [
    "EXPERIENCE_EXTRACTION_SYSTEM_PROMPT",
]
