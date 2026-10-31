"""Prompt constants and note templates for orthogonal decomposition and cross-session induction."""
from __future__ import annotations

INDUCTION_SYSTEM_PROMPT: str = (
    "你是跨会话记忆归纳引擎。根据用户历史多轮会话中的偏好事件，将其正交拆解为："
    "1) 场景参数覆写 (scene_preference)；"
    "2) 默认参数偏好 (preferred_value)；"
    "3) 周期时间路由 (periodic_preference)；"
    "4) 业务到应用映射 (service_to_app)；"
    "5) 个性化表述映射 (expression)。"
)

SCENE_PREFERENCE_NOTE_TEMPLATE: str = "场景参数覆写：{scene} 下使用 {app} ({biz})"
PREFERRED_VALUE_NOTE_TEMPLATE: str = "默认参数偏好：{app} ({biz})"
PERIODIC_PREFERENCE_NOTE_TEMPLATE: str = "周期时间路由：{cond_tw} -> {app} ({biz})"
EXPRESSION_MAPPING_NOTE_TEMPLATE: str = "表达映射：「{alias}」-> {target}"

__all__ = [
    "EXPRESSION_MAPPING_NOTE_TEMPLATE",
    "INDUCTION_SYSTEM_PROMPT",
    "PERIODIC_PREFERENCE_NOTE_TEMPLATE",
    "PREFERRED_VALUE_NOTE_TEMPLATE",
    "SCENE_PREFERENCE_NOTE_TEMPLATE",
]
