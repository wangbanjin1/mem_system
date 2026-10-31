"""Prompt constants and message builders for LLM structured event extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_EXTRACTION_FALLBACK_PROMPT: str = "Extract memory events as JSON {'events': [...]}."

EXTRACTION_SYSTEM_PROMPT: str = (
    "# 角色与任务\n\n"
    "你是 Agent 记忆系统的结构化事件抽取器。请仔细阅读给定的单次会话对话历史（带 turn 编号），"
    "从中抽取结构化记忆事件 `events[]`。\n"
    "只输出合法 JSON 对象，格式为：`{\"events\": [...]}`，不要包含任何多余文字。"
)


def load_extraction_system_prompt(prompt_path: Optional[Path] = None) -> str:
    """Load the extraction system prompt from markdown file with fallback to default constant."""
    if prompt_path is not None and prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return DEFAULT_EXTRACTION_FALLBACK_PROMPT


def build_extraction_user_prompt(session_id: str, numbered_turns: List[Dict[str, Any]]) -> str:
    """Format numbered dialogue turns into the JSON user prompt for LLM event extraction."""
    return json.dumps(
        {"session_id": session_id, "turns": numbered_turns},
        ensure_ascii=False,
        indent=2,
    )


__all__ = [
    "DEFAULT_EXTRACTION_FALLBACK_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_user_prompt",
    "load_extraction_system_prompt",
]
