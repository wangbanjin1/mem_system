"""Centralized prompt constants and templates for the Agent Memory System."""
from __future__ import annotations

from .experience_prompts import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT
from .extraction_prompts import (
    DEFAULT_EXTRACTION_FALLBACK_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
    load_extraction_system_prompt,
)
from .induction_prompts import (
    EXPRESSION_MAPPING_NOTE_TEMPLATE,
    INDUCTION_SYSTEM_PROMPT,
    PERIODIC_PREFERENCE_NOTE_TEMPLATE,
    PREFERRED_VALUE_NOTE_TEMPLATE,
    SCENE_PREFERENCE_NOTE_TEMPLATE,
)
from .router_prompts import ROUTER_SYSTEM_PROMPT, _ROUTER_SYSTEM_PROMPT

__all__ = [
    "DEFAULT_EXTRACTION_FALLBACK_PROMPT",
    "EXPERIENCE_EXTRACTION_SYSTEM_PROMPT",
    "EXPRESSION_MAPPING_NOTE_TEMPLATE",
    "EXTRACTION_SYSTEM_PROMPT",
    "INDUCTION_SYSTEM_PROMPT",
    "PERIODIC_PREFERENCE_NOTE_TEMPLATE",
    "PREFERRED_VALUE_NOTE_TEMPLATE",
    "ROUTER_SYSTEM_PROMPT",
    "SCENE_PREFERENCE_NOTE_TEMPLATE",
    "_ROUTER_SYSTEM_PROMPT",
    "build_extraction_user_prompt",
    "load_extraction_system_prompt",
]
