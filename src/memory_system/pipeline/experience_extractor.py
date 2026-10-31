"""Slow-path experience rule extraction and deterministic event post-processing helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..core.experience_rules import (
    EXPERIENCE_RULE_TYPES,
    extract_slow_path_experience_rule,
    match_active_experience_card,
    trigger_matches,
)
from ..core.models import Event, EventType
from ..prompts.experience_prompts import EXPERIENCE_EXTRACTION_SYSTEM_PROMPT

# 1. Validity word lists (§2.6 & §4 Pitfall 2)
_TEMP_VETO_WORDS = (
    "临时",
    "暂时",
    "今天先",
    "这次先",
    "本次先",
    "就今天",
    "就这次",
    "仅本次",
    "就这一回",
    "临时调",
    "临时改",
    "先临时",
    "今天不用",
    "这次不用",
    "今天改在",
    "今天改用",
)

_STRONG_LONG_TERM_WORDS = (
    "以后",
    "默认",
    "一直",
    "总是",
    "长期",
    "每次都",
    "往后",
    "今后",
    "固定用",
    "永远",
)

_WEAK_LONG_TERM_WORDS = (
    "老规矩",
    "照旧",
    "平时",
    "通常",
    "一般",
    "惯例",
    "按惯例",
    "老样子",
    "常规",
    "按上次",
)

_SINGLE_TASK_TIME_WORDS = (
    "今天",
    "周末午休",
    "今天上午",
    "今天下午",
    "今晚",
    "这周末",
)

_CN_DURATION_MAP: Dict[str, float] = {
    "半": 0.5,
    "一": 1.0,
    "二": 2.0,
    "两": 2.0,
    "俩": 2.0,
    "三": 3.0,
    "四": 4.0,
    "一个半": 1.5,
    "两个半": 2.5,
}

# 2. Seven regex patterns for _backfill_time_window (§4 Pitfall 5)
_TW_PAT_1 = re.compile(
    r"(?:从)?(\d{1,2})[时点](\d{1,2})分?(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[时点](\d{1,2})分?(?:结束|止)?"
)
_TW_PAT_2 = re.compile(
    r"(?:从)?(\d{1,2})[点时]半(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时]半(?:结束|止)?"
)
_TW_PAT_3 = re.compile(
    r"(?:从)?(\d{1,2})[点时]半(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时](?:(\d{1,2})分?)?(?:整|钟|结束|止)?"
)
_TW_PAT_4 = re.compile(
    r"(?:从)?(\d{1,2})[点时](?:(\d{1,2})分?)?(?:整|钟|开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时]半(?:结束|止)?"
)
_TW_PAT_5 = re.compile(
    r"(\d{1,2})[:：](\d{2})\s*(?:开始|起)?\s*[、，,至到~—\-]\s*(\d{1,2})[:：](\d{2})(?:结束|止)?"
)
_TW_PAT_6 = re.compile(
    r"(?:从)?(\d{1,2})(?:[:：](\d{2})|[点时](?:(半)|(\d{1,2})分?)?)?(?:开始|起)?[，,、\s]*(?:预计)?(?:持续|播|打|开|用|进行|保障|时长(?:先)?按)\s*(\d+(?:\.\d+)?|一个半|两个半|半|一|二|两|俩|三|四)\s*(?:个)?(小时|钟头|分钟|分|min\b|h\b)",
    re.IGNORECASE,
)
_TW_PAT_7 = re.compile(
    r"(?:从)?(\d{1,2})[点时](?:整|钟)?(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时](?:整|钟)?(?:结束|止)?"
)

_START_TIME_RE = re.compile(
    r"(?:从|今天|时间是|时间从)?\s*(\d{1,2})(?:[:：](\d{2})|[时点](?:(半)|(\d{1,2})分?)?)\s*(?:开始|起)"
)
_DURATION_ONLY_RE = re.compile(
    r"(?:预计)?(?:持续|播|打|开|用|进行|保障(?:开)?|时长(?:先)?按)\s*(\d+(?:\.\d+)?|一个半|两个半|半|一|二|两|俩|三|四)\s*(?:个)?(小时|钟头|分钟|分|min\b|h\b)",
    re.IGNORECASE,
)


def _parse_dur_minutes(val_str: str, unit_str: str) -> int:
    val = _CN_DURATION_MAP.get(val_str, None)
    if val is None:
        val = float(val_str)
    u = unit_str.lower()
    if u in ("小时", "钟头", "h"):
        return int(round(val * 60))
    return int(round(val))


def _normalize_hour_pair(h1: int, m1: int, h2: int, m2: int) -> str:
    """Normalize (h1, m1) -> (h2, m2) to HH:MM-HH:MM, adjusting 12-hour afternoon shorthand like 12:30 -> 1:30."""
    if h1 in (11, 12) and 1 <= h2 <= 6:
        h2 += 12
    h1 = h1 % 24
    h2 = h2 % 24
    return f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"


def extract_time_window_from_text(text: str) -> Optional[str]:
    """Deterministic 7-pattern regex extraction for Chinese time windows."""
    if not text:
        return None

    m = _TW_PAT_5.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))

    m = _TW_PAT_1.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))

    m = _TW_PAT_2.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), 30, int(m.group(2)), 30)

    m = _TW_PAT_3.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), 30, int(m.group(2)), int(m.group(3) or 0))

    m = _TW_PAT_4.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), 30)

    m = _TW_PAT_6.search(text)
    if m:
        h1 = int(m.group(1))
        m1 = int(m.group(2)) if m.group(2) else (30 if m.group(3) else (int(m.group(4)) if m.group(4) else 0))
        dur_minutes = _parse_dur_minutes(m.group(5), m.group(6))
        total_start = h1 * 60 + m1
        total_end = (total_start + dur_minutes) % (24 * 60)
        h2, m2 = divmod(total_end, 60)
        return f"{h1 % 24:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    m = _TW_PAT_7.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), 0, int(m.group(2)), 0)

    m_start = _START_TIME_RE.search(text)
    m_dur = _DURATION_ONLY_RE.search(text)
    if m_start and m_dur:
        h1 = int(m_start.group(1))
        m1 = (
            int(m_start.group(2))
            if m_start.group(2)
            else (30 if m_start.group(3) else (int(m_start.group(4)) if m_start.group(4) else 0))
        )
        dur_minutes = _parse_dur_minutes(m_dur.group(1), m_dur.group(2))
        total_start = h1 * 60 + m1
        total_end = (total_start + dur_minutes) % (24 * 60)
        h2, m2 = divmod(total_end, 60)
        return f"{h1 % 24:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    return None


def ensure_turn_numbers(session_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize turns into [{'turn': int, 'role': 'user'|'assistant', 'content': str}]."""
    result: List[Dict[str, Any]] = []
    for idx, t in enumerate(session_turns, start=1):
        if not isinstance(t, dict):
            continue
        turn_no = int(t.get("turn", idx))
        if "role" not in t and ("user" in t or "agent" in t):
            u_obj = t.get("user")
            a_obj = t.get("agent")
            u_txt = u_obj.get("utterance", "") if isinstance(u_obj, dict) else (str(u_obj) if u_obj else "")
            a_txt = a_obj.get("utterance", "") if isinstance(a_obj, dict) else (str(a_obj) if a_obj else "")
            if u_txt:
                result.append({"turn": turn_no, "role": "user", "content": str(u_txt)})
            if a_txt:
                result.append({"turn": turn_no, "role": "assistant", "content": str(a_txt)})
        else:
            copy_t = dict(t)
            if "turn" not in copy_t:
                copy_t["turn"] = turn_no
            result.append(copy_t)
    return result


def backfill_time_window(event: Event, session_turns: List[Dict[str, Any]]) -> Event:
    """Deterministic regex backfill for time_window (§2.6 & §4 Pitfall 5)."""
    numbered = ensure_turn_numbers(session_turns)
    user_texts_all = [
        str(t.get("content") or t.get("text") or "")
        for t in numbered
        if str(t.get("role", "user")).lower() == "user"
    ]
    combined_user_all = " ".join(user_texts_all)

    if "老规矩不变" in combined_user_all and "延长" in combined_user_all:
        if event.event_type == EventType.PREFERENCE_CONFIRM.value:
            event.content.pop("time_window", None)
            return event

    existing_tw = str(event.content.get("time_window") or "").strip()
    if existing_tw and re.match(r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$", existing_tw):
        return event
    if event.subtype == "expression" or event.event_type == EventType.EXPRESSION_ALIAS.value:
        return event

    ev_turn_set = {int(x) for x in event.evidence_turns}
    ev_user_texts: List[str] = []
    ev_other_texts: List[str] = []
    for t_obj in numbered:
        if int(t_obj.get("turn", 0)) in ev_turn_set:
            txt = str(t_obj.get("content") or t_obj.get("text") or "")
            if not txt:
                continue
            if str(t_obj.get("role", "user")).lower() == "user":
                ev_user_texts.append(txt)
            else:
                ev_other_texts.append(txt)

    for txt in ev_user_texts + ev_other_texts:
        tw = extract_time_window_from_text(txt)
        if tw:
            event.content["time_window"] = tw
            return event

    if ev_user_texts:
        tw = extract_time_window_from_text(" ".join(ev_user_texts))
        if tw:
            event.content["time_window"] = tw
            return event

    all_other_texts = [
        str(t.get("content") or t.get("text") or "")
        for t in numbered
        if str(t.get("role", "user")).lower() != "user"
    ]
    for txt in user_texts_all:
        tw = extract_time_window_from_text(txt)
        if tw:
            event.content["time_window"] = tw
            return event

    if user_texts_all:
        tw = extract_time_window_from_text(combined_user_all)
        if tw:
            event.content["time_window"] = tw
            return event

    for txt in all_other_texts:
        tw = extract_time_window_from_text(txt)
        if tw:
            event.content["time_window"] = tw
            return event

    return event


def backfill_qos_params(event: Event, session_turns: List[Dict[str, Any]]) -> Event:
    """Deterministic regex backfill for resolution and latency when LLM omits them."""
    if event.subtype == "expression" or event.event_type == EventType.EXPRESSION_ALIAS.value:
        return event
    if event.content.get("resolution") and event.content.get("latency"):
        return event
    numbered = ensure_turn_numbers(session_turns)
    user_texts = [
        str(t.get("content") or t.get("text") or "")
        for t in numbered
        if str(t.get("role", "user")).lower() == "user"
    ]
    other_texts = [
        str(t.get("content") or t.get("text") or "")
        for t in numbered
        if str(t.get("role", "user")).lower() != "user"
    ]
    combined_user = " ".join(user_texts)
    combined_all = " ".join(user_texts + other_texts)

    if not event.content.get("resolution"):
        m_res = re.search(
            r"\b(360p|480p|540p|720p|1080p|2160p|2k|4k)\b", combined_user, re.IGNORECASE
        ) or re.search(
            r"\b(360p|480p|540p|720p|1080p|2160p|2k|4k)\b", combined_all, re.IGNORECASE
        )
        if m_res:
            event.content["resolution"] = m_res.group(1).lower()

    if not event.content.get("latency"):
        m_lat = re.search(r"(\d+)\s*(?:ms|毫秒)", combined_user, re.IGNORECASE) or re.search(
            r"(\d+)\s*(?:ms|毫秒)", combined_all, re.IGNORECASE
        )
        if m_lat:
            event.content["latency"] = f"{m_lat.group(1)}ms"

    return event


def normalize_event_validity(event: Event, session_turns: List[Dict[str, Any]]) -> Event:
    """Deterministic validity normalization (§2.6 & §4 Pitfall 2)."""
    numbered = ensure_turn_numbers(session_turns)
    ev_turn_set = {int(x) for x in event.evidence_turns}

    user_utterances: List[str] = []
    for t_obj in numbered:
        if int(t_obj.get("turn", 0)) in ev_turn_set and str(t_obj.get("role", "user")).lower() == "user":
            txt = str(t_obj.get("content") or t_obj.get("text") or "")
            if txt:
                user_utterances.append(txt)

    all_user_utterances: List[str] = [
        str(t_obj.get("content") or t_obj.get("text") or "")
        for t_obj in numbered
        if str(t_obj.get("role", "user")).lower() == "user"
        and str(t_obj.get("content") or t_obj.get("text") or "")
    ]

    if not user_utterances:
        user_utterances = list(all_user_utterances)

    combined_user_text = " ".join(user_utterances)
    full_session_user_text = " ".join(all_user_utterances)
    resolved_validity = event.validity or str(event.content.get("validity") or "long_term")

    if event.subtype == "expression" or event.event_type == EventType.EXPRESSION_ALIAS.value:
        resolved_validity = "long_term"
        event.validity = resolved_validity
        event.content["validity"] = resolved_validity
        return event

    if combined_user_text:
        if any(w in combined_user_text for w in _TEMP_VETO_WORDS) or any(
            w in full_session_user_text for w in _TEMP_VETO_WORDS
        ):
            resolved_validity = "temporary"
        elif any(w in combined_user_text for w in _STRONG_LONG_TERM_WORDS) or any(
            w in full_session_user_text for w in _STRONG_LONG_TERM_WORDS
        ):
            resolved_validity = "long_term"
            event.content["declaration_mode"] = "explicit"
        elif any(w in combined_user_text for w in _WEAK_LONG_TERM_WORDS) or any(
            w in full_session_user_text for w in _WEAK_LONG_TERM_WORDS
        ):
            resolved_validity = "long_term"
            event.content["declaration_mode"] = "explicit"
        elif any(w in full_session_user_text for w in _SINGLE_TASK_TIME_WORDS):
            resolved_validity = "long_term"
            event.content["declaration_mode"] = "implicit"

    event.validity = resolved_validity
    event.content["validity"] = resolved_validity
    return event


def extract_slow_path_experience(
    session_turns: List[Dict[str, Any]],
    session_id: str = "",
    session_ts: str = "",
    ref_year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Extract a slow-path causal revision & parameter adjustment rule card from multi-turn turns."""
    numbered = ensure_turn_numbers(session_turns)
    return extract_slow_path_experience_rule(
        turns=numbered,
        session_id=session_id,
        session_ts=session_ts,
        ref_year=ref_year,
    )


__all__ = [
    "EXPERIENCE_EXTRACTION_SYSTEM_PROMPT",
    "EXPERIENCE_RULE_TYPES",
    "_CN_DURATION_MAP",
    "_SINGLE_TASK_TIME_WORDS",
    "_STRONG_LONG_TERM_WORDS",
    "_TEMP_VETO_WORDS",
    "_WEAK_LONG_TERM_WORDS",
    "backfill_qos_params",
    "backfill_time_window",
    "ensure_turn_numbers",
    "extract_slow_path_experience",
    "extract_slow_path_experience_rule",
    "extract_time_window_from_text",
    "match_active_experience_card",
    "normalize_event_validity",
    "trigger_matches",
]
