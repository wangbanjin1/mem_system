from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.logging_utils import JsonLinesLogger
from ..core.models import Event, EventType, now_utc_str
from ..core.registry import MemoryRegistry
from ..providers.llm import LLMProvider


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
# Pattern 1: "12时30分开始、13时30分结束" / "12点30分起至13点30分止"
_TW_PAT_1 = re.compile(
    r"(?:从)?(\d{1,2})[时点](\d{1,2})分?(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[时点](\d{1,2})分?(?:结束|止)?"
)
# Pattern 2: "从12点半到1点半" / "12时半至13时半"
_TW_PAT_2 = re.compile(
    r"(?:从)?(\d{1,2})[点时]半(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时]半(?:结束|止)?"
)
# Pattern 3: "12点半到14点(15分)"
_TW_PAT_3 = re.compile(
    r"(?:从)?(\d{1,2})[点时]半(?:开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时](?:(\d{1,2})分?)?(?:整|钟|结束|止)?"
)
# Pattern 4: "12点(15分)到1点半"
_TW_PAT_4 = re.compile(
    r"(?:从)?(\d{1,2})[点时](?:(\d{1,2})分?)?(?:整|钟|开始|起)?\s*(?:[、，,至到~—\-]|以及|和)\s*(\d{1,2})[点时]半(?:结束|止)?"
)
# Pattern 5: "18:45至19:45" / "18:45 - 19:45" / "18：45到19：45"
_TW_PAT_5 = re.compile(
    r"(\d{1,2})[:：](\d{2})\s*(?:开始|起)?\s*[、，,至到~—\-]\s*(\d{1,2})[:：](\d{2})(?:结束|止)?"
)
# Pattern 6: Start time + duration (handles cross-midnight, e.g., "23点开始持续2小时" -> "23:00-01:00")
_TW_PAT_6 = re.compile(
    r"(?:从)?(\d{1,2})(?:[:：](\d{2})|[点时](?:(半)|(\d{1,2})分?)?)?(?:开始|起)?[，,、\s]*(?:预计)?(?:持续|播|打|开|用|进行|保障|时长(?:先)?按)\s*(\d+(?:\.\d+)?|一个半|两个半|半|一|二|两|俩|三|四)\s*(?:个)?(小时|钟头|分钟|分|min\b|h\b)",
    re.IGNORECASE,
)
# Pattern 7: "从12点到14点" / "20时开始、22时结束"
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
        if m.group(2):
            m1 = int(m.group(2))
        elif m.group(3):
            m1 = 30
        elif m.group(4):
            m1 = int(m.group(4))
        else:
            m1 = 0
        dur_minutes = _parse_dur_minutes(m.group(5), m.group(6))
        total_start = h1 * 60 + m1
        total_end = (total_start + dur_minutes) % (24 * 60)
        h2, m2 = divmod(total_end, 60)
        return f"{h1 % 24:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    m = _TW_PAT_7.search(text)
    if m:
        return _normalize_hour_pair(int(m.group(1)), 0, int(m.group(2)), 0)

    # Cross-clause / cross-turn start time + duration in combined text
    m_start = _START_TIME_RE.search(text)
    m_dur = _DURATION_ONLY_RE.search(text)
    if m_start and m_dur:
        h1 = int(m_start.group(1))
        if m_start.group(2):
            m1 = int(m_start.group(2))
        elif m_start.group(3):
            m1 = 30
        elif m_start.group(4):
            m1 = int(m_start.group(4))
        else:
            m1 = 0
        dur_minutes = _parse_dur_minutes(m_dur.group(1), m_dur.group(2))
        total_start = h1 * 60 + m1
        total_end = (total_start + dur_minutes) % (24 * 60)
        h2, m2 = divmod(total_end, 60)
        return f"{h1 % 24:02d}:{m1:02d}-{h2:02d}:{m2:02d}"

    return None


class Extractor:
    """LLM structured event extractor with 3 deterministic guardrails:
    1. _backfill_time_window
    2. _normalize_validity
    3. validate
    """

    def __init__(
        self,
        registry: MemoryRegistry,
        llm_provider: Optional[LLMProvider] = None,
        data_dir: Optional[str | Path] = None,
        prompt_path: Optional[str | Path] = None,
        logger: Optional[JsonLinesLogger] = None,
    ) -> None:
        self.registry = registry
        self.llm = llm_provider
        self.logger = logger or JsonLinesLogger()

        repo_root = Path(__file__).resolve().parents[3]
        self.data_dir = Path(data_dir) if data_dir else (repo_root / "data")
        self.prompt_path = Path(prompt_path) if prompt_path else (repo_root / "prompts" / "extract_generic.md")

        self.app_map: Dict[str, List[str]] = self._load_json(self.data_dir / "app_map.json", {})
        self.app_config: Dict[str, Any] = self._load_json(self.data_dir / "app_config.json", {})
        self.support_matrix: Dict[str, List[str]] = self._load_json(self.data_dir / "support_matrix.json", {})

        # Reverse lookup maps for normalization
        self._alias_to_app: Dict[str, str] = {}
        for canonical, aliases in self.app_map.items():
            self._alias_to_app[canonical.lower()] = canonical
            for a in aliases:
                self._alias_to_app[a.lower()] = canonical

        self._alias_to_biz: Dict[str, str] = {}
        for canonical_biz, aliases in (self.app_config.get("businesses") or {}).items():
            self._alias_to_biz[canonical_biz.lower()] = canonical_biz
            for a in aliases:
                self._alias_to_biz[a.lower()] = canonical_biz

        self._valid_resolutions: Set[str] = {
            str(r).lower() for r in (self.app_config.get("resolutions") or [])
        }
        self._valid_event_types: Set[str] = {e.value for e in EventType}

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default
        return default

    def normalize_app(self, app_val: Any) -> Optional[str]:
        if not app_val:
            return None
        s = str(app_val).strip().lower()
        return self._alias_to_app.get(s, str(app_val).strip())

    def normalize_business(self, biz_val: Any) -> Optional[str]:
        if not biz_val:
            return None
        s = str(biz_val).strip().lower()
        return self._alias_to_biz.get(s, str(biz_val).strip())

    def extract(
        self,
        session_turns: List[Dict[str, Any]],
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "default",
    ) -> List[Event]:
        """Run single-call LLM extraction + 3-stage deterministic post-processing."""
        if self.llm is None or not session_turns:
            return []

        system_prompt = (
            self.prompt_path.read_text(encoding="utf-8")
            if self.prompt_path.exists()
            else "Extract memory events as JSON {'events': [...]}."
        )
        numbered_turns = self._ensure_turn_numbers(session_turns)
        user_prompt = json.dumps({"session_id": session_id, "turns": numbered_turns}, ensure_ascii=False, indent=2)

        try:
            raw_obj = self.llm.chat_json(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            self.logger.log("extract_llm_error", session_id=session_id, user_id=user_id, error=str(exc))
            return []

        raw_events = raw_obj.get("events", []) if isinstance(raw_obj, dict) else (raw_obj if isinstance(raw_obj, list) else [])
        valid_events: List[Event] = []

        for raw_ev in raw_events:
            if not isinstance(raw_ev, dict):
                continue
            ev = self._parse_raw_event(raw_ev, session_id=session_id, user_id=user_id, tenant_id=tenant_id)
            self._backfill_time_window(ev, numbered_turns)
            self._normalize_validity(ev, numbered_turns)
            if self.validate(ev, numbered_turns):
                valid_events.append(ev)

        return valid_events

    def _parse_raw_event(
        self,
        raw_ev: Dict[str, Any],
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "default",
    ) -> Event:
        content = dict(raw_ev.get("content") or {})
        key_fields = dict(raw_ev.get("memory_key_fields") or {})

        # Normalize rtt / rtt_max -> latency in content
        if "latency" not in content or not content["latency"]:
            rtt_val = content.get("rtt") or content.get("rtt_max")
            if rtt_val:
                content["latency"] = str(rtt_val).replace("毫秒", "ms").strip()
        if "latency" in content and isinstance(content["latency"], str):
            m_ms = re.search(r"(\d+)\s*(?:ms|毫秒)", content["latency"], re.IGNORECASE)
            if m_ms:
                content["latency"] = f"{m_ms.group(1)}ms"

        # Normalize app and business in both key_fields and content
        for d in (key_fields, content):
            if "application" in d and d["application"]:
                d["application"] = self.normalize_app(d["application"])
            if "business" in d and d["business"]:
                d["business"] = self.normalize_business(d["business"])
            if "resolution" in d and isinstance(d["resolution"], str):
                res_low = d["resolution"].strip().lower()
                if res_low in ("360p", "480p", "540p", "720p", "1080p", "2160p", "2k", "4k"):
                    d["resolution"] = res_low

        subtype = str(raw_ev.get("subtype") or raw_ev.get("target_memory_type") or "application_business")
        spec = self.registry.get(subtype)
        if spec is not None:
            for kf in spec.memory_key_fields:
                if kf not in key_fields and kf in content and content[kf] not in (None, ""):
                    key_fields[kf] = content[kf]
                if kf not in content and kf in key_fields and key_fields[kf] not in (None, ""):
                    content[kf] = key_fields[kf]

        ev_type = str(raw_ev.get("event_type", EventType.PREFERENCE_SET.value)).upper()
        # Map alias names if LLM used lowercase or alternative names
        if ev_type in ("EXPLICIT_LONG_TERM_STATEMENT", "LONG_TERM_PREFERENCE", "SET"):
            ev_type = EventType.PREFERENCE_SET.value
        elif ev_type in ("PREFERENCE_REINFORCEMENT", "CONFIRM"):
            ev_type = EventType.PREFERENCE_CONFIRM.value
        elif ev_type in ("CORRECTION", "CORRECT"):
            ev_type = EventType.PREFERENCE_CORRECT.value
        elif ev_type in ("CLEAR", "DELETE"):
            ev_type = EventType.PREFERENCE_CLEAR.value

        ev_turns_raw = raw_ev.get("evidence_turns") or []
        ev_turns: List[int] = []
        for t in ev_turns_raw:
            try:
                ev_turns.append(int(t))
            except Exception:
                continue

        validity_str = str(raw_ev.get("validity") or raw_ev.get("validity_type") or content.get("validity") or "long_term")
        return Event(
            event_type=ev_type,
            subtype=subtype,
            memory_key_fields=key_fields,
            content=content,
            validity=validity_str,
            confidence=float(raw_ev.get("confidence", 0.9)),
            source_session_id=session_id,
            evidence_turns=ev_turns,
            is_concession=bool(raw_ev.get("is_concession", False)),
            user_id=user_id,
            tenant_id=tenant_id,
        )

    @staticmethod
    def _ensure_turn_numbers(session_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize turns into [{'turn': int, 'role': 'user'|'assistant', 'content': str}].
        Supports both standard role/content dicts and cross_memory_data event_sequence items
        with {'turn': i, 'user': {'utterance': '...'}, 'agent': {'utterance': '...'}}.
        """
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

    def _backfill_time_window(self, event: Event, session_turns: List[Dict[str, Any]]) -> Event:
        """Deterministic regex backfill for time_window (§2.6 & §4 Pitfall 5, extractor.py:226-259).
        Only fills if LLM left time_window empty; searches evidence_turns first, then full session.
        """
        numbered = self._ensure_turn_numbers(session_turns)
        user_texts_all = [
            str(t.get("content") or t.get("text") or "")
            for t in numbered
            if str(t.get("role", "user")).lower() == "user"
        ]
        combined_user_all = " ".join(user_texts_all)

        # If user is only extending today's duration while keeping baseline ("老规矩不变...不过今天...延长"),
        # do not overwrite baseline time_window with today's extended window.
        if "老规矩不变" in combined_user_all and "延长" in combined_user_all:
            if event.event_type == EventType.PREFERENCE_CONFIRM.value:
                event.content.pop("time_window", None)
                return event

        existing_tw = event.content.get("time_window")
        if existing_tw and str(existing_tw).strip():
            return event
        if event.subtype == "expression" or event.event_type == EventType.EXPRESSION_ALIAS.value:
            return event

        ev_turn_set = {int(x) for x in event.evidence_turns}

        # 1. Search evidence_turns first (prefer user utterances, then assistant in evidence_turns)
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

        # 2. Fallback to all turns in the session (user utterances first, then combined user utterances, then assistant)
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

    def _normalize_validity(self, event: Event, session_turns: List[Dict[str, Any]]) -> Event:
        """Deterministic validity normalization (§2.6 & §4 Pitfall 2, extractor.py:263-312).
        Rules:
        - Inspect ONLY user utterances (agent system phrasing like '长期配置' can cause false positives).
        - Any temporary veto word ('临时', etc.) -> one-vote veto 'temporary' (even with '老规矩'/'以后').
        - Strong long-term words ('以后', '默认', '一直', '总是'...) -> 'long_term'.
        - Weak long-term words ('老规矩', '照旧', '平时', '通常', '一般', '惯例') -> 'long_term'.
        - If none matched, preserve LLM's original validity.
        - Overwrite BOTH event.validity and event.content['validity'].
        """
        numbered = self._ensure_turn_numbers(session_turns)
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

        # Expression alias definitions ("我说‘X’，就是Y，以后也照这个标准理解") are long-term unless the alias itself is temporary
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
                # Single-task implicit behavior without explicit long-term declaration:
                # record as candidate (provisional, vote_threshold=2) to prevent single-behavior overfitting!
                resolved_validity = "long_term"
                event.content["declaration_mode"] = "implicit"

        event.validity = resolved_validity
        event.content["validity"] = resolved_validity
        return event

    def validate(self, event: Event, session_turns: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Pure Python deterministic validation (§2.6, extractor.py:314-384).
        Discards invalid events and logs 'extract_discard' without retrying.
        """
        # 1. Closed event_type set
        if event.event_type not in self._valid_event_types:
            self.logger.log("extract_discard", reason="invalid_event_type", event_type=event.event_type)
            return False

        # 2. Subtype exists in registry
        spec = self.registry.get(event.subtype)
        if spec is None:
            self.logger.log("extract_discard", reason="unknown_subtype", subtype=event.subtype)
            return False

        # 3. Content schema & memory_key_fields completeness
        if not isinstance(event.content, dict) or not event.content:
            self.logger.log("extract_discard", reason="empty_content", subtype=event.subtype)
            return False

        for kf in spec.memory_key_fields:
            val = event.memory_key_fields.get(kf)
            if val in (None, "", "_"):
                val = event.content.get(kf)
                if val not in (None, "", "_"):
                    event.memory_key_fields[kf] = val
            if val in (None, "", "_"):
                self.logger.log(
                    "extract_discard",
                    reason="missing_memory_key_field",
                    subtype=event.subtype,
                    field=kf,
                )
                return False
            # Ensure content repeats memory_key_fields
            event.content.setdefault(kf, val)

        for rf in spec.required_fields:
            if event.content.get(rf) in (None, "", "_"):
                self.logger.log(
                    "extract_discard",
                    reason="missing_required_field",
                    subtype=event.subtype,
                    field=rf,
                )
                return False

        # 4. Resolution value domain check
        res = event.content.get("resolution")
        if res not in (None, "") and self._valid_resolutions:
            if str(res).strip().lower() not in self._valid_resolutions:
                self.logger.log(
                    "extract_discard",
                    reason="invalid_resolution",
                    resolution=res,
                )
                return False

        # 5. Support matrix check for (application, business)
        app = event.content.get("application")
        biz = event.content.get("business")
        if app and biz and self.support_matrix:
            norm_app = self.normalize_app(app) or str(app)
            norm_biz = self.normalize_business(biz) or str(biz)
            if norm_app not in self.support_matrix:
                self.logger.log(
                    "extract_discard",
                    reason="unsupported_application",
                    application=app,
                )
                return False
            allowed_biz = [b.lower() for b in self.support_matrix.get(norm_app, [])]
            if norm_biz.lower() not in allowed_biz and str(biz).lower() not in allowed_biz:
                self.logger.log(
                    "extract_discard",
                    reason="unsupported_app_business_pair",
                    application=norm_app,
                    business=biz,
                )
                return False

        # 6. evidence_turns must actually exist in session_turns
        if session_turns is not None:
            numbered = self._ensure_turn_numbers(session_turns)
            valid_turn_ids = {int(t["turn"]) for t in numbered if "turn" in t}
            if not event.evidence_turns:
                self.logger.log("extract_discard", reason="empty_evidence_turns", subtype=event.subtype)
                return False
            for t_id in event.evidence_turns:
                if int(t_id) not in valid_turn_ids:
                    self.logger.log(
                        "extract_discard",
                        reason="nonexistent_evidence_turn",
                        turn=t_id,
                        valid_turns=sorted(valid_turn_ids),
                    )
                    return False

        return True
