"""LLM structured event extractor with deterministic post-processing guardrails."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.logging_utils import JsonLinesLogger
from ..core.models import Event, EventType
from ..core.registry import MemoryRegistry
from ..providers.llm import LLMProvider
from .experience_extractor import (
    _CN_DURATION_MAP,
    _SINGLE_TASK_TIME_WORDS,
    _STRONG_LONG_TERM_WORDS,
    _TEMP_VETO_WORDS,
    _WEAK_LONG_TERM_WORDS,
    backfill_qos_params,
    backfill_time_window,
    ensure_turn_numbers,
    extract_slow_path_experience,
    extract_time_window_from_text,
    normalize_event_validity,
)

__all__ = [
    "Extractor",
    "_CN_DURATION_MAP",
    "_SINGLE_TASK_TIME_WORDS",
    "_STRONG_LONG_TERM_WORDS",
    "_TEMP_VETO_WORDS",
    "_WEAK_LONG_TERM_WORDS",
    "extract_time_window_from_text",
]


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
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default

    def normalize_app(self, app_val: Any) -> Optional[str]:
        """Normalize an application name or alias to its canonical application ID."""
        if not app_val:
            return None
        s = str(app_val).strip().lower()
        return self._alias_to_app.get(s, str(app_val).strip())

    def normalize_business(self, biz_val: Any) -> Optional[str]:
        """Normalize a business/service name or alias to its canonical business code."""
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

        raw_events = (
            raw_obj.get("events", [])
            if isinstance(raw_obj, dict)
            else (raw_obj if isinstance(raw_obj, list) else [])
        )
        valid_events: List[Event] = []

        for raw_ev in raw_events:
            if not isinstance(raw_ev, dict):
                continue
            ev = self._parse_raw_event(raw_ev, session_id=session_id, user_id=user_id, tenant_id=tenant_id)
            self._backfill_time_window(ev, numbered_turns)
            self._backfill_qos_params(ev, numbered_turns)
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

        if "latency" not in content or not content["latency"]:
            rtt_val = content.get("rtt") or content.get("rtt_max")
            if rtt_val:
                content["latency"] = str(rtt_val).replace("毫秒", "ms").strip()
        if "latency" in content and isinstance(content["latency"], str):
            m_ms = re.search(r"(\d+)\s*(?:ms|毫秒)", content["latency"], re.IGNORECASE)
            if m_ms:
                content["latency"] = f"{m_ms.group(1)}ms"

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
            except (ValueError, TypeError):
                continue

        validity_str = str(
            raw_ev.get("validity") or raw_ev.get("validity_type") or content.get("validity") or "long_term"
        )
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
        return ensure_turn_numbers(session_turns)

    def _backfill_time_window(self, event: Event, session_turns: List[Dict[str, Any]]) -> Event:
        return backfill_time_window(event, session_turns)

    def _backfill_qos_params(self, event: Event, session_turns: List[Dict[str, Any]]) -> Event:
        return backfill_qos_params(event, session_turns)

    def _normalize_validity(self, event: Event, session_turns: List[Dict[str, Any]]) -> Event:
        return normalize_event_validity(event, session_turns)

    def validate(self, event: Event, session_turns: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Pure Python deterministic validation (§2.6).
        Discards invalid events and logs 'extract_discard' without retrying.
        """
        if event.event_type not in self._valid_event_types:
            self.logger.log("extract_discard", reason="invalid_event_type", event_type=event.event_type)
            return False

        spec = self.registry.get(event.subtype)
        if spec is None:
            self.logger.log("extract_discard", reason="unknown_subtype", subtype=event.subtype)
            return False

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

        res = event.content.get("resolution")
        if res not in (None, "") and self._valid_resolutions:
            if str(res).strip().lower() not in self._valid_resolutions:
                self.logger.log(
                    "extract_discard",
                    reason="invalid_resolution",
                    resolution=res,
                )
                return False

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

    def extract_slow_path_experience(
        self,
        session_turns: List[Dict[str, Any]],
        session_id: str = "",
        session_ts: str = "",
        ref_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract a general slow-path causal revision & parameter adjustment rule card from multi-turn turns."""
        return extract_slow_path_experience(
            session_turns=session_turns,
            session_id=session_id,
            session_ts=session_ts,
            ref_year=ref_year,
        )
