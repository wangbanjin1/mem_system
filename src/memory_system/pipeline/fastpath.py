"""Rule-based zero-LLM fast path for alias mapping, slot matching, corrections, and noise filtering."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.models import Event, EventType
from .extractor import Extractor, extract_time_window_from_text

# Parameter signal regex (§4 Pitfall 3): if matched, MUST NOT skip LLM!
_PARAM_SIGNAL_RE = re.compile(
    r"(?:360p|480p|540p|720p|1080p|2160p|2k|4k|\d+\s*ms|\d+\s*毫秒|画质|分辨率|清晰度|时延|延迟|老规矩|照旧|码率|帧率|稳定优先|画质优先|低时延|流畅优先|弱网|\d{1,2}[:：点时]\d{0,2})",
    re.IGNORECASE,
)

# Concession patterns (§4 Pitfall 3): "行，那就按X办吧" -> forces LLM and marks concession
_CONCESSION_RE = re.compile(
    r"(?:行[，,\s]*那就按|好吧[，,\s]*那就|那先按.+办|那就按你说的|那就这样吧|行吧[，,\s]*按)",
    re.IGNORECASE,
)

# Pure greeting / noise patterns
_NOISE_EXACT_RE = re.compile(
    r"^(?:[\s，,。！!？?、]*(?:你好|您好|谢谢|多谢|感谢|好的|收到|好勒|明白|知道了|ok|okay|嗯|嗯嗯|再见|拜拜|没事了|不用了))+[\s，,。！!？?、]*$",
    re.IGNORECASE,
)

# Alias rules covering general and cross_memory_data expression_vault templates
_ALIAS_RULE_PATTERNS = [
    re.compile(
        r"(?:以后|默认|平时)?(?:我)?(?:一)?(?:说|提到|提)[“\"‘'「『]([^”\"’'」』，,。]{2,16})[”\"’'」』](?:时)?[，,\s]*(?:画质|时延(?:上限)?|时长|时间)?(?:通常)?(?:就是|代表|指的就是|指的是|说的就是|意思是|等于|就按|按)([^。！!，,\n]{2,32})"
    ),
    re.compile(
        r"(?:我)?(?:平时|通常|以后)?把(?:这个应用|它)?叫[“\"‘'「『]?([^”\"’'」』，,。]{2,16})[”\"’'」』]?[，,\s]*(?:指的就是|指的是|就是|代表)([^。！!，,\n]{2,32})"
    ),
    re.compile(
        r"(?:以后|默认|平时)?(?:我)?(?:一)?说[“\"‘'「『]?([^”\"’'」』，,。]{2,16})[”\"’'」』]?[，,\s]*(?:就是|代表|指的就是|指的是|意思是|等于)([^。！!，,\n]{2,32})"
    ),
    re.compile(
        r"(?:画质|清晰度|分辨率)(?:按|要)?\s*([^，,。！!；;\s]{2,12}?)(?:就行|即可|就可以)?[，,\s]*(?:档位)?按\s*(360p|480p|540p|720p|1080p|2160p|2k|4k)\s*(?:开|来|算|控制)?[，,\s]*(?:以后|往后)(?:也|都)?照这个标准(?:理解|记)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:时延|延迟)(?:要|按)?\s*([^，,。！!；;\s]{2,12}?)(?:就行|即可|就可以)?[，,\s]*(?:上限)?按\s*(\d+\s*(?:ms|毫秒))\s*(?:控制|开|来|算)?[，,\s]*(?:以后|往后)(?:也|都)?照这个标准(?:理解|记)",
        re.IGNORECASE,
    ),
]

# Multi-turn disambiguation: agent asks "您说的‘X’具体指..." and user confirms "对，就是画质档位720p..."
_AGENT_DISAMBIG_RE = re.compile(
    r"您说的[“\"‘'「『]([^”\"’'」』]{2,16})[”\"’'」』]具体(?:是)?指"
)
_USER_DISAMBIG_CONFIRM_RE = re.compile(
    r"(?:对|是的|没错|就是).*?(360p|480p|540p|720p|1080p|2160p|2k|4k|\d+\s*ms|\d+\s*毫秒|\d+\s*min|\d+\s*分钟)",
    re.IGNORECASE,
)

# Correction rule: 不是X，是Y
_CORRECTION_RE = re.compile(
    r"不是\s*([^，,。！!\s]{1,20})\s*[，,\s]*(?:而是|是)\s*([^，,。！!\n]{1,20})"
)

# Clear rule: 取消/清除/删掉 ... 偏好/默认配置
_CLEAR_RE = re.compile(
    r"(?:取消|清除|删除|忘掉|作废).*(?:偏好|默认|老规矩|设置|配置)"
)


def _clean_alias_target(raw_target: str) -> str:
    t = raw_target.strip().rstrip("。！!，,")
    for prefix in ("时延控制在", "画质档位", "画质就是", "时延上限就按", "时长按", "时间就按"):
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()
    for suffix in ("以内", "来", "算", "处理", "理解"):
        if t.endswith(suffix) and len(t) > len(suffix) + 1:
            t = t[: -len(suffix)].strip()
    return t


@dataclass
class FastPathResult:
    confirmed: List[Event] = field(default_factory=list)
    needs_llm: bool = True
    has_param_signal: bool = False
    has_concession: bool = False
    is_noise: bool = False


class FastPathProcessor:
    """Rule-based zero-LLM fast path for alias mapping, slot matching, corrections, and noise filtering."""

    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.data_dir = Path(data_dir) if data_dir else (repo_root / "data")
        self.app_map: Dict[str, List[str]] = self._load_json(self.data_dir / "app_map.json", {})
        self.app_config: Dict[str, Any] = self._load_json(self.data_dir / "app_config.json", {})
        self.support_matrix: Dict[str, List[str]] = self._load_json(self.data_dir / "support_matrix.json", {})

        self._alias_to_app: List[Tuple[str, str]] = []
        for canonical, aliases in self.app_map.items():
            for a in set([canonical, *aliases]):
                self._alias_to_app.append((a, canonical))
        self._alias_to_app.sort(key=lambda x: len(x[0]), reverse=True)

        self._alias_to_biz: List[Tuple[str, str]] = []
        for canonical_biz, aliases in (self.app_config.get("businesses") or {}).items():
            for a in set([canonical_biz, *aliases]):
                self._alias_to_biz.append((a, canonical_biz))
        self._alias_to_biz.sort(key=lambda x: len(x[0]), reverse=True)

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default
        return default

    def _detect_app_and_biz(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        cleaned = re.sub(r"(?:今天|这次|本次|先)?(?:不用|不开|不走|别用)[^，,。；;！!]*", "", text or "")
        low_clean = (cleaned or text).lower()
        low_full = (text or "").lower()
        found_app: Optional[str] = None
        for alias, canonical in self._alias_to_app:
            if alias.lower() in low_clean:
                found_app = canonical
                break
        if found_app is None:
            for alias, canonical in self._alias_to_app:
                if alias.lower() in low_full:
                    found_app = canonical
                    break
        found_biz: Optional[str] = None
        for alias, canonical_biz in self._alias_to_biz:
            if alias.lower() in low_clean:
                found_biz = canonical_biz
                break
        if found_biz is None:
            for alias, canonical_biz in self._alias_to_biz:
                if alias.lower() in low_full:
                    found_biz = canonical_biz
                    break
        return found_app, found_biz

    @staticmethod
    def _detect_validity(text: str) -> str:
        if any(
            w in text
            for w in (
                "临时",
                "暂时",
                "今天先",
                "这次先",
                "就今天",
                "就这次",
                "今天不用",
                "这次不用",
                "今天改在",
                "今天改用",
            )
        ):
            return "temporary"
        return "long_term"

    @staticmethod
    def _match_alias_in_text(text: str) -> Optional[Tuple[str, str]]:
        for pat in _ALIAS_RULE_PATTERNS:
            m = pat.search(text)
            if m:
                alias_str = m.group(1).strip(" '‘’\"“”「『」』")
                target_str = _clean_alias_target(m.group(2))
                if alias_str and target_str:
                    return alias_str, target_str
        return None

    def _try_working_memory_fastpath(
        self,
        numbered_turns: List[Dict[str, Any]],
        seen_aliases: Set[str],
        session_id: str,
        user_id: str,
        tenant_id: str,
    ) -> List[Event]:
        """Extract multi-turn disambiguation aliases where agent asks for clarification and user confirms."""
        events: List[Event] = []
        pending_disambig_alias: Optional[str] = None
        for t in numbered_turns:
            role = str(t.get("role", "user")).lower()
            txt = str(t.get("content") or t.get("text") or "").strip()
            turn_no = int(t.get("turn", 1))
            if role != "user":
                m_ag = _AGENT_DISAMBIG_RE.search(txt)
                if m_ag:
                    pending_disambig_alias = m_ag.group(1).strip(" '‘’\"“”")
            elif pending_disambig_alias:
                m_cf = _USER_DISAMBIG_CONFIRM_RE.search(txt)
                if m_cf and pending_disambig_alias not in seen_aliases:
                    target_val = m_cf.group(1).strip().lower().replace("毫秒", "ms").replace("分钟", "min")
                    seen_aliases.add(pending_disambig_alias)
                    events.append(
                        Event(
                            event_type=EventType.EXPRESSION_ALIAS.value,
                            subtype="expression",
                            memory_key_fields={"alias": pending_disambig_alias},
                            content={
                                "alias": pending_disambig_alias,
                                "source_expression": pending_disambig_alias,
                                "target": target_val,
                                "validity": "long_term",
                                "note": f"用户确认「{pending_disambig_alias}」表示「{target_val}」",
                            },
                            validity="long_term",
                            confidence=0.95,
                            source_session_id=session_id,
                            evidence_turns=[turn_no],
                            user_id=user_id,
                            tenant_id=tenant_id,
                        )
                    )
                pending_disambig_alias = None
        return events

    def _try_experience_rule_fastpath(
        self,
        turn_no: int,
        txt: str,
        seen_aliases: Set[str],
        is_conc: bool,
        session_id: str,
        user_id: str,
        tenant_id: str,
    ) -> Optional[Event]:
        """Match explicit expression alias or preference clear rules in a single user turn."""
        m_alias = self._match_alias_in_text(txt)
        if m_alias:
            alias_str, target_str = m_alias
            if alias_str not in seen_aliases:
                seen_aliases.add(alias_str)
                app, biz = self._detect_app_and_biz(target_str)
                content: Dict[str, Any] = {
                    "alias": alias_str,
                    "source_expression": alias_str,
                    "target": target_str,
                    "validity": "long_term",
                    "note": f"用户定义「{alias_str}」表示「{target_str}」",
                }
                if app:
                    content["application"] = app
                if biz:
                    content["business"] = biz
                return Event(
                    event_type=EventType.EXPRESSION_ALIAS.value,
                    subtype="expression",
                    memory_key_fields={"alias": alias_str},
                    content=content,
                    validity="long_term",
                    confidence=0.95,
                    source_session_id=session_id,
                    evidence_turns=[turn_no],
                    is_concession=is_conc,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )

        if _CLEAR_RE.search(txt):
            app, biz = self._detect_app_and_biz(txt)
            if app and biz:
                return Event(
                    event_type=EventType.PREFERENCE_CLEAR.value,
                    subtype="application_business",
                    memory_key_fields={"application": app, "business": biz},
                    content={"application": app, "business": biz, "note": txt},
                    validity="long_term",
                    confidence=0.95,
                    source_session_id=session_id,
                    evidence_turns=[turn_no],
                    user_id=user_id,
                    tenant_id=tenant_id,
                )
        return None

    def _try_factual_slot_fastpath(
        self,
        turn_no: int,
        txt: str,
        full_user_text: str,
        validity: str,
        is_conc: bool,
        session_id: str,
        user_id: str,
        tenant_id: str,
    ) -> Optional[Event]:
        """Match preference correction ('不是X，是Y') or explicit app+business slot statements."""
        m_corr = _CORRECTION_RE.search(txt)
        if m_corr:
            wrong_part, right_part = m_corr.group(1).strip(), m_corr.group(2).strip()
            app, biz = self._detect_app_and_biz(right_part)
            if not app or not biz:
                app2, biz2 = self._detect_app_and_biz(txt)
                app = app or app2
                biz = biz or biz2
            if app and biz and self._is_supported(app, biz):
                content: Dict[str, Any] = {
                    "application": app,
                    "business": biz,
                    "corrected_from": wrong_part,
                    "corrected_to": right_part,
                    "validity": validity,
                    "note": txt,
                }
                self._extract_inline_params(txt, content)
                return Event(
                    event_type=EventType.PREFERENCE_CORRECT.value,
                    subtype="application_business",
                    memory_key_fields={"application": app, "business": biz},
                    content=content,
                    validity=validity,
                    confidence=0.92,
                    source_session_id=session_id,
                    evidence_turns=[turn_no],
                    is_concession=is_conc,
                    user_id=user_id,
                    tenant_id=tenant_id,
                )

        app, biz = self._detect_app_and_biz(txt)
        if app and biz and self._is_supported(app, biz):
            content = {
                "application": app,
                "business": biz,
                "validity": validity,
                "note": txt,
            }
            self._extract_inline_params(txt, content)
            if validity != "temporary":
                tmp_full: Dict[str, Any] = {}
                self._extract_inline_params(full_user_text, tmp_full)
                for k_miss in ("resolution", "latency", "time_window", "priority"):
                    if not content.get(k_miss) and tmp_full.get(k_miss):
                        content[k_miss] = tmp_full[k_miss]
            return Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="application_business",
                memory_key_fields={"application": app, "business": biz},
                content=content,
                validity=validity,
                confidence=0.9,
                source_session_id=session_id,
                evidence_turns=[turn_no],
                is_concession=is_conc,
                user_id=user_id,
                tenant_id=tenant_id,
            )
        return None

    def _finalize_fastpath_decision(
        self,
        res: FastPathResult,
        numbered_turns: List[Dict[str, Any]],
        user_turns: List[Tuple[int, str]],
        full_user_text: str,
        session_id: str,
        user_id: str,
        tenant_id: str,
    ) -> FastPathResult:
        """Apply cross-turn fallback confirmation and decide whether LLM extraction is still required."""
        has_pref_ev = any(
            ev.event_type != EventType.EXPRESSION_ALIAS.value and ev.subtype != "expression"
            for ev in res.confirmed
        )
        if not has_pref_ev and len(user_turns) >= 2:
            u_app, u_biz = self._detect_app_and_biz(full_user_text)
            full_all_text = " ".join(str(t.get("content") or t.get("text") or "") for t in numbered_turns)
            if not u_app or not u_biz:
                a_app, a_biz = self._detect_app_and_biz(full_all_text)
                u_app = u_app or a_app
                u_biz = u_biz or a_biz
            if u_app and u_biz and self._is_supported(u_app, u_biz):
                val_all = self._detect_validity(full_user_text)
                c_all: Dict[str, Any] = {
                    "application": u_app,
                    "business": u_biz,
                    "validity": val_all,
                    "note": full_user_text,
                }
                self._extract_inline_params(full_user_text, c_all)
                if val_all != "temporary":
                    tmp_a: Dict[str, Any] = {}
                    self._extract_inline_params(full_all_text, tmp_a)
                    for k_miss in ("resolution", "latency", "time_window", "priority"):
                        if not c_all.get(k_miss) and tmp_a.get(k_miss):
                            c_all[k_miss] = tmp_a[k_miss]
                res.confirmed.append(
                    Event(
                        event_type=EventType.PREFERENCE_CONFIRM.value,
                        subtype="application_business",
                        memory_key_fields={"application": u_app, "business": u_biz},
                        content=c_all,
                        validity=val_all,
                        confidence=0.9,
                        source_session_id=session_id,
                        evidence_turns=[user_turns[-1][0]],
                        is_concession=res.has_concession,
                        user_id=user_id,
                        tenant_id=tenant_id,
                    )
                )

        res.needs_llm = not (res.confirmed and not res.has_param_signal and not res.has_concession)
        return res

    def process(
        self,
        session_turns: List[Dict[str, Any]],
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "default",
    ) -> FastPathResult:
        """Orchestrate zero-LLM fast-path stages and determine whether LLM extraction is needed."""
        res = FastPathResult()
        if not session_turns:
            res.is_noise = True
            res.needs_llm = False
            return res

        numbered_turns = Extractor._ensure_turn_numbers(session_turns)
        user_turns: List[Tuple[int, str]] = [
            (int(t.get("turn", idx)), str(t.get("content") or t.get("text") or "").strip())
            for idx, t in enumerate(numbered_turns, start=1)
            if str(t.get("role", "user")).lower() == "user"
            and str(t.get("content") or t.get("text") or "").strip()
        ]

        if not user_turns or all(_NOISE_EXACT_RE.match(txt) for _, txt in user_turns):
            res.is_noise = True
            res.needs_llm = False
            return res

        full_user_text = " ".join(txt for _, txt in user_turns)
        res.has_param_signal = bool(_PARAM_SIGNAL_RE.search(full_user_text))
        res.has_concession = bool(_CONCESSION_RE.search(full_user_text))

        seen_aliases: Set[str] = set()
        res.confirmed.extend(
            self._try_working_memory_fastpath(numbered_turns, seen_aliases, session_id, user_id, tenant_id)
        )

        for turn_no, txt in user_turns:
            validity = self._detect_validity(txt)
            is_conc = bool(_CONCESSION_RE.search(txt))

            rule_ev = self._try_experience_rule_fastpath(
                turn_no, txt, seen_aliases, is_conc, session_id, user_id, tenant_id
            )
            if rule_ev is not None:
                res.confirmed.append(rule_ev)
                continue

            slot_ev = self._try_factual_slot_fastpath(
                turn_no, txt, full_user_text, validity, is_conc, session_id, user_id, tenant_id
            )
            if slot_ev is not None:
                res.confirmed.append(slot_ev)

        return self._finalize_fastpath_decision(
            res, numbered_turns, user_turns, full_user_text, session_id, user_id, tenant_id
        )

    def _is_supported(self, app: str, biz: str) -> bool:
        if not self.support_matrix:
            return True
        allowed = [b.lower() for b in self.support_matrix.get(app, [])]
        return biz.lower() in allowed

    @staticmethod
    def _extract_inline_params(text: str, content: Dict[str, Any]) -> None:
        m_res = re.search(r"\b(360p|480p|540p|720p|1080p|2160p|2k|4k)\b", text, re.IGNORECASE)
        if m_res:
            content["resolution"] = m_res.group(1).lower()
        m_lat = re.search(r"(\d+)\s*(?:ms|毫秒)", text, re.IGNORECASE)
        if m_lat:
            content["latency"] = f"{m_lat.group(1)}ms"
        if "稳定优先" in text:
            content["priority"] = "stability"
        elif "画质优先" in text:
            content["priority"] = "quality"
        elif "低时延" in text or "时延优先" in text:
            content["priority"] = "latency"
        elif "流畅优先" in text:
            content["priority"] = "fluency"
        tw = extract_time_window_from_text(text)
        if tw:
            content["time_window"] = tw
