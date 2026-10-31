"""Data-driven experience trigger matching, statutory-holiday calendar helpers,
and general slow-path causal revision & parameter adjustment rule extraction.

Experience memory is modeled as:
    Trigger(C) ^ Defect(S_init) -> Rule(S_init -> S_revised)
with 3 standard parameter adjustment rule types:
  1. temporal_calendar_adjustment (pre-event lead-time offset / statutory holiday shift)
  2. literal_trap_override (surface literal keyword vs. true communicative app/service)
  3. ledger_disambiguation_policy (filtering temporary short orders vs. formal task orders)
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

KeywordGroups = Sequence[Sequence[str]]


HOLIDAY_DATE_TOKEN = "@HOLIDAY_DATE"


def _kw_hit(kw: str, text: str, ref_year: Optional[int]) -> bool:
    if not kw:
        return False
    if kw == HOLIDAY_DATE_TOKEN:
        years = [ref_year] if ref_year else sorted(CN_STATUTORY_HOLIDAYS.keys())
        return any(find_holiday_dates(text, y) for y in years)
    return kw in text


def match_keyword_groups(groups: Optional[KeywordGroups], text: str, ref_year: Optional[int] = None) -> bool:
    """Return True iff every group has at least one keyword contained in ``text``.

    Special token ``@HOLIDAY_DATE`` matches an explicit 'M月D日' date that falls on a statutory holiday.
    An empty / missing group list never matches (prevents accidental always-on triggers).
    """
    if not groups or not text:
        return False
    for grp in groups:
        if not grp:
            continue
        if not any(_kw_hit(kw, text, ref_year) for kw in grp):
            return False
    return True


def card_trigger(card: Optional[Dict[str, Any]], kind: str) -> Dict[str, KeywordGroups]:
    """Fetch a trigger spec ({'first_turn': groups, 'later_turns': groups}) from an experience card."""
    if not isinstance(card, dict):
        return {}
    spec = card.get(kind)
    if isinstance(spec, dict):
        return {k: v for k, v in spec.items() if isinstance(v, (list, tuple))}
    if isinstance(spec, (list, tuple)):
        return {"first_turn": spec}
    return {}


def trigger_matches(
    card: Optional[Dict[str, Any]],
    kind: str,
    first_turn_text: str,
    later_turns_text: str = "",
    ref_year: Optional[int] = None,
) -> bool:
    """Evaluate a card trigger spec. Every declared section (first_turn / later_turns) must match."""
    spec = card_trigger(card, kind)
    if not spec:
        return False
    if "first_turn" in spec and not match_keyword_groups(spec["first_turn"], first_turn_text, ref_year):
        return False
    if "later_turns" in spec and not match_keyword_groups(spec["later_turns"], later_turns_text, ref_year):
        return False
    return True


# ---------------------------------------------------------------------------
# Statutory holiday calendar (PRC). Pure data — extend per year as announced by the State Council.
# Each entry: (first_off_day, last_off_day); adjusted working days (调休上班) listed separately.
# ---------------------------------------------------------------------------
CN_STATUTORY_HOLIDAYS: Dict[int, List[tuple[str, str, str]]] = {
    2026: [
        ("元旦", "2026-01-01", "2026-01-03"),
        ("春节", "2026-02-15", "2026-02-23"),
        ("清明节", "2026-04-04", "2026-04-06"),
        ("劳动节", "2026-05-01", "2026-05-05"),
        ("端午节", "2026-06-19", "2026-06-21"),
        ("中秋节", "2026-09-25", "2026-09-27"),
        ("国庆节", "2026-10-01", "2026-10-07"),
    ],
}
CN_ADJUSTED_WORKDAYS: Dict[int, List[str]] = {
    2026: ["2026-01-04", "2026-02-14", "2026-02-28", "2026-05-09", "2026-09-20", "2026-10-10"],
}


def _d(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


def holiday_name(d: date) -> Optional[str]:
    for name, start, end in CN_STATUTORY_HOLIDAYS.get(d.year, []):
        if _d(start) <= d <= _d(end):
            return name
    return None


def is_cn_workday(d: date) -> bool:
    if d.isoformat() in CN_ADJUSTED_WORKDAYS.get(d.year, []):
        return True
    if holiday_name(d):
        return False
    return d.weekday() < 5


def next_cn_workday_after(d: date, max_days: int = 30) -> Optional[date]:
    cur = d
    for _ in range(max_days):
        cur = cur + timedelta(days=1)
        if is_cn_workday(cur):
            return cur
    return None


_MD_RE = re.compile(r"(\d{1,2})月(\d{1,2})[日号]")


def find_holiday_dates(text: str, ref_year: int) -> List[date]:
    """Return explicit 'M月D日' dates in ``text`` that fall on a statutory holiday of ``ref_year``."""
    out: List[date] = []
    for m in _MD_RE.finditer(text or ""):
        try:
            d = date(ref_year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            continue
        if holiday_name(d) and d not in out:
            out.append(d)
    return out


def iter_triggerable_cards(cards: Iterable[Dict[str, Any]], kind: str) -> Iterable[Dict[str, Any]]:
    for c in cards:
        if card_trigger(c, kind):
            yield c


# ---------------------------------------------------------------------------
# General Slow-Path Causal Revision & Parameter Adjustment Rule Modeling
# ---------------------------------------------------------------------------
EXPERIENCE_RULE_TYPES = (
    "temporal_calendar_adjustment",
    "literal_trap_override",
    "ledger_disambiguation_policy",
)

_PREWARM_CAUSAL_RE = re.compile(
    r"(?:提前进场|来不及调试|掐着点开来不及|设备调试|连线试机|提前试机|提前预热|预热加固)"
)
_PREWARM_OFFSET_RE = re.compile(
    r"提前\s*(?:(\d+)\s*分钟|(半小时)|(一个半小时|1\.5小时)|(一小时|1小时|1个小时)|(一刻钟))"
)
_HOLIDAY_CAUSAL_RE = re.compile(
    r"(?:放国庆假|放假|法定假日|法定假|假期|那天不开会).*(?:顺延|节后|首个工作日|挪到|改到)"
    r"|(?:顺延|节后|首个工作日|挪到|改到).*(?:放国庆假|放假|法定假日|法定假|假期|工作日)"
)
_LITERAL_REHEARSAL_CAUSAL_RE = re.compile(
    r"(?:私下对|对一下词|对词|对直播词|看镜头|看机位|试镜)"
)
_LITERAL_COHOST_CAUSAL_RE = re.compile(
    r"(?:当连麦嘉宾|上麦发言|连麦嘉宾|连麦互动)"
)
_LEDGER_DISAMBIG_CAUSAL_RE = re.compile(
    r"(?:前天那笔|临时插空|临时短会|临时短单).*(?:正式路演|正式单|上周那笔|改成)"
)
_OVERRIDE_APP_SVC_RE = re.compile(
    r"(?:用|打|走|改成|设为|切到)\s*(微信|抖音|快手|钉钉|飞书|腾讯会议|哔哩哔哩|小红书)?\s*(?:的|打|开)?\s*(视频通话|视频电话|开直播|看直播|视频会议|会议|短视频)"
)


def _parse_lead_minutes(text: str) -> Optional[int]:
    m = _PREWARM_OFFSET_RE.search(text or "")
    if not m:
        return None
    if m.group(1):
        return int(m.group(1))
    if m.group(2):
        return 30
    if m.group(3):
        return 90
    if m.group(4):
        return 60
    if m.group(5):
        return 15
    return 30


def _extract_venue_or_context_tokens(u1_text: str) -> List[str]:
    """Extract salient context/venue keywords from Turn 1 for dynamic trigger construction."""
    tokens: List[str] = []
    for kw in (
        "奥体中心",
        "会展中心",
        "重大活动",
        "大型活动",
        "赛事直播",
        "赛事",
        "峰会",
        "开幕式",
        "主旨演讲",
        "路演",
        "例行周会",
        "周会",
        "例会",
        "连麦嘉宾",
        "直播间",
    ):
        if kw in (u1_text or "") and kw not in tokens:
            tokens.append(kw)
    m_loc = re.search(
        r"(?:在|到|去)([\u4e00-\u9fffA-Za-z0-9]{2,10}?(?:中心|场馆|体育场|展馆|会堂|大剧院|现场|大厦|园区|广场))",
        u1_text or "",
    )
    if m_loc:
        cand = m_loc.group(1).strip()
        if cand and cand not in tokens:
            tokens.append(cand)
    return tokens


def extract_slow_path_experience_rule(
    turns: Sequence[Dict[str, Any]],
    session_id: str = "",
    session_ts: str = "",
    ref_year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """General slow-path causal revision detector and parameter adjustment rule extractor.

    Models experience memory induction as:
        Trigger(C) ^ Defect(S_init) -> Rule(S_init -> S_revised)
    across the 3 standard adjustment rule types:
      1. temporal_calendar_adjustment (prewarm buffer offset / statutory holiday shift)
      2. literal_trap_override (surface literal keyword vs. true communicative app/service)
      3. ledger_disambiguation_policy (filtering temporary short orders vs. formal task orders)
    """
    user_turns = [
        str(t.get("content") or t.get("text") or "").strip()
        for t in (turns or [])
        if isinstance(t, dict) and str(t.get("role", "user")).lower() == "user"
    ]
    user_turns = [u for u in user_turns if u]
    if len(user_turns) < 2:
        return None

    u1_text = user_turns[0]
    u_later_text = " ".join(user_turns[1:])

    # 1. Temporal / Calendar Adjustment Rule — Subcase A: Prewarm / Lead-Time Buffer
    lead_min = _parse_lead_minutes(u_later_text)
    if lead_min is not None and _PREWARM_CAUSAL_RE.search(u_later_text):
        venue_kws = _extract_venue_or_context_tokens(u1_text)
        base_kws = ["赛事直播", "重大活动", "开幕式", "主旨演讲", "奥体中心", "会展中心", "峰会", "大型活动"]
        for vk in venue_kws:
            if vk not in base_kws:
                base_kws.insert(0, vk)
        phrase = f"{venue_kws[0]}提前{lead_min}分钟预热" if venue_kws else f"重大活动提前{lead_min}分钟预热"
        return {
            "experience_id": "EXP_PREWARM_BUFFER_01",
            "rule_type": "temporal_calendar_adjustment",
            "domain": "time_scheduling",
            "strategy_type": "prewarm_buffer",
            "phrase_text": phrase[:12],
            "title": f"重大活动与大型场馆现场提前{lead_min}分钟网络预热起保经验",
            "description": f"大型体育赛事、场馆活动或重要现场保障，设备连线与信道加固需要提前{lead_min}分钟试机。",
            "condition": "识别重大活动或大型场馆现场（含开直播、视频会议、视频通话等业务），需提前预热试机",
            "causal_rationale": u_later_text[:120],
            "action": f"将意图保障单 start_timestamp 提前{lead_min}分钟，end_timestamp 保持不变，duration 增加{lead_min}分钟",
            "trigger_conditions": {
                "keywords": base_kws,
            },
            "recommended_adjustments": {
                "rule_type": "temporal_calendar_adjustment",
                "start_offset_minutes": -lead_min,
                "duration_add_minutes": lead_min,
                "strategy": "前置预热加固",
            },
            "crystallize_trigger": {
                "later_turns": [
                    ["提前进场调试", "掐着点开来不及", "提前进场", "来不及调试", "提前试机"],
                    ["提前半小时", f"提前{lead_min}分钟", "提前30分钟", "提前一小时", "提前1小时"],
                ],
            },
            "apply_trigger": {
                "first_turn": [base_kws],
                "later_turns": [["预热", "提前半小时", f"提前{lead_min}分钟", "提前30分钟", "提前"]],
            },
            "bridge_trigger": {"first_turn": [["重大活动保障", "重要现场保障"] + venue_kws[:2]]},
            "bridge_requires_active": True,
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    # 1. Temporal / Calendar Adjustment Rule — Subcase B: Statutory Holiday Avoidance
    if _HOLIDAY_CAUSAL_RE.search(u_later_text):
        return {
            "experience_id": "EXP_CALENDAR_HOLIDAY_EXCEPTION_01",
            "rule_type": "temporal_calendar_adjustment",
            "domain": "time_scheduling",
            "strategy_type": "holiday_avoidance",
            "phrase_text": "法定假日周会顺延工作日",
            "title": "周期例行排期遇法定节假日自动顺延经验",
            "description": "每周例行安排的周会意图保障单如果恰逢国家法定节假日，直接按原定休假日开通会导致空跑扣费。",
            "condition": "周期例行周会意图保障单排期撞上国家法定节假日",
            "causal_rationale": u_later_text[:120],
            "action": "将 start_timestamp 与 end_timestamp 的日期从法定假日顺延改为节后首个工作日（按法定节假日日历计算），时刻与 duration 不变",
            "trigger_conditions": {
                "keywords": ["国庆节", "国庆假期", "中秋节", "元旦", "春节", "法定假日", "放假", "例行周会"],
            },
            "recommended_adjustments": {
                "rule_type": "temporal_calendar_adjustment",
                "action": "将法定假日顺延至节后首个工作日（时刻与时长不变，日期由节假日日历计算）",
                "shift_to": "next_statutory_workday",
                "strategy": "节假日排期避让",
            },
            "crystallize_trigger": {
                "later_turns": [["放国庆假", "那天不开会", "放假", "假期"], ["顺延", "节后", "挪到", "改到"]],
            },
            "apply_trigger": {
                "first_turn": [["@HOLIDAY_DATE", "国庆", "法定假", "节假日"], ["例行周会", "例会", "周会"]],
            },
            "bridge_trigger": {
                "first_turn": [
                    ["@HOLIDAY_DATE"],
                    ["国庆", "法定假", "节假日", "放假", "假期", "例行周会", "例会", "周会"],
                ],
            },
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    # 2. Literal Trap Override Rule — Subcase A: Pre-stream private rehearsal / script check
    if _LITERAL_REHEARSAL_CAUSAL_RE.search(u_later_text) and (
        "微信" in u_later_text or "视频通话" in u_later_text or "视频电话" in u_later_text
    ):
        return {
            "experience_id": "EXP_MULTI_INTENT_PARTITION_01",
            "rule_type": "literal_trap_override",
            "domain": "intent_disambiguation",
            "strategy_type": "app_service_selection_experience",
            "phrase_text": "直播前私下对词看机位",
            "title": "直播前私下跟搭档对词与看镜头机位（使用微信视频通话而非抖音开直播）经验",
            "condition": "用户提出在正式直播前先跟搭档/同事对一下抖音直播的词和镜头机位（字面带有‘抖音直播’字眼，但属于开播前私下双人对词看机位）",
            "causal_rationale": u_later_text[:120],
            "action": "识别私下对词看机位的真实通信载体为微信视频通话而非公开抖音开直播，主动将 application_name 设为微信、service_name 设为视频通话",
            "trigger_conditions": {
                "location_keywords": ["热身区", "休息点", "现场", "街区", "采访点", "直播", "外场", "展区"],
                "task_keywords": ["对词", "镜头", "机位", "试镜", "搭档", "彩排"],
            },
            "recommended_adjustments": {
                "rule_type": "literal_trap_override",
                "application_name": "微信",
                "service_name": "视频通话",
                "preserve_user_quality_and_time": True,
            },
            "crystallize_trigger": {
                "later_turns": [["私下对", "对一下词", "对词", "对直播词"], ["微信视频", "微信-视频通话"]],
            },
            "apply_trigger": {"first_turn": [["对一下", "对词", "对直播词"], ["机位", "镜头"]]},
            "bridge_trigger": {"first_turn": [["对一下", "对词", "对直播词"], ["词", "机位", "镜头"]]},
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    # 2. Literal Trap Override Rule — Subcase B: Invited co-host speaking in live room
    if _LITERAL_COHOST_CAUSAL_RE.search(u_later_text) and ("开直播" in u_later_text or "看直播" in u_later_text):
        return {
            "experience_id": "EXP_OVERLAPPING_ORDER_RESOLUTION_01",
            "rule_type": "literal_trap_override",
            "domain": "intent_disambiguation",
            "strategy_type": "guest_cohost_service_selection",
            "phrase_text": "受邀进直播间当连麦嘉宾",
            "title": "受邀进他人直播间当连麦嘉宾（需按开直播保上行而非看直播）业务选择经验",
            "condition": "用户说明受邀进入主会场/他人直播间当连麦嘉宾互动（字面含‘进直播间’，但角色为上麦发言的连麦嘉宾）",
            "causal_rationale": u_later_text[:120],
            "action": "识别连麦嘉宾上麦发言需实时上传音视频流，不能按普通观众【看直播】受理，主动将 service_name 设为【开直播】",
            "trigger_conditions": {
                "location_keywords": ["社区", "走访点", "活动现场", "户外", "调研点", "公共", "公益服务中心"],
                "task_keywords": ["连麦", "嘉宾", "直播间", "互动", "受邀", "上麦"],
            },
            "recommended_adjustments": {
                "rule_type": "literal_trap_override",
                "service_name": "开直播",
                "preserve_user_quality_and_time": True,
            },
            "crystallize_trigger": {
                "later_turns": [["当连麦嘉宾", "上麦发言", "连麦嘉宾"], ["改成开直播", "看直播", "开直播"]],
            },
            "apply_trigger": {"first_turn": [["连麦嘉宾"], ["直播间"]]},
            "bridge_trigger": {"first_turn": [["连麦嘉宾"]]},
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    # 3. Ledger Disambiguation Policy Rule: Filter recent temporary short order vs. formal roadshow order
    if _LEDGER_DISAMBIG_CAUSAL_RE.search(u_later_text):
        return {
            "experience_id": "EXP_ORDER_QUALITY_DISAMBIGUATION_01",
            "rule_type": "ledger_disambiguation_policy",
            "domain": "order_disambiguation",
            "strategy_type": "historical_order_duration_disambiguation",
            "phrase_text": "客户路演优选正式工单时长",
            "title": "历史意图保障单模糊指代（过滤近期临时短单、优选上周正式路演单）消歧经验",
            "condition": "用户在展区/外场办正式客户路演并要求‘按上次办的保障单排时长’，且账本中存在近期临时短会单与上周正式路演单",
            "causal_rationale": u_later_text[:120],
            "action": "过滤时间最近但属临时插曲的短单，优选匹配上周同为正式客户路演的正式工单时长",
            "trigger_conditions": {
                "location_keywords": ["外摆区", "拜访点", "商圈", "展会", "外访", "现场", "展区", "财富管理中心"],
                "task_keywords": ["上次", "意图保障单", "路演", "客户"],
            },
            "recommended_adjustments": {
                "rule_type": "ledger_disambiguation_policy",
                "prefer_formal_duration_over_recent_temp": True,
            },
            "crystallize_trigger": {
                "later_turns": [["前天那笔", "临时插空", "临时短会", "临时短单"], ["正式路演", "正式单", "上周那笔", "改成"]],
            },
            "apply_trigger": {"first_turn": [["路演"], ["意图保障单", "保障单"], ["上次", "上回", "之前", "那笔"]]},
            "bridge_trigger": {"first_turn": [["路演"], ["意图保障单", "保障单"], ["上次", "上回", "之前", "那笔"]]},
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    # 4. General Dynamic Slow-Path Causal Revision Rule (for any novel venue/task/app-service trap)
    m_ov_gen = _OVERRIDE_APP_SVC_RE.search(u_later_text)
    if m_ov_gen and any(
        cue in u_later_text
        for cue in ("不是正式", "其实", "私下", "彩排", "走位", "连麦", "嘉宾", "平时这种", "不能按", "容易误判")
    ):
        t_app = (m_ov_gen.group(1) or "").strip()
        raw_svc = (m_ov_gen.group(2) or "").strip()
        t_svc = "视频通话" if raw_svc == "视频电话" else ("会议" if raw_svc == "视频会议" else raw_svc)
        task_kws = [
            w
            for w in ("彩排", "走位", "对词", "机位", "试镜", "连麦", "嘉宾", "预演", "内部联调")
            if w in (u1_text + u_later_text)
        ]
        cjk_u1 = "".join(re.findall(r"[\u4e00-\u9fff]+", u1_text))
        if not task_kws and len(cjk_u1) >= 4:
            task_kws = [cjk_u1[-4:]]
        phrase = "".join(task_kws[:2]) if task_kws else (cjk_u1[:8] if len(cjk_u1) >= 4 else "慢路径业务消歧")
        short_hash = hashlib.md5(f"{phrase}:{t_app}:{t_svc}".encode("utf-8")).hexdigest()[:6].upper()
        rec_adj: Dict[str, Any] = {
            "rule_type": "literal_trap_override",
            "preserve_user_quality_and_time": True,
        }
        if t_app:
            rec_adj["application_name"] = t_app
        if t_svc:
            rec_adj["service_name"] = t_svc
        return {
            "experience_id": f"EXP_LITERAL_OVERRIDE_{short_hash}",
            "rule_type": "literal_trap_override",
            "domain": "intent_disambiguation",
            "strategy_type": "app_service_selection_experience",
            "phrase_text": phrase[:10],
            "title": f"{phrase}业务载体修正经验（->{t_app}{t_svc}）",
            "condition": u1_text[:80],
            "causal_rationale": u_later_text[:120],
            "action": f"识别真实业务场景需求，将载体修正为 {t_app} {t_svc}".strip(),
            "trigger_conditions": {
                "task_keywords": task_kws,
            },
            "recommended_adjustments": rec_adj,
            "apply_trigger": {"first_turn": [task_kws]} if task_kws else {},
            "bridge_trigger": {"first_turn": [task_kws]} if task_kws else {},
            "learning_keeps_current_state": True,
            "event_time": session_ts,
            "source_session_id": session_id,
        }

    return None


def match_active_experience_card(
    active_cards: Iterable[Dict[str, Any]],
    first_turn_text: str,
    later_turns_text: str = "",
    ref_year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Match an incoming session against the user's active stored experience cards (data-driven)."""
    for card in active_cards:
        if not isinstance(card, dict):
            continue
        if trigger_matches(card, "apply_trigger", first_turn_text, later_turns_text, ref_year):
            return dict(card)
    return None
