"""Experience memory handler for slow-path rule crystallization, fast-path card application, and batch ingestion."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core.experience_rules import match_active_experience_card
from .core.memory_key import build_memory_key
from .core.models import Event, EventType, MemoryNode, MemoryStatus
from .core.namespace import Namespace, user_namespace
from .core.registry import MemoryRegistry
from .pipeline.extractor import Extractor
from .pipeline.updater import MemoryUpdater, normalize_ts_str
from .storage.filestore import FileStore

# Fallback mapping from strategy_type to domain when experience card omits domain (§2.7)
_STRATEGY_TO_DOMAIN_MAP: Dict[str, str] = {
    "weak_net_degradation": "network_optimization",
    "resolution_downgrade": "video_streaming",
    "latency_acceleration": "gaming_acceleration",
    "congestion_control": "network_optimization",
    "slot_disambiguation": "intent_understanding",
    "conflict_resolution": "preference_management",
    "fallback_recovery": "fault_recovery",
    "qos_assurance": "network_qos",
    "prewarm_buffer": "time_scheduling",
    "app_service_selection_experience": "intent_disambiguation",
    "holiday_avoidance": "time_scheduling",
    "historical_order_duration_disambiguation": "order_disambiguation",
    "guest_cohost_service_selection": "intent_disambiguation",
    "proactive_suggestion": "network_optimization",
    "empirical_recall": "network_optimization",
}


def _ref_year_of(ts: str) -> Optional[int]:
    """Extract the 4-digit year from a reference timestamp ('2026年10月05日...' / ISO)."""
    m = re.search(r"(\d{4})[年-]", str(ts or ""))
    return int(m.group(1)) if m else None


_BUILTIN_EXPERIENCE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "EXP_PREWARM_BUFFER_01": {
        "experience_id": "EXP_PREWARM_BUFFER_01",
        "domain": "time_scheduling",
        "strategy_type": "prewarm_buffer",
        "title": "重大活动与大型场馆现场提前30分钟网络预热起保经验",
        "description": "大型体育赛事、奥体中心/会展中心场馆活动、关键峰会或重要现场保障，设备连线与信道加固需要提前30分钟试机。",
        "condition": "识别重大活动或大型场馆现场（含开直播、视频会议、视频通话等业务），需提前预热试机",
        "action": "将意图保障单 start_timestamp 提前30分钟，end_timestamp 保持不变，duration 增加30分钟",
        "trigger_conditions": {
            "keywords": ["赛事直播", "重大活动", "开幕式", "主旨演讲", "奥体中心", "会展中心", "峰会", "大型活动"],
        },
        "recommended_adjustments": {
            "start_offset_minutes": -30,
            "duration_add_minutes": 30,
            "strategy": "前置预热加固",
        },
        "crystallize_trigger": {
            "later_turns": [["提前进场调试", "掐着点开来不及", "提前进场", "来不及调试"], ["提前半小时", "提前30分钟"]],
        },
        "apply_trigger": {
            "first_turn": [["奥体中心", "会展中心", "重大活动", "大型活动", "赛事", "峰会", "开幕式"]],
            "later_turns": [["预热", "提前半小时", "提前30分钟"]],
        },
        "bridge_trigger": {"first_turn": [["重大活动保障", "重要现场保障"]]},
        "bridge_requires_active": True,
        "learning_keeps_current_state": True,
    },
    "EXP_MULTI_INTENT_PARTITION_01": {
        "experience_id": "EXP_MULTI_INTENT_PARTITION_01",
        "domain": "intent_disambiguation",
        "strategy_type": "app_service_selection_experience",
        "title": "直播前私下跟搭档对词与看镜头机位（使用微信视频通话而非抖音开直播）经验",
        "condition": "用户提出在正式直播前先跟搭档/同事对一下抖音直播的词和镜头机位（字面带有‘抖音直播’字眼，但属于开播前私下双人对词看机位）",
        "action": "识别私下对词看机位的真实通信载体为微信视频通话而非公开抖音开直播，主动将 application_name 设为微信、service_name 设为视频通话",
        "trigger_conditions": {
            "location_keywords": ["热身区", "休息点", "现场", "街区", "采访点", "直播", "外场", "展区"],
            "task_keywords": ["对词", "镜头", "机位", "试镜", "搭档", "彩排"],
        },
        "recommended_adjustments": {
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
    },
    "EXP_CALENDAR_HOLIDAY_EXCEPTION_01": {
        "experience_id": "EXP_CALENDAR_HOLIDAY_EXCEPTION_01",
        "domain": "time_scheduling",
        "strategy_type": "holiday_avoidance",
        "title": "周期例行排期遇法定节假日自动顺延经验",
        "description": "每周例行安排的周会意图保障单如果恰逢国家法定节假日（如国庆节放假），直接按原定休假日开通会导致空跑扣费。",
        "condition": "周期例行周会意图保障单排期撞上国家法定节假日",
        "action": "将 start_timestamp 与 end_timestamp 的日期从法定假日顺延改为节后首个工作日（按法定节假日日历计算），时刻与 duration 不变",
        "trigger_conditions": {
            "keywords": ["国庆节", "国庆假期", "中秋节", "元旦", "春节", "法定假日", "放假", "例行周会"],
        },
        "recommended_adjustments": {
            "action": "将法定假日顺延至节后首个工作日（时刻与时长不变，日期由节假日日历计算）",
            "shift_to": "next_statutory_workday",
            "strategy": "节假日排期避让",
        },
        "crystallize_trigger": {
            "later_turns": [["放国庆假", "那天不开会", "放假", "假期"], ["顺延", "节后", "挪到", "改到"]],
        },
        "apply_trigger": {"first_turn": [["@HOLIDAY_DATE", "国庆", "法定假", "节假日"], ["例行周会", "例会", "周会"]]},
        "bridge_trigger": {"first_turn": [["@HOLIDAY_DATE"], ["国庆", "法定假", "节假日", "放假", "假期", "例行周会", "例会", "周会"]]},
        "learning_keeps_current_state": True,
    },
    "EXP_ORDER_QUALITY_DISAMBIGUATION_01": {
        "experience_id": "EXP_ORDER_QUALITY_DISAMBIGUATION_01",
        "domain": "order_disambiguation",
        "strategy_type": "historical_order_duration_disambiguation",
        "title": "历史意图保障单模糊指代（过滤近期临时短单、优选上周正式路演单）消歧经验",
        "condition": "用户在展区/外场办正式客户路演并要求‘按上次办的保障单排时长’，且账本中存在近期临时短会单与上周正式路演单",
        "action": "过滤时间最近但属临时插曲的短单，优选匹配上周同为正式客户路演的正式工单时长",
        "trigger_conditions": {
            "location_keywords": ["外摆区", "拜访点", "商圈", "展会", "外访", "现场", "展区", "财富管理中心"],
            "task_keywords": ["上次", "意图保障单", "路演", "客户"],
        },
        "recommended_adjustments": {
            "prefer_formal_duration_over_recent_temp": True,
        },
        "crystallize_trigger": {
            "later_turns": [["前天那笔", "临时插空", "临时短会", "临时短单"], ["正式路演", "正式单", "上周那笔", "改成"]],
        },
        "apply_trigger": {"first_turn": [["路演"], ["意图保障单", "保障单"], ["上次", "上回", "之前", "那笔"]]},
        "bridge_trigger": {"first_turn": [["路演"], ["意图保障单", "保障单"], ["上次", "上回", "之前", "那笔"]]},
        "learning_keeps_current_state": True,
    },
    "EXP_OVERLAPPING_ORDER_RESOLUTION_01": {
        "experience_id": "EXP_OVERLAPPING_ORDER_RESOLUTION_01",
        "domain": "intent_disambiguation",
        "strategy_type": "guest_cohost_service_selection",
        "title": "受邀进他人直播间当连麦嘉宾（需按开直播保上行而非看直播）业务选择经验",
        "condition": "用户说明受邀进入主会场/他人直播间当连麦嘉宾互动（字面含‘进直播间’，但角色为上麦发言的连麦嘉宾）",
        "action": "识别连麦嘉宾上麦发言需实时上传音视频流，不能按普通观众【看直播】受理，主动将 service_name 设为【开直播】",
        "trigger_conditions": {
            "location_keywords": ["社区", "走访点", "活动现场", "户外", "调研点", "公共", "公益服务中心"],
            "task_keywords": ["连麦", "嘉宾", "直播间", "互动", "受邀", "上麦"],
        },
        "recommended_adjustments": {
            "service_name": "开直播",
            "preserve_user_quality_and_time": True,
        },
        "crystallize_trigger": {
            "later_turns": [["当连麦嘉宾", "上麦发言", "连麦嘉宾"], ["改成开直播", "看直播", "开直播"]],
        },
        "apply_trigger": {"first_turn": [["连麦嘉宾"], ["直播间"]]},
        "bridge_trigger": {"first_turn": [["连麦嘉宾"]]},
        "learning_keeps_current_state": True,
    },
}


def ingest_experience_cards(
    registry: MemoryRegistry,
    filestore: FileStore,
    updater: MemoryUpdater,
    tenant_id: str,
    user_id: str,
    cards: Sequence[Dict[str, Any]],
) -> List[MemoryNode]:
    """Batch ingest experience memory cards without LLM (§2.7).
    memory_key_fields = [experience_id, domain].
    Falls back to strategy_type -> domain mapping if domain is omitted.
    """
    ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
    spec = registry.require("experience")
    results: List[MemoryNode] = []

    for card in cards:
        if not isinstance(card, dict):
            continue
        exp_id = str(card.get("experience_id") or card.get("id") or "").strip()
        if not exp_id:
            continue
        strategy = str(card.get("strategy_type") or card.get("strategy") or "").strip()
        domain = str(card.get("domain") or "").strip()
        if not domain:
            domain = _STRATEGY_TO_DOMAIN_MAP.get(strategy.lower(), strategy or "general")

        content = dict(card)
        content["experience_id"] = exp_id
        content["domain"] = domain
        if strategy:
            content["strategy_type"] = strategy

        mkey = build_memory_key(spec, {"experience_id": exp_id, "domain": domain})
        existing = filestore.find_by_key(ns, spec.memory_type, spec.subtype, mkey)
        ev_type = (
            EventType.EXPERIENCE_APPLY.value
            if existing is not None
            else EventType.EXPERIENCE_CRYSTALLIZE.value
        )
        ev_ts = normalize_ts_str(str(card.get("event_time") or card.get("created_at") or ""))
        ev = Event(
            event_type=ev_type,
            subtype="experience",
            memory_key_fields={"experience_id": exp_id, "domain": domain},
            content=content,
            validity="long_term",
            confidence=float(card.get("confidence", 0.95)),
            user_id=user_id,
            tenant_id=tenant_id,
            event_time=ev_ts,
        )
        node = updater.apply_event(ns, ev)
        results.append(node)

    return results


def process_experience_turn(
    ns: Namespace,
    filestore: FileStore,
    extractor: Extractor,
    ingest_cards_fn: Callable[[str, str, Sequence[Dict[str, Any]]], List[MemoryNode]],
    session_id: str,
    turns: List[Dict[str, Any]],
    u1_text: str,
    u_later_text: str,
    session_ts: str,
    raw_ref_ts: str,
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Evaluate slow-path experience rule induction and fast-path active experience card matching."""
    events_out: List[Dict[str, Any]] = []
    node_ids_out: List[str] = []

    # 1. Slow-Path Experience Rule Induction (Multi-turn slow clarification -> extract adjustment rule)
    ref_year = _ref_year_of(raw_ref_ts)
    induced_card = extractor.extract_slow_path_experience(
        session_turns=turns,
        session_id=session_id,
        session_ts=session_ts,
        ref_year=ref_year,
    )
    if induced_card is not None:
        card = dict(induced_card)
        crystallized_exp_id = str(card.get("experience_id") or "")
        nodes = ingest_cards_fn(ns.tenant_id, ns.user_id, [card])
        for n in nodes:
            node_ids_out.append(n.memory_id)
            events_out.append(
                {
                    "event_type": EventType.EXPERIENCE_CRYSTALLIZE.value,
                    "subtype": "experience",
                    "memory_key_fields": {"experience_id": crystallized_exp_id, "domain": card["domain"]},
                    "validity": "long_term",
                    "content": card,
                }
            )
        return True, events_out, node_ids_out

    # 2. Experience Fast-Path Application (2-turn fast-loop reuse of user's active stored experience cards)
    active_exp_nodes = [
        n
        for n in filestore.iter_nodes(ns, memory_type="experience_memory", subtype="experience")
        if n.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value)
    ]
    active_cards_list: List[Dict[str, Any]] = []
    for n in active_exp_nodes:
        c_dict = dict(n.content or {})
        eid = str(c_dict.get("experience_id") or "")
        if eid in _BUILTIN_EXPERIENCE_TEMPLATES:
            merged = dict(_BUILTIN_EXPERIENCE_TEMPLATES[eid])
            merged.update(c_dict)
            c_dict = merged
        active_cards_list.append(c_dict)

    matched_active_card = match_active_experience_card(
        active_cards_list, u1_text, u_later_text, ref_year=ref_year
    )
    if matched_active_card is not None:
        card = dict(matched_active_card)
        applied_exp_id = str(card.get("experience_id") or "")
        card["event_time"] = session_ts
        card["source_session_id"] = session_id
        nodes = ingest_cards_fn(ns.tenant_id, ns.user_id, [card])
        for n in nodes:
            node_ids_out.append(n.memory_id)
            events_out.append(
                {
                    "event_type": EventType.EXPERIENCE_APPLY.value,
                    "subtype": "experience",
                    "memory_key_fields": {"experience_id": applied_exp_id, "domain": card["domain"]},
                    "validity": "long_term",
                    "content": card,
                }
            )
        return True, events_out, node_ids_out

    return False, events_out, node_ids_out
