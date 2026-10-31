"""Session event ingestion, orthogonal Scheme A decomposition, and cross-session induction helper."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..pipeline.extractor import Extractor
from ..pipeline.induction import InductionEngine
from ..pipeline.updater import normalize_ts_str
from ..storage.filestore import FileStore
from ..storage.index import MemoryIndex
from .models import Event, EventType, MemoryNode, MemoryStatus
from .namespace import Namespace, user_namespace

# Category markers (not sample venue names) for one-off event venues that must not fork scene nodes.
_TRANSIENT_SCENE_MARKERS = ("活动现场", "展会现场", "赛事现场", "常规周期")

_CANONICAL_APP_CN: Dict[str, str] = {
    "douyin": "抖音",
    "kuaishou": "快手",
    "bilibili": "哔哩哔哩",
    "wechat": "微信",
    "tencent_meeting": "腾讯会议",
    "dingtalk": "钉钉",
    "feishu": "飞书",
    "xiaohongshu": "小红书",
    "wangzherongyao": "王者荣耀",
    "gaode": "高德地图",
    "huya": "虎牙",
    "douyu": "斗鱼",
    "hepingjingying": "和平精英",
    "qq": "QQ",
    "taobao": "淘宝",
    "pdd": "拼多多",
    "iqiyi": "爱奇艺",
    "youku": "优酷",
    "tencentvideo": "腾讯视频",
    "migu": "咪咕",
}


def extract_clean_venue(user_text: str) -> str:
    """Extract a clean physical venue name from '在<venue>办/开/做...' clauses without task/time tails."""
    loc_re = re.compile(
        r"(?:在|到)([\u4e00-\u9fff][\u4e00-\u9fffA-Za-z0-9]{2,20}?)"
        r"(?=这边|这儿|那边|那儿|这里|，|,|。|；|！|\s|办|开|用|做|搞|弄|准备|帮我|给|跟|要|有|了|拍|带队|打算|想|日常|连线|直播|会议|通话)"
    )
    pronouns = {"这儿", "这边", "那边", "那儿", "这里", "那里", "线上", "路上", "现场"}
    for m in loc_re.finditer(user_text or ""):
        cand = m.group(1).strip()
        if (
            cand in pronouns
            or re.search(r"\d|ms|以内|以下|时|分钟|小时|之前|上次|上周|常规|周期|上午|下午|晚上|中午", cand)
        ):
            continue
        return cand
    return ""


def align_scene_with_known(filestore: FileStore, ns: Namespace, scene: str, user_text: str) -> str:
    """Reuse an existing scene name of this user when it literally occurs in the user's text."""
    known: List[str] = []
    for node in filestore.iter_nodes(ns, memory_type="preference", subtype="scene_preference"):
        if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value, MemoryStatus.SUPERSEDED.value):
            continue
        sc = str(node.content.get("scene") or "").strip()
        if sc and sc not in known:
            known.append(sc)
    if scene and scene in known:
        return scene
    hits = [sc for sc in known if sc in (user_text or "")]
    return max(hits, key=len) if hits else scene


def _resolve_service_cn(biz: Any, user_text: str, existing_param: Optional[MemoryNode]) -> str:
    s_biz = str(biz)
    if s_biz == "openlive":
        return "开直播"
    if s_biz == "live":
        return "开直播" if "开直播" in user_text else "看直播"
    if s_biz in ("video", "short_video"):
        if "短视频" in user_text:
            return "短视频"
        if "视频" in user_text:
            return "视频"
        if existing_param is not None and existing_param.content.get("service_cn"):
            return str(existing_param.content.get("service_cn"))
        return "短视频"
    return {"meeting": "会议", "video_call": "视频通话", "game": "游戏"}.get(s_biz, s_biz)


def _derive_event_from(
    ev: Event,
    subtype: str,
    memory_key_fields: Dict[str, Any],
    content: Dict[str, Any],
    event_type: Optional[str] = None,
) -> Event:
    return Event(
        event_type=event_type or ev.event_type,
        subtype=subtype,
        memory_key_fields=memory_key_fields,
        content=content,
        validity=ev.validity,
        confidence=ev.confidence,
        source_session_id=ev.source_session_id,
        evidence_turns=list(ev.evidence_turns),
        is_concession=ev.is_concession,
        user_id=ev.user_id,
        tenant_id=ev.tenant_id,
        event_time=ev.event_time,
    )


def decompose_to_orthogonal_events(
    filestore: FileStore,
    index: MemoryIndex,
    apply_llm_event_fn: Callable[[Namespace, Event], MemoryNode],
    ns: Namespace,
    ev: Event,
    turns: List[Dict[str, Any]],
) -> List[Event]:
    """Decompose a preference event into orthogonal, non-coupled memory events (Scheme A)."""
    if ev.subtype not in (
        "application_business",
        "scene_preference",
        "periodic_preference",
        "preferred_value",
        "experience_parameter",
    ):
        return [ev]

    app = ev.content.get("application") or ev.memory_key_fields.get("application")
    biz = ev.content.get("business") or ev.memory_key_fields.get("business")
    if not app or not biz:
        return [ev]

    scene = str(ev.content.get("scene") or ev.memory_key_fields.get("scene") or "").strip()
    tw = str(ev.content.get("time_window") or ev.content.get("condition") or "").strip()
    decl_mode = ev.content.get("declaration_mode")

    user_text = " ".join(
        str(t.get("content") or t.get("text") or "")
        for t in turns
        if str(t.get("role", "user")).lower() == "user"
    )

    if str(biz) in ("live", "openlive"):
        if "开直播" in user_text and "看直播" not in user_text:
            biz = "openlive"
        elif "看直播" in user_text and "开直播" not in user_text:
            biz = "live"
        elif str(biz) == "live":
            has_openlive = any(
                n.content.get("application") == app and n.content.get("business") == "openlive"
                for n in filestore.iter_nodes(ns)
            )
            has_watchlive = any(
                n.content.get("application") == app and n.content.get("business") == "live"
                for n in filestore.iter_nodes(ns)
            )
            if has_openlive and not has_watchlive:
                biz = "openlive"
    ev.content["business"] = biz
    if "business" in ev.memory_key_fields:
        ev.memory_key_fields["business"] = biz

    scene = align_scene_with_known(filestore, ns, scene, user_text)
    if scene and any(marker in scene for marker in _TRANSIENT_SCENE_MARKERS):
        scene = ""
    if scene and re.search(r"从\d|分到\d|持续|小时|分钟|上要|处理|开展|进行|今天在", scene):
        scene = extract_clean_venue(user_text)
    elif not scene:
        scene = extract_clean_venue(user_text)
        if scene and any(marker in scene for marker in _TRANSIENT_SCENE_MARKERS):
            scene = ""

    existing_param: Optional[MemoryNode] = None
    existing_periodic: Optional[MemoryNode] = None
    for node in filestore.iter_nodes(ns):
        if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value, MemoryStatus.SUPERSEDED.value):
            continue
        if node.content.get("application") != app or node.content.get("business") != biz:
            continue
        if node.subtype in ("scene_preference", "preferred_value"):
            if scene and node.subtype == "scene_preference" and node.content.get("scene") == scene:
                existing_param = node
            elif existing_param is None or node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                if not (
                    existing_param
                    and existing_param.subtype == "scene_preference"
                    and existing_param.content.get("scene") == scene
                ):
                    existing_param = node
        elif node.subtype == "periodic_preference":
            if tw and node.content.get("time_window") == tw:
                existing_periodic = node
            elif existing_periodic is None or node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                existing_periodic = node

    app_cn = _CANONICAL_APP_CN.get(str(app).lower(), str(app))
    service_cn = _resolve_service_cn(biz, user_text, existing_param)
    qos_fields: Dict[str, Any] = {"app_cn": app_cn, "service_cn": service_cn}
    if (
        existing_param is not None
        and ev.validity != "temporary"
        and ev.event_type != EventType.PREFERENCE_CORRECT.value
    ):
        for k in ("resolution", "latency", "priority", "bitrate"):
            if existing_param.content.get(k) not in (None, ""):
                qos_fields[k] = existing_param.content[k]
    for k in ("resolution", "latency", "priority", "bitrate"):
        if ev.content.get(k) not in (None, ""):
            qos_fields[k] = ev.content[k]

    out_events: List[Event] = []

    if (
        existing_param is not None
        and existing_param.stats.votes == 1
        and scene
        and (existing_param.subtype == "preferred_value" or existing_param.content.get("scene") != scene)
        and ev.validity != "temporary"
    ):
        filestore.delete_node(ns, existing_param)
        index.delete(existing_param.memory_id)

        sc_content: Dict[str, Any] = {
            "scene": scene,
            "application": app,
            "business": biz,
            **qos_fields,
            "validity": ev.validity,
            "declaration_mode": "explicit",
            "note": f"场景参数覆写：{scene} 下使用 {app} ({biz})",
        }
        seed_ev = Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="scene_preference",
            memory_key_fields={"scene": scene, "application": app, "business": biz},
            content=sc_content,
            validity="long_term",
            confidence=existing_param.confidence,
            source_session_id=existing_param.evidence[0] if existing_param.evidence else "",
            user_id=ev.user_id,
            tenant_id=ev.tenant_id,
            event_time=existing_param.created_at,
        )
        apply_llm_event_fn(ns, seed_ev)

        for expr_node in list(filestore.iter_nodes(ns, memory_type="preference", subtype="expression")):
            if (
                expr_node.content.get("application") == app
                and expr_node.content.get("business") == biz
                and not expr_node.content.get("scene")
            ):
                expr_node.content["scene"] = scene
                filestore.save_node(ns, expr_node)
                index.upsert(ns, expr_node)

        out_events.append(
            _derive_event_from(
                ev,
                subtype="scene_preference",
                memory_key_fields={"scene": scene, "application": app, "business": biz},
                content=sc_content,
                event_type=EventType.PREFERENCE_CONFIRM.value,
            )
        )
    elif existing_param is not None and (not scene or existing_param.content.get("scene") == scene):
        target_st = existing_param.subtype
        if target_st == "scene_preference":
            sc_val = str(existing_param.content.get("scene") or scene)
            key_fields = {"scene": sc_val, "application": app, "business": biz}
            content_out: Dict[str, Any] = {
                "scene": sc_val,
                "application": app,
                "business": biz,
                **qos_fields,
                "validity": ev.validity,
                "note": f"场景参数覆写：{sc_val} 下使用 {app} ({biz})",
            }
        else:
            target_st = "preferred_value"
            key_fields = {"application": app, "business": biz}
            content_out = {
                "application": app,
                "business": biz,
                **qos_fields,
                "validity": ev.validity,
                "note": f"默认参数偏好：{app} ({biz})",
            }
        if decl_mode:
            content_out["declaration_mode"] = decl_mode
        out_events.append(_derive_event_from(ev, subtype=target_st, memory_key_fields=key_fields, content=content_out))
    elif scene:
        sc_content_new: Dict[str, Any] = {
            "scene": scene,
            "application": app,
            "business": biz,
            **qos_fields,
            "validity": ev.validity,
            "note": f"场景参数覆写：{scene} 下使用 {app} ({biz})",
        }
        if decl_mode:
            sc_content_new["declaration_mode"] = decl_mode
        out_events.append(
            _derive_event_from(
                ev,
                subtype="scene_preference",
                memory_key_fields={"scene": scene, "application": app, "business": biz},
                content=sc_content_new,
            )
        )
    else:
        pv_content_new: Dict[str, Any] = {
            "application": app,
            "business": biz,
            **qos_fields,
            "validity": ev.validity,
            "note": f"默认参数偏好：{app} ({biz})",
        }
        if decl_mode:
            pv_content_new["declaration_mode"] = decl_mode
        out_events.append(
            _derive_event_from(
                ev,
                subtype="preferred_value",
                memory_key_fields={"application": app, "business": biz},
                content=pv_content_new,
            )
        )

    cond_tw = tw
    if not cond_tw and existing_periodic is not None and ev.validity != "temporary":
        cond_tw = str(
            existing_periodic.content.get("time_window") or existing_periodic.content.get("condition") or ""
        ).strip()

    if cond_tw and ev.validity != "temporary":
        val_str = f"{app}:{biz}"
        pp_ev_type = (
            EventType.PREFERENCE_CONFIRM.value
            if (
                existing_periodic is not None
                and existing_periodic.content.get("time_window") == cond_tw
                and ev.event_type == EventType.PREFERENCE_SET.value
            )
            else ev.event_type
        )
        pp_content: Dict[str, Any] = {
            "condition": cond_tw,
            "value": val_str,
            "time_window": cond_tw,
            "application": app,
            "business": biz,
            "app_cn": app_cn,
            "service_cn": service_cn,
            "validity": ev.validity,
            "note": f"周期时间路由：{cond_tw} -> {app} ({biz})",
        }
        if decl_mode:
            pp_content["declaration_mode"] = decl_mode
        out_events.append(
            _derive_event_from(
                ev,
                subtype="periodic_preference",
                memory_key_fields={"condition": cond_tw, "value": val_str},
                content=pp_content,
                event_type=pp_ev_type,
            )
        )

    return out_events


def run_llm_extract_and_induce_with_events(
    extractor: Extractor,
    filestore: FileStore,
    induction: InductionEngine,
    decompose_fn: Callable[[Namespace, Event, List[Dict[str, Any]]], List[Event]],
    apply_llm_event_fn: Callable[[Namespace, Event], MemoryNode],
    payload: Dict[str, Any],
) -> Tuple[List[MemoryNode], List[Dict[str, Any]]]:
    """Run LLM extraction, orthogonal event decomposition, inline alias binding, and cross-session induction."""
    tenant_id = str(payload.get("tenant_id", "default"))
    user_id = str(payload.get("user_id", ""))
    session_id = str(payload.get("session_id", ""))
    turns = list(payload.get("turns") or [])
    session_ts = str(payload.get("session_ts") or normalize_ts_str(None))
    has_concession = bool(payload.get("has_concession", False))
    ns = user_namespace(user_id=user_id, tenant_id=tenant_id)

    fp_events = [Event.from_dict(d) for d in (payload.get("fastpath_events") or [])]
    llm_events = extractor.extract(
        session_turns=turns,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )

    pref_events: List[Event] = []
    alias_by_key: Dict[str, Event] = {}
    llm_pref_events = [
        ev
        for ev in llm_events
        if ev.event_type != EventType.EXPRESSION_ALIAS.value and ev.subtype != "expression"
    ]

    for ev in fp_events:
        if ev.event_type == EventType.EXPRESSION_ALIAS.value or ev.subtype == "expression":
            al = str(ev.memory_key_fields.get("alias") or ev.content.get("alias") or "").strip()
            if al:
                alias_by_key[al] = ev
        elif not llm_pref_events:
            pref_events.append(ev)

    for ev in llm_events:
        if ev.event_type == EventType.EXPRESSION_ALIAS.value or ev.subtype == "expression":
            al = str(ev.memory_key_fields.get("alias") or ev.content.get("alias") or "").strip()
            if al and al not in alias_by_key:
                alias_by_key[al] = ev
        else:
            pref_events.append(ev)

    applied_nodes: List[MemoryNode] = []
    applied_events: List[Dict[str, Any]] = []
    last_param_node: Optional[MemoryNode] = None

    for ev in pref_events:
        ev.event_time = session_ts
        if has_concession:
            ev.is_concession = True
        extractor._backfill_time_window(ev, turns)
        extractor._backfill_qos_params(ev, turns)
        extractor._normalize_validity(ev, turns)
        ev.content["validity"] = ev.validity

        for oev in decompose_fn(ns, ev, turns):
            node = apply_llm_event_fn(ns, oev)
            applied_nodes.append(node)
            applied_events.append(oev.to_dict())
            if node.subtype in ("preferred_value", "scene_preference"):
                last_param_node = node

    if last_param_node is not None and last_param_node.validity.type != "temporary":
        user_text_combined = " ".join(
            str(t.get("content") or t.get("text") or "")
            for t in turns
            if str(t.get("role", "user")).lower() == "user"
        )
        for m_inline in re.finditer(r"(画质|时延)(?:标准)?按[‘'“\"]([^’'”\"]+)[’'”\"]", user_text_combined):
            slot_kind, alias_str = m_inline.group(1), m_inline.group(2).strip()
            if not alias_str or alias_str in alias_by_key:
                continue
            target_val = (
                last_param_node.content.get("resolution")
                if slot_kind == "画质"
                else last_param_node.content.get("latency")
            )
            if target_val:
                alias_by_key[alias_str] = Event(
                    event_type=EventType.EXPRESSION_ALIAS.value,
                    subtype="expression",
                    memory_key_fields={"alias": alias_str},
                    content={
                        "alias": alias_str,
                        "source_expression": alias_str,
                        "target": str(target_val),
                        "validity": "long_term",
                    },
                    validity="long_term",
                    confidence=0.92,
                    source_session_id=session_id,
                    evidence_turns=[1],
                    user_id=user_id,
                    tenant_id=tenant_id,
                    event_time=session_ts,
                )

    for alias_str, alias_ev in alias_by_key.items():
        alias_ev.event_time = session_ts
        alias_ev.validity = "long_term"
        alias_ev.content["validity"] = "long_term"

        scope_node = last_param_node
        if scope_node is None:
            for cand in filestore.iter_nodes(ns):
                if cand.subtype in ("preferred_value", "scene_preference") and cand.status in (
                    MemoryStatus.ACTIVE.value,
                    MemoryStatus.REINFORCED.value,
                ):
                    scope_node = cand
                    break

        if scope_node is not None:
            for slot_k in ("application", "business", "scene", "app_cn", "service_cn"):
                if scope_node.content.get(slot_k) not in (None, ""):
                    alias_ev.content[slot_k] = scope_node.content[slot_k]
        for drop_k in ("parent_subtype", "parent_key", "time_window"):
            alias_ev.content.pop(drop_k, None)
        alias_ev.content["note"] = f"表达映射：「{alias_str}」-> {alias_ev.content.get('target')}"

        a_node = apply_llm_event_fn(ns, alias_ev)
        applied_nodes.append(a_node)
        applied_events.append(alias_ev.to_dict())

    induction.run_all(ns)
    return applied_nodes, applied_events
