"""Pure phrase synthesis, FTS5 tokenization, and temporal window matching helpers for storage indexing."""
from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence

from ..core.models import MemoryNode

_NOISY_CONTENT_KEYS = {
    "validity",
    "declaration_mode",
    "category",
    "confidence",
    "owner_type",
    "user_id",
    "tenant_id",
    "source_session_id",
}

_CN_TO_CANONICAL_APP: Dict[str, str] = {
    "抖音": "douyin",
    "快手": "kuaishou",
    "哔哩哔哩": "bilibili",
    "微信": "wechat",
    "腾讯会议": "tencent_meeting",
    "钉钉": "dingtalk",
    "飞书": "feishu",
    "小红书": "xiaohongshu",
    "王者荣耀": "wangzherongyao",
    "高德地图": "gaode",
    "虎牙": "huya",
    "斗鱼直播": "douyu",
    "和平精英": "hepingjingying",
    "QQ": "qq",
    "淘宝": "taobao",
    "拼多多": "pdd",
    "爱奇艺": "iqiyi",
    "优酷": "youku",
    "腾讯视频": "tencentvideo",
    "咪咕": "migu",
}

_CN_TO_CANONICAL_BIZ: Dict[str, str] = {
    "开直播": "openlive",
    "看直播": "watch_live",
    "短视频": "video",
    "视频": "video",
    "视频通话": "video_call",
    "会议": "meeting",
    "游戏": "game",
    "云游戏": "game",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}")


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two dense float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    try:
        import numpy as np  # type: ignore

        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except ImportError:
        dot = sum(x * y for x, y in zip(vec_a, vec_b))
        na = math.sqrt(sum(x * x for x in vec_a))
        nb = math.sqrt(sum(y * y for y in vec_b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)


def build_pure_phrase_text(node: MemoryNode) -> str:
    """Extract pure 4~12 character semantic trigger phrase (phrase_text) for high-discriminability embedding.

    Args:
        node: Authoritative memory node to extract trigger phrase from.

    Returns:
        Short natural-language phrase suitable for dense embedding indexing.
    """
    c = node.content or {}
    st = node.subtype

    if st == "expression":
        return str(c.get("phrase_text") or c.get("alias") or c.get("source_expression") or node.memory_key or "").strip()
    if st in ("scene_preference", "environment"):
        return str(c.get("phrase_text") or c.get("scene") or c.get("environment") or "").strip()
    if st == "periodic_preference":
        return str(
            c.get("phrase_text") or c.get("window_label") or c.get("time_window") or c.get("condition") or ""
        ).strip()
    if st == "service_to_app":
        return str(c.get("phrase_text") or c.get("service_cn") or c.get("business") or "").strip()
    if st in ("preferred_value", "application_business", "experience_parameter"):
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        app_cn = str(c.get("app_cn") or c.get("application") or "").strip()
        svc_cn = str(c.get("service_cn") or c.get("business") or "").strip()
        return f"{app_cn}{svc_cn}".strip()
    if st == "working_record":
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        loc = str(c.get("location") or c.get("scene") or "").strip()
        svc = str(c.get("service_name") or c.get("business") or "").strip()
        return f"{loc} {svc}".strip() if loc else svc
    if st == "experience":
        if c.get("phrase_text"):
            return str(c["phrase_text"]).strip()
        trig = c.get("trigger_conditions")
        if isinstance(trig, dict):
            kws: List[str] = []
            for tk in ("keywords", "task_keywords", "location_keywords"):
                sub = trig.get(tk)
                if isinstance(sub, list):
                    for item in sub:
                        s_item = str(item).strip()
                        if s_item and s_item not in kws:
                            kws.append(s_item)
            if kws:
                return " ".join(kws[:4])
        return str(c.get("title") or c.get("condition") or node.description or "").strip()
    return str(c.get("phrase_text") or node.description or node.memory_key or "").strip()


def build_search_text(node: MemoryNode) -> str:
    """Build semantic text for FTS5 from node fields and content (zero hardcoded synonym lists).

    Args:
        node: MemoryNode instance.

    Returns:
        Combined searchable text string.
    """
    c = node.content or {}
    st = node.subtype
    pure_phrase = build_pure_phrase_text(node)

    semantic_values: List[str] = []
    if pure_phrase:
        semantic_values.append(pure_phrase)

    if st == "expression":
        alias = str(c.get("alias") or c.get("source_expression") or node.memory_key or "").strip()
        target = str(c.get("target") or "").strip()
        note = str(c.get("note") or "").strip()
        for v in (alias, target, note):
            if v and v not in semantic_values:
                semantic_values.append(v)
    elif st in ("scene_preference", "environment"):
        sc = str(c.get("scene") or c.get("environment") or "").strip()
        app = str(c.get("application") or "").strip()
        biz = str(c.get("business") or "").strip()
        for v in (sc, app, biz):
            if v and v not in semantic_values:
                semantic_values.append(v)
        for k in ("resolution", "max_delay", "latency", "bit_rate", "priority", "note"):
            if c.get(k):
                semantic_values.append(str(c[k]))
    elif st == "working_record":
        loc = str(c.get("location") or c.get("scene") or "").strip()
        app_n = str(c.get("application_name") or c.get("application") or "").strip()
        svc_n = str(c.get("service_name") or c.get("business") or "").strip()
        for v in (loc, app_n, svc_n):
            if v and v not in semantic_values:
                semantic_values.append(v)
        for k in (
            "start_timestamp",
            "end_timestamp",
            "duration",
            "resolution",
            "rtt_max",
            "latency",
            "summary",
            "note",
        ):
            if c.get(k):
                semantic_values.append(str(c[k]))
        t_aliases = c.get("temporal_reference_aliases")
        if isinstance(t_aliases, dict):
            for av in t_aliases.values():
                if av:
                    semantic_values.append(str(av))
    elif st == "experience":
        for k in (
            "title",
            "condition",
            "action",
            "description",
            "content",
            "strategy_type",
            "domain",
            "rule_type",
            "causal_rationale",
        ):
            if c.get(k):
                semantic_values.append(str(c[k]))
        trig = c.get("trigger_conditions")
        if isinstance(trig, dict):
            for tk in ("keywords", "location_keywords", "task_keywords"):
                kws = trig.get(tk)
                if isinstance(kws, list):
                    semantic_values.extend(str(x) for x in kws if x)
    else:
        for k, v in c.items():
            if k in _NOISY_CONTENT_KEYS or v is None:
                continue
            if isinstance(v, (str, int, float)):
                semantic_values.append(str(v))
            elif isinstance(v, (list, dict)):
                semantic_values.append(json.dumps(v, ensure_ascii=False))

    parts: List[str] = [
        node.memory_key,
        node.description,
        node.overview,
        node.details,
        " ".join(semantic_values),
    ]
    return " ".join(p for p in parts if p)


def tokenize_for_fts(text: str) -> str:
    """Segment Chinese characters and alphanumeric tokens with spaces so SQLite unicode61 FTS5 matches."""
    if not text:
        return ""
    tokens: List[str] = []
    for m in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
        chunk = m.group(0)
        if re.match(r"^[A-Za-z0-9_]+$", chunk):
            tokens.append(chunk.lower())
        else:
            tokens.extend(list(chunk))
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return " ".join(tokens)


def _time_to_minutes(hhmm: str) -> Optional[int]:
    m = re.match(r"^(\d{1,2}):(\d{2})$", hhmm.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def time_matches_window(queried_time: str, stored_window: str) -> bool:
    """Check if queried_time ('12:30-13:30' or single time point '12:45') matches stored_window ('12:30-13:30')."""
    if not queried_time or not stored_window:
        return False
    qt = queried_time.strip()
    sw = stored_window.strip()
    if qt == sw:
        return True
    if "-" not in sw:
        return False

    s_parts = sw.split("-", 1)
    s_start = _time_to_minutes(s_parts[0])
    s_end = _time_to_minutes(s_parts[1])
    if s_start is None or s_end is None:
        return False

    if "-" in qt:
        q_parts = qt.split("-", 1)
        q_start = _time_to_minutes(q_parts[0])
        q_end = _time_to_minutes(q_parts[1])
        if q_start is None or q_end is None:
            return False
        if s_end >= s_start and q_end >= q_start:
            return max(s_start, q_start) <= min(s_end, q_end)
        return False

    q_pt = _time_to_minutes(qt)
    if q_pt is None:
        return False
    lead_start = (s_start - 30) % (24 * 60)
    if s_end >= s_start:
        if s_start >= 30:
            return (s_start - 30) <= q_pt <= s_end
        return q_pt >= lead_start or q_pt <= s_end
    return q_pt >= lead_start or q_pt <= s_end


def _extract_ymd(ts_str: str) -> str:
    """Extract YYYY-MM-DD from Chinese or ISO date string."""
    if not ts_str:
        return ""
    m_cn = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", ts_str)
    if m_cn:
        return f"{int(m_cn.group(1)):04d}-{int(m_cn.group(2)):02d}-{int(m_cn.group(3)):02d}"
    m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts_str)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
    return ""
