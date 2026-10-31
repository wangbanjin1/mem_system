"""Query preprocessing, negation clause separation, and overlapping character sliding-window generation."""
from __future__ import annotations

import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_BIZ_PROTOTYPE_TEXTS: Dict[str, str] = {
    "openlive": "开直播 主播开播 直播推流 上麦发言 出镜带货 跟粉丝连线互动 现场直播",
    "live": "看直播 观看直播 刷直播间 蹲直播 看主播",
    "meeting": "视频会议 线上开会 远程碰头 投屏演示 对齐周报 团队连线讨论 汇报",
    "game": "打游戏 玩手游 排位赛 上号开黑 竞技团战",
    "video": "刷短视频 看视频 刷抖音快手短视频",
    "video_call": "视频通话 打视频聊天 语音连线通话",
}

_BIZ_KEYWORDS: List[Tuple[str, str]] = [
    ("视频通话", "video_call"),
    ("视频聊天", "video_call"),
    ("语音通话", "voice_call"),
    ("上麦开播", "openlive"),
    ("开播推流", "openlive"),
    ("直播推流", "openlive"),
    ("出镜开播", "openlive"),
    ("连麦开播", "openlive"),
    ("上麦发言", "openlive"),
    ("开直播", "openlive"),
    ("开播", "openlive"),
    ("推流", "openlive"),
    ("上麦", "openlive"),
    ("看直播", "live"),
    ("刷直播", "live"),
    ("蹲直播间", "live"),
    ("看主播", "live"),
    ("直播", "live"),
    ("短视频", "video"),
    ("刷短视频", "video"),
    ("刷视频", "video"),
    ("长视频", "long_video"),
    ("追剧", "long_video"),
    ("看剧", "long_video"),
    ("视频", "video"),
    ("腾讯会议", "meeting"),
    ("视频会议", "meeting"),
    ("线上投屏", "meeting"),
    ("远程碰头", "meeting"),
    ("视频对齐", "meeting"),
    ("线上碰头", "meeting"),
    ("投屏开会", "meeting"),
    ("开个会", "meeting"),
    ("开会", "meeting"),
    ("会议", "meeting"),
    ("打把排位", "game"),
    ("排位赛", "game"),
    ("上号开黑", "game"),
    ("组队开黑", "game"),
    ("打游戏", "game"),
    ("玩游戏", "game"),
    ("开黑", "game"),
    ("排位", "game"),
    ("网游", "game"),
    ("手游", "game"),
    ("游戏", "game"),
    ("大文件下载", "download"),
    ("下载", "download"),
    ("上传", "upload"),
]

_CLAUSE_SPLIT_RE = re.compile(
    r"[，,。；;！!？?\n]+|(?=(?:但|不过|而是|改用|换成|改在|改到|改成|转为|替成))"
)
_SLOT_NEGATION_RE = re.compile(
    r"(?:不用|别用|不按|别按|不再用|不要用|不开|别开|不看|别看|不玩|别玩|不打|别打|不刷|别刷|取消|放弃|停止|不是用|也没用|没用)"
)
_CJK_CHARS_RE = re.compile(r"[\u4e00-\u9fff]+")

_QUERY_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\b")
_QUERY_CN_TIME_RE = re.compile(r"(\d{1,2})[时点](?:(\d{1,2})分?)?(?:到|至|-)(\d{1,2})[时点](?:(\d{1,2})分?)?")
_QUERY_CN_PT_RE = re.compile(r"(?<!\d)(\d{1,2})[时点](?:(\d{1,2})分|半)?")


def _build_sliding_windows(clauses: Sequence[str], win_size: int = 8, step: int = 4) -> List[str]:
    """Slice positive clauses into overlapping 6~10 character sliding windows (default win_size=8, step=4)
    so short stored phrase_text vectors (4~10 chars) achieve high cosine similarity against long multi-clause queries.
    """
    windows: List[str] = []
    for raw_c in clauses:
        c = re.sub(r"\s+", "", str(raw_c or "")).strip()
        if not c:
            continue
        if c not in windows:
            windows.append(c)
        for ws in (win_size, 10) if win_size != 10 else (win_size,):
            if len(c) <= ws:
                continue
            for i in range(0, len(c) - ws + 1, step):
                w = c[i : i + ws]
                if w and w not in windows:
                    windows.append(w)
            tail = c[-ws:]
            if tail and tail not in windows:
                windows.append(tail)
    return windows


def _split_positive_and_negated_clauses(query: str) -> Tuple[List[str], List[str]]:
    """Split a natural language query into positive clauses and negated clauses."""
    q = (query or "").strip()
    if not q:
        return [], []
    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(q) if c and c.strip()]
    pos_clauses: List[str] = []
    neg_clauses: List[str] = []
    for cl in raw_clauses:
        if _SLOT_NEGATION_RE.search(cl):
            neg_clauses.append(cl)
        else:
            pos_clauses.append(cl)
    if not pos_clauses and not neg_clauses:
        pos_clauses = [q]
    return pos_clauses, neg_clauses


def _extract_positive_clauses(query: str) -> List[str]:
    """Return only the non-negated positive clauses from a natural language query."""
    pos_clauses, _ = _split_positive_and_negated_clauses(query)
    return pos_clauses


def _parse_relative_date_window(query: str, now_ts: Optional[str]) -> Optional[Tuple[str, str]]:
    """Compute [start_ymd, end_ymd] window from colloquial temporal expressions ('昨天', '上周', '上个月初', '之前的意图保障单')."""
    if not query or not now_ts:
        return None
    s_ts = str(now_ts)
    ref_d: Optional[date] = None
    m_cn = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", s_ts)
    if m_cn:
        try:
            ref_d = date(int(m_cn.group(1)), int(m_cn.group(2)), int(m_cn.group(3)))
        except ValueError:
            ref_d = None
    else:
        m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s_ts)
        if m_iso:
            try:
                ref_d = date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
            except ValueError:
                ref_d = None
    if ref_d is None:
        return None

    if "上个月" in query or "上月" in query:
        return ((ref_d - timedelta(days=50)).isoformat(), (ref_d - timedelta(days=10)).isoformat())
    if "上周" in query or "上一周" in query:
        return ((ref_d - timedelta(days=25)).isoformat(), (ref_d - timedelta(days=2)).isoformat())
    if "昨天预约" in query or "前天" in query or "昨天" in query:
        return ((ref_d - timedelta(days=45)).isoformat(), ref_d.isoformat())
    if "意图保障单" in query and ("之前" in query or "上次" in query):
        return ((ref_d - timedelta(days=60)).isoformat(), ref_d.isoformat())
    return None


def _fuzzy_phrase_match(phrase: str, text: str) -> bool:
    """Match an alias or scene phrase against query text without any hardcoded synonym dictionary."""
    p = (phrase or "").strip()
    t = (text or "").strip()
    if not p or not t:
        return False
    if p in t:
        return True

    cjk_p = "".join(_CJK_CHARS_RE.findall(p))
    if len(cjk_p) < 4:
        return False

    chunks: List[str] = []
    for i in range(0, len(cjk_p) - 1, 2):
        ck = cjk_p[i : i + 2]
        if ck not in chunks:
            chunks.append(ck)
    tail_ck = cjk_p[-2:]
    if tail_ck not in chunks:
        chunks.append(tail_ck)

    max_span = max(14, len(cjk_p) * 3)
    matched_positions: List[int] = []
    for ck in chunks:
        pos = t.find(ck)
        if pos >= 0:
            matched_positions.append(pos)

    if len(matched_positions) >= 2:
        required = len(chunks) if len(chunks) == 2 else max(2, math.ceil(len(chunks) * 0.66))
        if len(matched_positions) >= required and max(matched_positions) - min(matched_positions) <= max_span:
            return True

    all_bigrams = [cjk_p[i : i + 2] for i in range(len(cjk_p) - 1)]
    if not any(bg in t for bg in all_bigrams):
        return False

    char_positions: List[int] = []
    search_start = 0
    for ch in cjk_p:
        pos = t.find(ch, search_start)
        if pos < 0:
            pos = t.find(ch)
        if pos >= 0:
            char_positions.append(pos)
            search_start = pos + 1

    min_chars = max(3, math.ceil(len(cjk_p) * 0.66))
    if len(char_positions) >= min_chars and max(char_positions) - min(char_positions) <= max_span:
        return True
    return False


def _load_default_app_map() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load alias -> primary application name and equivalence variants from data/app_map.json."""
    mapping: Dict[str, str] = {}
    variants: Dict[str, List[str]] = {}
    try:
        repo_root = Path(__file__).resolve().parents[3]
        p = repo_root / "data" / "app_map.json"
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            for canonical, info in raw.items():
                if canonical.startswith("_"):
                    continue
                alias_list: List[str] = []
                if isinstance(info, dict):
                    alias_list = [str(a) for a in info.get("aliases", []) if a]
                elif isinstance(info, list):
                    alias_list = [str(a) for a in info if a]
                primary = alias_list[0] if alias_list else canonical
                equiv = list(dict.fromkeys([primary, canonical]))
                variants[primary] = equiv
                variants[canonical] = equiv
                mapping[canonical.lower()] = primary
                mapping[primary.lower()] = primary
                for al in alias_list:
                    mapping[al.lower()] = primary
    except (OSError, ValueError, TypeError):
        return mapping, variants
    return mapping, variants
