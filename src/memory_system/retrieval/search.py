from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.models import MemoryNode, MemoryStatus, now_utc_str
from ..core.namespace import Namespace
from ..core.registry import MemoryRegistry
from ..pipeline.updater import (
    compute_decayed_heat,
    compute_ebbinghaus_retention,
    normalize_ts_str,
    parse_ts_epoch,
)
from ..providers.embedding import EmbeddingProvider
from ..storage.filestore import FileStore
from ..storage.graph import GraphStore
from ..storage.index import MemoryIndex, _cosine_similarity, build_search_text, time_matches_window


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
_PARAPHRASE_SYNONYMS: List[Tuple[str, str]] = [
    ("大巴车", "大巴"),
    ("通勤大巴", "大巴"),
    ("转场班车", "大巴转场"),
    ("班车", "大巴"),
    ("路上", "途中"),
    ("车上", "途中"),
    ("半路", "途中"),
    ("中午", "午间"),
    ("正午", "午间"),
    ("早上", "早间"),
    ("清晨", "早间"),
    ("晚上", "晚间"),
    ("夜间", "晚间"),
    ("总结大会", "阶段工作总结会"),
    ("阶段总结会", "阶段工作总结会"),
    ("复盘总结会", "阶段工作总结会"),
    ("驻地酒店", "外勤驻地"),
    ("外勤酒店", "外勤驻地"),
]
_CJK_CHARS_RE = re.compile(r"[\u4e00-\u9fff]+")


def _normalize_paraphrase_text(text: str) -> str:
    """Normalize common colloquial temporal/scene synonyms into canonical form."""
    out = text
    for src, dst in _PARAPHRASE_SYNONYMS:
        if src in out and dst not in out:
            out = out.replace(src, dst)
    return out


def _fuzzy_phrase_match(phrase: str, text: str) -> bool:
    """Match an alias or scene phrase against query text via:
    1. Exact substring match
    2. Synonym-normalized substring match (e.g. '中午' -> '午间', '大巴车...路上' -> '大巴...途中')
    3. Split-token proximity match (e.g. '午间那个例行档期' -> '午间例行', '画质保个基本档位' -> '保个基本画质')
    """
    p = (phrase or "").strip()
    t = (text or "").strip()
    if not p or not t:
        return False
    if p in t:
        return True

    norm_p = _normalize_paraphrase_text(p)
    norm_t = _normalize_paraphrase_text(t)
    if norm_p in norm_t:
        return True

    cjk_p = "".join(_CJK_CHARS_RE.findall(norm_p))
    if len(cjk_p) < 4:
        return False

    # Build 2-character core CJK chunks
    chunks: List[str] = []
    for i in range(0, len(cjk_p) - 1, 2):
        ck = cjk_p[i : i + 2]
        if ck not in chunks:
            chunks.append(ck)
    tail_ck = cjk_p[-2:]
    if tail_ck not in chunks:
        chunks.append(tail_ck)

    if len(chunks) < 2:
        return False

    matched_positions: List[int] = []
    for ck in chunks:
        pos = norm_t.find(ck)
        if pos >= 0:
            matched_positions.append(pos)

    if len(matched_positions) < 2:
        return False
    required = len(chunks) if len(chunks) == 2 else max(2, math.ceil(len(chunks) * 0.66))
    if len(matched_positions) < required:
        return False

    span = max(matched_positions) - min(matched_positions)
    return span <= max(14, len(cjk_p) * 3)


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

_QUERY_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)\b")
_QUERY_CN_TIME_RE = re.compile(r"(\d{1,2})[时点](?:(\d{1,2})分?)?(?:到|至|-)(\d{1,2})[时点](?:(\d{1,2})分?)?")
_QUERY_CN_PT_RE = re.compile(r"(?<!\d)(\d{1,2})[时点](?:(\d{1,2})分|半)?")


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
    except Exception:
        pass
    return mapping, variants


class SearchEngine:
    """Hybrid retrieval engine with:
    - Scoped vector + FTS5 recall
    - Ebbinghaus per-subtype decay_lambda freshness
    - Half-life effective_heat activation & hit callback (+0.05)
    - Temporary node 0.5x score penalty
    - L0 / L0+L1 / full tiered loading
    - Explainability dictionary per hit
    """

    def __init__(
        self,
        registry: MemoryRegistry,
        filestore: FileStore,
        index: MemoryIndex,
        embedding_provider: Optional[EmbeddingProvider] = None,
        graph: Optional[GraphStore] = None,
        heat_callback: Optional[Callable[[Namespace, MemoryNode, float], Any]] = None,
        alpha: float = 0.5,
        beta: float = 0.2,
        gamma: float = 0.1,
        delta: float = 0.2,
        temporary_weight: float = 0.5,
        heat_half_life_hours: float = 168.0,
        heat_bump_on_hit: float = 0.05,
    ) -> None:
        self.registry = registry
        self.filestore = filestore
        self.index = index
        self.embedding_provider = embedding_provider
        self.graph = graph
        self.heat_callback = heat_callback
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.temporary_weight = float(temporary_weight)
        self.heat_half_life_hours = float(heat_half_life_hours)
        self.heat_bump_on_hit = float(heat_bump_on_hit)
        self.app_alias_map, self.app_variants_map = _load_default_app_map()
        self._biz_prototype_vecs: Optional[Dict[str, List[float]]] = None

    def _get_biz_prototype_vecs(self) -> Dict[str, List[float]]:
        if self._biz_prototype_vecs is not None:
            return self._biz_prototype_vecs
        if self.embedding_provider is None:
            return {}
        try:
            keys = list(_BIZ_PROTOTYPE_TEXTS.keys())
            texts = [_BIZ_PROTOTYPE_TEXTS[k] for k in keys]
            embs = self.embedding_provider.embed(texts)
            self._biz_prototype_vecs = {k: emb for k, emb in zip(keys, embs)}
        except Exception:
            self._biz_prototype_vecs = {}
        return self._biz_prototype_vecs

    def _expand_apps(self, app_list: Sequence[str]) -> List[str]:
        out: List[str] = []
        for a in app_list:
            for v in self.app_variants_map.get(a, [a]):
                if v and v not in out:
                    out.append(v)
        return out

    def _compute_effective_heat(self, node: MemoryNode, now_ts: str) -> float:
        """effective_heat = heat * 0.5^(delta_hours / half_life) (§2.3, search.py:167-182)."""
        return compute_decayed_heat(
            node.stats.heat,
            node.stats.last_heat_at,
            now_ts=now_ts,
            half_life_hours=self.heat_half_life_hours,
        )

    def _get_decay_lambda(self, subtype: str) -> float:
        """Read Ebbinghaus decay_lambda from registry (§2.4, search.py:184-190)."""
        spec = self.registry.get(subtype)
        if spec is not None:
            return float(spec.decay_lambda)
        return 0.02

    def _compute_freshness_and_stability(self, node: MemoryNode, now_ts: str) -> Tuple[float, float, float]:
        """Compute Ebbinghaus retention R(t, n), stability_factor S(n), and effective_lambda (§2.4)."""
        lam = self._get_decay_lambda(node.subtype)
        return compute_ebbinghaus_retention(
            decay_lambda=lam,
            last_reinforced_at=node.stats.last_reinforced_at or node.updated_at or node.created_at,
            now_ts=now_ts,
            positive_votes=node.stats.positive_votes,
            reuse_count=node.stats.reuse_count,
        )

    def _compute_freshness(self, node: MemoryNode, now_ts: str) -> float:
        """Backward-compatible wrapper returning retention R(t, n)."""
        retention, _, _ = self._compute_freshness_and_stability(node, now_ts)
        return retention

    def _get_active_slot_rows(self, ns: Namespace) -> List[Dict[str, Any]]:
        """Read active structured slot rows from SQLite index (zero disk I/O), with filestore fallback."""
        try:
            rows = self.index.list_active_slot_rows(ns)
            if rows:
                return rows
        except Exception:
            pass

        fallback_rows: List[Dict[str, Any]] = []
        for node in self.filestore.iter_nodes(ns):
            if node.status in (
                MemoryStatus.ARCHIVED.value,
                MemoryStatus.EXPIRED.value,
                MemoryStatus.SUPERSEDED.value,
            ):
                continue
            c = node.content or {}
            fallback_rows.append(
                {
                    "memory_id": node.memory_id,
                    "memory_type": node.memory_type,
                    "subtype": node.subtype,
                    "memory_key": node.memory_key,
                    "status": node.status,
                    "app_col": str(c.get("application") or "").strip(),
                    "biz_col": str(c.get("business") or "").strip(),
                    "scene_col": str(c.get("scene") or c.get("environment") or "").strip(),
                    "time_window_col": str(c.get("time_window") or c.get("condition") or "").strip(),
                    "alias_col": str(c.get("alias") or c.get("source_expression") or "").strip(),
                    "target_col": str(c.get("target") or "").strip(),
                }
            )
        return fallback_rows

    def _mine_slots_and_resolve_two_hop(
        self,
        ns: Namespace,
        query: str,
        now_ts: Optional[str] = None,
        query_slots: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Two-Hop Alias Resolution & Structured Slot Mining with Clause-Level Anti-Negation & Fuzzy Match (§2.5):
        - Step 0: Split query into positive vs negated clauses to prevent negated habits/apps from polluting recall.
        - Hop 1: Match user's active `expression` dictionary rows (exact + split-token/paraphrase match) over positive clauses.
        - Slot Mining: Extract structured (applications, businesses, scenes, time_windows) from positive clauses + expanded text + optional upstream `query_slots`.
        - Hop 2: Resolve lightweight routing pointer rows (`periodic_preference` and `service_to_app`) with scope and negation guards.
        """
        q_raw = (query or "").strip()
        if not q_raw and not query_slots:
            return {
                "resolved_aliases": {},
                "applications": [],
                "businesses": [],
                "scenes": [],
                "time_windows": [],
                "negated_applications": [],
                "negated_businesses": [],
                "negated_scenes": [],
                "negated_aliases": [],
                "negated_time_windows": [],
                "pointer_hops": [],
                "expanded_query": q_raw,
                "alias_hit_ids": {},
            }

        pos_clauses, neg_clauses = _split_positive_and_negated_clauses(q_raw)
        pos_text = " ， ".join(pos_clauses) if pos_clauses else ""
        neg_text = " ， ".join(neg_clauses) if neg_clauses else ""
        pos_low = pos_text.lower()
        neg_low = neg_text.lower()

        resolved_aliases: Dict[str, str] = {}
        alias_hit_ids: Dict[str, str] = {}
        apps: List[str] = []
        bizs: List[str] = []
        scenes: List[str] = []
        time_windows: List[str] = []
        negated_apps: List[str] = []
        negated_bizs: List[str] = []
        negated_scenes: List[str] = []
        negated_aliases: List[str] = []
        negated_time_windows: List[str] = []
        pointer_hops: List[str] = []

        # Merge optional upstream NLU / Router pre-extracted slots
        if isinstance(query_slots, dict):
            for k, dst in (
                ("applications", apps),
                ("businesses", bizs),
                ("scenes", scenes),
                ("time_windows", time_windows),
                ("negated_applications", negated_apps),
                ("negated_businesses", negated_bizs),
                ("negated_scenes", negated_scenes),
            ):
                for val in query_slots.get(k) or []:
                    s_val = str(val).strip()
                    if s_val and s_val not in dst:
                        dst.append(s_val)

        # 0. Mine negated apps & businesses from negated clauses
        if neg_low:
            for alias_low, canonical_app in sorted(self.app_alias_map.items(), key=lambda x: len(x[0]), reverse=True):
                if alias_low and alias_low in neg_low and canonical_app not in negated_apps:
                    negated_apps.append(canonical_app)
            for kw, canonical_biz in _BIZ_KEYWORDS:
                if kw in neg_low and canonical_biz not in negated_bizs:
                    negated_bizs.append(canonical_biz)

        # Pre-mine explicit positive apps and businesses from positive clauses for scope conflict checking
        explicit_apps: List[str] = list(apps)
        for alias_low, canonical_app in sorted(self.app_alias_map.items(), key=lambda x: len(x[0]), reverse=True):
            if alias_low and alias_low in pos_low and canonical_app not in explicit_apps:
                explicit_apps.append(canonical_app)
        explicit_apps = self._expand_apps(explicit_apps)
        negated_apps = self._expand_apps(negated_apps)

        explicit_bizs: List[str] = list(bizs)
        for kw, canonical_biz in _BIZ_KEYWORDS:
            if kw in pos_low and canonical_biz not in explicit_bizs:
                explicit_bizs.append(canonical_biz)

        # Fetch active slot rows from SQLite (zero disk I/O)
        slot_rows = self._get_active_slot_rows(ns)

        # --- Hop 1: Scan user's expression dictionary (with negation guard + fuzzy match + parent scope guard) ---
        for row in slot_rows:
            if row.get("subtype") != "expression":
                continue
            alias = str(row.get("alias_col") or row.get("memory_key") or "").strip()
            target = str(row.get("target_col") or "").strip()
            if not alias or not target:
                continue

            bound_app = str(row.get("app_col") or "").strip()
            bound_biz = str(row.get("biz_col") or "").strip()
            bound_scene = str(row.get("scene_col") or "").strip()

            # Check if alias is negated in negated_clauses
            if neg_text and _fuzzy_phrase_match(alias, neg_text) and not _fuzzy_phrase_match(alias, pos_text):
                if alias not in negated_aliases:
                    negated_aliases.append(alias)
                for m_tw in _QUERY_TIME_RE.finditer(target):
                    tw_neg = m_tw.group(1)
                    if tw_neg not in negated_time_windows:
                        negated_time_windows.append(tw_neg)
                if bound_app and bound_app not in explicit_apps and bound_app not in negated_apps:
                    negated_apps.append(bound_app)
                if bound_biz and bound_biz not in explicit_bizs and bound_biz not in negated_bizs:
                    negated_bizs.append(bound_biz)
                continue

            if _fuzzy_phrase_match(alias, pos_text):
                # Scope Conflict Guard: if positive query explicitly specifies a different app or business,
                # do not activate an alias bound to another storyline!
                if explicit_apps and bound_app and bound_app not in explicit_apps:
                    continue
                if explicit_bizs and bound_biz and bound_biz not in explicit_bizs:
                    continue

                resolved_aliases[alias] = target
                scope_str = f" (作用域: {bound_app}|{bound_biz})" if bound_app or bound_biz else ""
                match_mode = "精确命中" if alias in pos_text else "近似语义命中"
                alias_hit_ids[row["memory_id"]] = f"{match_mode}个性化词典: {alias} -> {target}{scope_str}"
                if bound_app and bound_app not in apps:
                    apps.append(bound_app)
                if bound_biz and bound_biz not in bizs:
                    bizs.append(bound_biz)
                if bound_scene and bound_scene not in scenes:
                    scenes.append(bound_scene)

        combined_pos_text = (pos_text + " " + " ".join(resolved_aliases.values())).strip()
        combined_pos_low = combined_pos_text.lower()

        # --- Slot Mining: Applications (longest alias match first, on positive text) ---
        for alias_low, canonical_app in sorted(self.app_alias_map.items(), key=lambda x: len(x[0]), reverse=True):
            if alias_low and alias_low in combined_pos_low:
                if canonical_app not in apps:
                    apps.append(canonical_app)
        apps = self._expand_apps(apps)

        # --- Slot Mining: Businesses (on positive text) ---
        for kw, canonical_biz in _BIZ_KEYWORDS:
            if kw in combined_pos_low:
                if canonical_biz not in bizs:
                    bizs.append(canonical_biz)

        # --- Slot Mining: Time Windows & Time Points (on positive text) ---
        for m in _QUERY_TIME_RE.finditer(combined_pos_text):
            tw = m.group(1)
            if tw not in time_windows:
                time_windows.append(tw)
        for m in _QUERY_CN_TIME_RE.finditer(combined_pos_text):
            h1, m1, h2, m2 = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4) or 0)
            tw = f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
            if tw not in time_windows:
                time_windows.append(tw)
        for m in _QUERY_CN_PT_RE.finditer(combined_pos_text):
            hh = int(m.group(1))
            mm = 30 if "半" in m.group(0) else int(m.group(2) or 0)
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                pt = f"{hh:02d}:{mm:02d}"
                if pt not in time_windows and not any(pt in w for w in time_windows):
                    time_windows.append(pt)

        # --- Slot Mining: Scenes (from existing scene_preference / environment rows) ---
        norm_combined_pos = _normalize_paraphrase_text(combined_pos_text)
        norm_neg = _normalize_paraphrase_text(neg_text)
        for row in slot_rows:
            sc = str(row.get("scene_col") or "").strip()
            if not sc:
                continue
            if neg_text and _fuzzy_phrase_match(sc, norm_neg) and not _fuzzy_phrase_match(sc, norm_combined_pos):
                if sc not in negated_scenes:
                    negated_scenes.append(sc)
                continue
            matched_sc = _fuzzy_phrase_match(sc, norm_combined_pos) or any(
                tok in sc and tok in norm_combined_pos
                for tok in ("大巴", "转场", "驻地", "外勤", "午休", "总结会")
            )
            if matched_sc and sc not in scenes:
                scenes.append(sc)
                if row.get("subtype") == "scene_preference":
                    s_app = str(row.get("app_col") or "").strip()
                    s_biz = str(row.get("biz_col") or "").strip()
                    if s_app and s_app not in apps and (not explicit_apps or s_app in explicit_apps):
                        if s_app not in negated_apps:
                            apps.append(s_app)
                    if s_biz and s_biz not in bizs and (not explicit_bizs or s_biz in explicit_bizs):
                        if s_biz not in negated_bizs:
                            bizs.append(s_biz)

        # If no explicit time window in query, check if now_ts provides a specific HH:MM
        if not time_windows and now_ts:
            m_ts = re.search(r"T(\d{2}:\d{2})", str(now_ts))
            if not m_ts:
                m_ts_cn = re.search(r"(\d{1,2})时(\d{1,2})分", str(now_ts))
                if m_ts_cn:
                    hh_mm = f"{int(m_ts_cn.group(1)):02d}:{int(m_ts_cn.group(2)):02d}"
                    if hh_mm != "00:00" and not any(time_matches_window(hh_mm, ntw) for ntw in negated_time_windows):
                        time_windows.append(hh_mm)
            else:
                hh_mm = m_ts.group(1)
                if hh_mm != "00:00" and not any(time_matches_window(hh_mm, ntw) for ntw in negated_time_windows):
                    time_windows.append(hh_mm)

        # --- Hop 2: Pointer Resolution (service_to_app & periodic_preference) ---
        if bizs and not apps:
            for row in slot_rows:
                if row.get("subtype") != "service_to_app":
                    continue
                s_biz = str(row.get("biz_col") or "").strip()
                s_app = str(row.get("app_col") or "").strip()
                if s_biz in bizs and s_app and s_app not in apps and s_app not in negated_apps:
                    apps.append(s_app)
                    pointer_hops.append(f"service_to_app[{s_biz}] -> {s_app}")
        elif apps and not bizs:
            for row in slot_rows:
                if row.get("subtype") != "service_to_app":
                    continue
                s_biz = str(row.get("biz_col") or "").strip()
                s_app = str(row.get("app_col") or "").strip()
                if s_app in apps and s_biz and s_biz not in bizs and s_biz not in negated_bizs:
                    bizs.append(s_biz)
                    pointer_hops.append(f"service_to_app[{s_app}] -> {s_biz}")

        if time_windows:
            for row in slot_rows:
                if row.get("subtype") != "periodic_preference":
                    continue
                cond_tw = str(row.get("time_window_col") or "").strip()
                if cond_tw and any(time_matches_window(qt, cond_tw) for qt in time_windows):
                    if any(time_matches_window(ntw, cond_tw) for ntw in negated_time_windows):
                        continue
                    p_app = str(row.get("app_col") or "").strip()
                    p_biz = str(row.get("biz_col") or "").strip()
                    if p_app in negated_apps or p_biz in negated_bizs:
                        continue
                    if explicit_apps and p_app and p_app not in explicit_apps:
                        continue
                    if explicit_bizs and p_biz and p_biz not in explicit_bizs:
                        continue
                    if p_app and p_app not in apps:
                        apps.append(p_app)
                    if p_biz and p_biz not in bizs:
                        bizs.append(p_biz)
                    if p_app or p_biz:
                        pointer_hops.append(f"periodic_preference[{cond_tw}] -> {p_app}|{p_biz}")

        if apps and bizs:
            for row in slot_rows:
                if row.get("subtype") != "periodic_preference":
                    continue
                p_app = str(row.get("app_col") or "").strip()
                p_biz = str(row.get("biz_col") or "").strip()
                cond_tw = str(row.get("time_window_col") or "").strip()
                if (
                    p_app in apps
                    and p_biz in bizs
                    and cond_tw
                    and cond_tw not in time_windows
                    and cond_tw not in negated_time_windows
                ):
                    time_windows.append(cond_tw)
                    pointer_hops.append(f"periodic_preference[{p_app}|{p_biz}] -> {cond_tw}")

        apps = self._expand_apps(apps)
        negated_apps = self._expand_apps(negated_apps)

        # Remove any positive slots that were only in negated lists if not explicitly requested in positive clauses
        return {
            "resolved_aliases": resolved_aliases,
            "applications": apps,
            "businesses": bizs,
            "scenes": scenes,
            "time_windows": time_windows,
            "explicit_applications": explicit_apps,
            "explicit_businesses": explicit_bizs,
            "negated_applications": [a for a in negated_apps if a not in apps],
            "negated_businesses": [b for b in negated_bizs if b not in bizs],
            "negated_scenes": [s for s in negated_scenes if s not in scenes],
            "negated_aliases": negated_aliases,
            "negated_time_windows": negated_time_windows,
            "pointer_hops": pointer_hops,
            "vector_anchor_hits": [],
            "expanded_query": combined_pos_text if combined_pos_text else q_raw,
            "alias_hit_ids": alias_hit_ids,
        }

    def _adsorb_slots_from_vector_hits(
        self,
        ns: Namespace,
        slot_plan: Dict[str, Any],
        vector_sim_map: Dict[str, float],
        q_emb: Sequence[float],
    ) -> bool:
        """Vector-Anchor-to-SQL-SELECT Bridge:
        When `q_emb` hits an `expression`, `scene_preference`, or business prototype vector with high cosine similarity,
        read the structured columns (`app_col`, `biz_col`, `scene_col`, `target_col`) from that SQLite row,
        run Hop 2 pointer resolution (`periodic_preference` / `service_to_app`), and return True if new slots unlocked.
        """
        slot_rows = self._get_active_slot_rows(ns)
        if not slot_rows:
            return False

        apps: List[str] = slot_plan["applications"]
        bizs: List[str] = slot_plan["businesses"]
        scenes: List[str] = slot_plan["scenes"]
        time_windows: List[str] = slot_plan["time_windows"]
        resolved_aliases: Dict[str, str] = slot_plan["resolved_aliases"]
        alias_hit_ids: Dict[str, str] = slot_plan["alias_hit_ids"]
        pointer_hops: List[str] = slot_plan["pointer_hops"]
        vector_anchor_hits: List[Dict[str, Any]] = slot_plan.setdefault("vector_anchor_hits", [])
        explicit_apps = set(slot_plan.get("explicit_applications") or [])
        explicit_bizs = set(slot_plan.get("explicit_businesses") or [])
        negated_apps = set(slot_plan.get("negated_applications") or [])
        negated_bizs = set(slot_plan.get("negated_businesses") or [])
        negated_scenes = set(slot_plan.get("negated_scenes") or [])
        negated_aliases = set(slot_plan.get("negated_aliases") or [])
        negated_tws = list(slot_plan.get("negated_time_windows") or [])

        changed = False

        # 1. Adsorb expression & scene_preference anchor rows from SQLite vector_sim_map
        for row in slot_rows:
            mid = row.get("memory_id") or ""
            st = row.get("subtype") or ""
            v_sim = float(vector_sim_map.get(mid, 0.0))
            if v_sim <= 0.0 and row.get("embedding_json"):
                try:
                    r_emb = json.loads(row["embedding_json"])
                    if r_emb:
                        v_sim = _cosine_similarity(q_emb, r_emb)
                except Exception:
                    v_sim = 0.0

            if st == "expression" and v_sim >= 0.42:
                alias = str(row.get("alias_col") or row.get("memory_key") or "").strip()
                target = str(row.get("target_col") or "").strip()
                if not alias or not target or alias in resolved_aliases or alias in negated_aliases:
                    continue
                bound_app = str(row.get("app_col") or "").strip()
                bound_biz = str(row.get("biz_col") or "").strip()
                bound_scene = str(row.get("scene_col") or "").strip()
                if explicit_apps and bound_app and bound_app not in explicit_apps:
                    continue
                if explicit_bizs and bound_biz and bound_biz not in explicit_bizs:
                    continue
                if bound_app in negated_apps or bound_biz in negated_bizs:
                    continue

                resolved_aliases[alias] = target
                scope_str = f" (作用域: {bound_app}|{bound_biz})" if bound_app or bound_biz else ""
                alias_hit_ids[mid] = f"向量锚点吸附词典(v_sim={v_sim:.2f}): {alias} -> {target}{scope_str}"
                vector_anchor_hits.append(
                    {
                        "memory_id": mid,
                        "subtype": "expression",
                        "anchor": alias,
                        "target": target,
                        "vector_sim": round(v_sim, 4),
                        "unlocked_slots": {
                            "application": bound_app,
                            "business": bound_biz,
                            "scene": bound_scene,
                            "target": target,
                        },
                    }
                )
                for m_tw in _QUERY_TIME_RE.finditer(target):
                    tw = m_tw.group(1)
                    if tw not in time_windows and not any(time_matches_window(tw, ntw) for ntw in negated_tws):
                        time_windows.append(tw)
                        changed = True
                if bound_app and bound_app not in apps:
                    apps.append(bound_app)
                    changed = True
                if bound_biz and bound_biz not in bizs:
                    bizs.append(bound_biz)
                    changed = True
                if bound_scene and bound_scene not in scenes:
                    scenes.append(bound_scene)
                    changed = True

            elif st in ("scene_preference", "environment") and v_sim >= 0.42:
                sc = str(row.get("scene_col") or "").strip()
                if not sc or sc in scenes or sc in negated_scenes:
                    continue
                s_app = str(row.get("app_col") or "").strip()
                s_biz = str(row.get("biz_col") or "").strip()
                if explicit_apps and s_app and s_app not in explicit_apps:
                    continue
                if explicit_bizs and s_biz and s_biz not in explicit_bizs:
                    continue
                if s_app in negated_apps or s_biz in negated_bizs:
                    continue
                scenes.append(sc)
                changed = True
                if s_app and s_app not in apps:
                    apps.append(s_app)
                if s_biz and s_biz not in bizs:
                    bizs.append(s_biz)
                vector_anchor_hits.append(
                    {
                        "memory_id": mid,
                        "subtype": st,
                        "anchor": sc,
                        "target": f"{s_app}|{s_biz}",
                        "vector_sim": round(v_sim, 4),
                        "unlocked_slots": {"scene": sc, "application": s_app, "business": s_biz},
                    }
                )

        # 2. Business Prototype Vector Bank adsorption (if business is still unresolved)
        if not bizs and q_emb:
            proto_vecs = self._get_biz_prototype_vecs()
            best_biz = ""
            best_sim = 0.0
            for b_code, p_vec in proto_vecs.items():
                if b_code in negated_bizs:
                    continue
                b_sim = _cosine_similarity(q_emb, p_vec)
                if b_sim > best_sim:
                    best_sim = b_sim
                    best_biz = b_code
            if best_biz and best_sim >= 0.38:
                bizs.append(best_biz)
                changed = True
                vector_anchor_hits.append(
                    {
                        "memory_id": f"PROTO_{best_biz.upper()}",
                        "subtype": "business_prototype",
                        "anchor": _BIZ_PROTOTYPE_TEXTS.get(best_biz, best_biz)[:18],
                        "target": best_biz,
                        "vector_sim": round(best_sim, 4),
                        "unlocked_slots": {"business": best_biz},
                    }
                )

        if not changed:
            return False

        # 3. Re-run Hop 2 Pointer Resolution on newly unlocked slots
        if bizs and not apps:
            for row in slot_rows:
                if row.get("subtype") != "service_to_app":
                    continue
                s_biz = str(row.get("biz_col") or "").strip()
                s_app = str(row.get("app_col") or "").strip()
                if s_biz in bizs and s_app and s_app not in apps and s_app not in negated_apps:
                    apps.append(s_app)
                    hop_str = f"service_to_app[{s_biz}] -> {s_app}"
                    if hop_str not in pointer_hops:
                        pointer_hops.append(hop_str)
        elif apps and not bizs:
            for row in slot_rows:
                if row.get("subtype") != "service_to_app":
                    continue
                s_biz = str(row.get("biz_col") or "").strip()
                s_app = str(row.get("app_col") or "").strip()
                if s_app in apps and s_biz and s_biz not in bizs and s_biz not in negated_bizs:
                    bizs.append(s_biz)
                    hop_str = f"service_to_app[{s_app}] -> {s_biz}"
                    if hop_str not in pointer_hops:
                        pointer_hops.append(hop_str)

        if time_windows:
            for row in slot_rows:
                if row.get("subtype") != "periodic_preference":
                    continue
                cond_tw = str(row.get("time_window_col") or "").strip()
                if cond_tw and any(time_matches_window(qt, cond_tw) for qt in time_windows):
                    if any(time_matches_window(ntw, cond_tw) for ntw in negated_tws):
                        continue
                    p_app = str(row.get("app_col") or "").strip()
                    p_biz = str(row.get("biz_col") or "").strip()
                    if p_app in negated_apps or p_biz in negated_bizs:
                        continue
                    if explicit_apps and p_app and p_app not in explicit_apps:
                        continue
                    if explicit_bizs and p_biz and p_biz not in explicit_bizs:
                        continue
                    if p_app and p_app not in apps:
                        apps.append(p_app)
                    if p_biz and p_biz not in bizs:
                        bizs.append(p_biz)
                    if p_app or p_biz:
                        hop_str = f"periodic_preference[{cond_tw}] -> {p_app}|{p_biz}"
                        if hop_str not in pointer_hops:
                            pointer_hops.append(hop_str)

        slot_plan["applications"] = self._expand_apps(apps)
        return True

    def search(
        self,
        ns: Namespace,
        query: str,
        top_k: int = 10,
        levels: str = "L0+L1",
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        follow_edges: int = 0,
        bump_heat: bool = True,
        now_ts: Optional[str] = None,
        query_slots: Optional[Dict[str, Any]] = None,
        cascade_short_circuit: bool = False,
    ) -> List[Dict[str, Any]]:
        ts = normalize_ts_str(now_ts)

        # Resolve memory_types filter: allow passing either memory_type ('working_memory') or subtype ('working_record')
        resolved_subtypes: Optional[List[str]] = list(subtypes) if subtypes else None
        resolved_mtypes: Optional[List[str]] = None
        if memory_types:
            sub_list: List[str] = list(resolved_subtypes or [])
            mt_list: List[str] = []
            for mt in memory_types:
                if mt in self.registry:
                    sub_list.append(mt)
                else:
                    mt_list.append(mt)
            if sub_list and not mt_list:
                resolved_subtypes = sub_list
            elif mt_list and not sub_list:
                resolved_mtypes = mt_list
            else:
                # Expand memory_types into subtypes so combined filter works seamlessly
                for spec in self.registry.all_specs():
                    if spec.memory_type in mt_list:
                        sub_list.append(spec.subtype)
                resolved_subtypes = list(dict.fromkeys(sub_list))

        sim_map: Dict[str, float] = {}
        select_sim_map: Dict[str, float] = {}
        vector_sim_map: Dict[str, float] = {}
        fts_sim_map: Dict[str, float] = {}
        structured_reasons: Dict[str, str] = {}

        # 0. Two-Hop Alias Resolution + Keyword Slot Mining + Structured SQL SELECT (§2.5)
        slot_plan = self._mine_slots_and_resolve_two_hop(
            ns, query, now_ts=now_ts, query_slots=query_slots
        )
        expanded_query = slot_plan.get("expanded_query") or query

        for alias_mid, alias_reason in (slot_plan.get("alias_hit_ids") or {}).items():
            sim_map[alias_mid] = max(sim_map.get(alias_mid, 0.0), 0.88)
            select_sim_map[alias_mid] = max(select_sim_map.get(alias_mid, 0.0), 0.88)
            structured_reasons[alias_mid] = alias_reason

        max_select_sim = 0.0
        if any(slot_plan.get(k) for k in ("applications", "businesses", "scenes", "time_windows")):
            try:
                for mid, s_sim, s_reason in self.index.select_by_slots(
                    ns,
                    applications=slot_plan.get("applications"),
                    businesses=slot_plan.get("businesses"),
                    scenes=slot_plan.get("scenes"),
                    time_windows=slot_plan.get("time_windows"),
                    subtypes=resolved_subtypes,
                    memory_types=resolved_mtypes,
                    statuses=statuses,
                ):
                    f_ssim = float(s_sim)
                    sim_map[mid] = max(sim_map.get(mid, 0.0), f_ssim)
                    select_sim_map[mid] = max(select_sim_map.get(mid, 0.0), f_ssim)
                    max_select_sim = max(max_select_sim, f_ssim)
                    hop_prefix = ""
                    if slot_plan.get("resolved_aliases"):
                        al_str = ", ".join(f"{k}->{v}" for k, v in slot_plan["resolved_aliases"].items())
                        hop_prefix += f"[个性化词典展开: {al_str}] "
                    if slot_plan.get("pointer_hops"):
                        hop_prefix += f"[指针跳: {'; '.join(slot_plan['pointer_hops'])}] "
                    structured_reasons[mid] = hop_prefix + s_reason
            except Exception:
                pass

        # Cascading short-circuit check: if cascade_short_circuit is enabled and SQL SELECT already locked onto
        # a complete parameter card (select_sim >= 0.93), skip remote Embedding API call.
        vector_short_circuited = bool(cascade_short_circuit and max_select_sim >= 0.93)

        # 1. Dense vector recall (scoped by ns, using expanded_query) + Vector Anchor Slot Adsorption
        if not vector_short_circuited and self.embedding_provider is not None and expanded_query.strip():
            try:
                q_emb = self.embedding_provider.embed_one(expanded_query)
                for mid, v_sim in self.index.search_vector(
                    ns,
                    q_emb,
                    subtypes=resolved_subtypes,
                    memory_types=resolved_mtypes,
                    statuses=statuses,
                    top_k=max(top_k * 3, 30),
                ):
                    sim_map[mid] = max(sim_map.get(mid, 0.0), float(v_sim))
                    vector_sim_map[mid] = max(vector_sim_map.get(mid, 0.0), float(v_sim))

                # 1.5 Vector-Anchor Adsorption -> Read Row Slot Columns -> Hop 2 -> SQL SELECT Bridge
                if self._adsorb_slots_from_vector_hits(ns, slot_plan, vector_sim_map, q_emb):
                    for alias_mid, alias_reason in (slot_plan.get("alias_hit_ids") or {}).items():
                        sim_map[alias_mid] = max(sim_map.get(alias_mid, 0.0), 0.88)
                        select_sim_map[alias_mid] = max(select_sim_map.get(alias_mid, 0.0), 0.88)
                        structured_reasons[alias_mid] = alias_reason

                    for mid, s_sim, s_reason in self.index.select_by_slots(
                        ns,
                        applications=slot_plan.get("applications"),
                        businesses=slot_plan.get("businesses"),
                        scenes=slot_plan.get("scenes"),
                        time_windows=slot_plan.get("time_windows"),
                        subtypes=resolved_subtypes,
                        memory_types=resolved_mtypes,
                        statuses=statuses,
                    ):
                        f_ssim = float(s_sim)
                        sim_map[mid] = max(sim_map.get(mid, 0.0), f_ssim)
                        select_sim_map[mid] = max(select_sim_map.get(mid, 0.0), f_ssim)
                        max_select_sim = max(max_select_sim, f_ssim)
                        hop_prefix = ""
                        if slot_plan.get("vector_anchor_hits"):
                            anc_str = ", ".join(
                                f"{h['anchor']}(cos={h['vector_sim']})->{h['target']}"
                                for h in slot_plan["vector_anchor_hits"]
                            )
                            hop_prefix += f"[向量锚点吸附: {anc_str}] "
                        if slot_plan.get("pointer_hops"):
                            hop_prefix += f"[指针跳: {'; '.join(slot_plan['pointer_hops'])}] "
                        structured_reasons[mid] = hop_prefix + s_reason
            except Exception:
                pass

        # 2. Sparse FTS5 keyword recall (scoped by ns, using expanded_query)
        if expanded_query.strip():
            try:
                for mid, f_sim in self.index.search_fts(
                    ns,
                    expanded_query,
                    subtypes=resolved_subtypes,
                    memory_types=resolved_mtypes,
                    statuses=statuses,
                    top_k=max(top_k * 3, 30),
                ):
                    sim_map[mid] = max(sim_map.get(mid, 0.0), float(f_sim))
                    fts_sim_map[mid] = max(fts_sim_map.get(mid, 0.0), float(f_sim))
            except Exception:
                pass

        # 3. Also include scoped candidates from file/index if query is empty or keyword substring matches
        scoped_ids = self.index.list_scoped_ids(
            ns,
            subtypes=resolved_subtypes,
            memory_types=resolved_mtypes,
            statuses=statuses,
        )
        for mid in scoped_ids:
            if mid not in sim_map:
                sim_map[mid] = 0.0

        # Load authoritative nodes from FileStore for recalled IDs in sim_map
        if not sim_map:
            return []

        id_to_node: Dict[str, MemoryNode] = {}
        for node in self.filestore.iter_nodes(ns):
            if node.memory_id not in sim_map:
                continue
            if statuses:
                if node.status not in statuses:
                    continue
            else:
                if node.status in ("archived", "expired", "superseded"):
                    continue
            if resolved_subtypes and node.subtype not in resolved_subtypes:
                continue
            if resolved_mtypes and node.memory_type not in resolved_mtypes:
                continue
            id_to_node[node.memory_id] = node

        scored_items: List[Dict[str, Any]] = []
        q_low = (expanded_query or query or "").strip().lower()
        neg_apps_set = set(slot_plan.get("negated_applications") or [])
        neg_bizs_set = set(slot_plan.get("negated_businesses") or [])
        neg_scenes_set = set(slot_plan.get("negated_scenes") or [])
        neg_aliases_set = set(slot_plan.get("negated_aliases") or [])
        neg_tws_list = list(slot_plan.get("negated_time_windows") or [])

        for mid, base_sim in sim_map.items():
            node = id_to_node.get(mid)
            if node is None:
                continue

            # Boost keyword / trigger_conditions match for experience cards & exact key matches
            sim = float(base_sim)
            if q_low:
                searchable = build_search_text(node).lower()
                if q_low in searchable:
                    sim = max(sim, 0.85)
                # Check experience trigger_conditions (keywords / task_keywords / location_keywords) (§2.7)
                trig = node.content.get("trigger_conditions")
                if isinstance(trig, dict):
                    all_kws: List[Any] = []
                    for tk in ("keywords", "task_keywords", "location_keywords"):
                        sub_kws = trig.get(tk)
                        if isinstance(sub_kws, list):
                            all_kws.extend(sub_kws)
                    if all_kws:
                        matched_kws = [kw for kw in all_kws if str(kw).lower() in q_low or q_low in str(kw).lower()]
                        if matched_kws:
                            sim = max(sim, min(1.0, 0.75 + 0.1 * len(matched_kws)))

            freshness, stability_factor, eff_lam = self._compute_freshness_and_stability(node, ts)
            eff_heat = self._compute_effective_heat(node, ts)
            conf = float(node.confidence)
            lam = self._get_decay_lambda(node.subtype)

            raw_score = (
                self.alpha * sim
                + self.beta * freshness
                + self.gamma * eff_heat
                + self.delta * conf
            )
            # Temporary node 0.5x penalty (§2.5 & §4 Pitfall 2, search.py:163-164)
            final_score = raw_score * self.temporary_weight if node.validity.type == "temporary" else raw_score

            # Clause-Level Negation Suppression Penalty (0.15x) if node belongs to a negated slot
            n_app = str(node.content.get("application") or "").strip()
            n_biz = str(node.content.get("business") or "").strip()
            n_scene = str(node.content.get("scene") or node.content.get("environment") or "").strip()
            n_alias = str(node.content.get("alias") or node.content.get("source_expression") or "").strip()
            n_tw = str(node.content.get("time_window") or node.content.get("condition") or "").strip()
            negation_penalized = False
            if (
                (n_app and n_app in neg_apps_set)
                or (n_biz and n_biz in neg_bizs_set)
                or (n_scene and n_scene in neg_scenes_set)
                or (n_alias and n_alias in neg_aliases_set)
                or (n_tw and any(time_matches_window(ntw, n_tw) for ntw in neg_tws_list))
            ):
                final_score *= 0.15
                negation_penalized = True

            scored_items.append(
                {
                    "node": node,
                    "sim": sim,
                    "select_sim": select_sim_map.get(mid, 0.0),
                    "vector_sim": vector_sim_map.get(mid, 0.0),
                    "fts_sim": fts_sim_map.get(mid, 0.0),
                    "freshness": freshness,
                    "stability_factor": stability_factor,
                    "effective_lambda": eff_lam,
                    "heat": eff_heat,
                    "confidence": conf,
                    "decay_lambda": lam,
                    "structured_match": structured_reasons.get(mid, ""),
                    "query_slot_plan": slot_plan,
                    "vector_short_circuited": vector_short_circuited,
                    "negation_penalized": negation_penalized,
                    "final_score": final_score,
                }
            )

        scored_items.sort(key=lambda x: x["final_score"], reverse=True)
        top_items = scored_items[:top_k]

        results: List[Dict[str, Any]] = []
        for item in top_items:
            node: MemoryNode = item["node"]
            # Bump heat on hit before returning (§2.3, search.py:137-143)
            if bump_heat and self.heat_callback is not None:
                try:
                    self.heat_callback(ns, node, self.heat_bump_on_hit)
                except Exception:
                    pass

            res_dict = self._format_result_item(ns, node, item, levels=levels, follow_edges=follow_edges)
            results.append(res_dict)

        return results

    def _format_result_item(
        self,
        ns: Namespace,
        node: MemoryNode,
        score_info: Dict[str, Any],
        levels: str = "L0+L1",
        follow_edges: int = 0,
    ) -> Dict[str, Any]:
        lvl = (levels or "L0+L1").upper()
        result: Dict[str, Any] = {
            "memory_id": node.memory_id,
            "memory_type": node.memory_type,
            "subtype": node.subtype,
            "memory_key": node.memory_key,
            "status": node.status,
            "validity": node.validity.type,
            "score": round(float(score_info["final_score"]), 6),
            "description": node.description,
            "content": dict(node.content),
            "explain": {
                "sim": round(float(score_info["sim"]), 6),
                "select_sim": round(float(score_info.get("select_sim", 0.0)), 6),
                "vector_sim": round(float(score_info.get("vector_sim", 0.0)), 6),
                "fts_sim": round(float(score_info.get("fts_sim", 0.0)), 6),
                "freshness": round(float(score_info["freshness"]), 6),
                "stability_factor": round(float(score_info.get("stability_factor", 1.0)), 4),
                "effective_lambda": round(float(score_info.get("effective_lambda", score_info["decay_lambda"])), 6),
                "heat": round(float(score_info["heat"]), 6),
                "confidence": round(float(score_info["confidence"]), 6),
                "decay_lambda": round(float(score_info["decay_lambda"]), 6),
                "validity": node.validity.type,
                "structured_match": score_info.get("structured_match", ""),
                "vector_anchor_hits": (score_info.get("query_slot_plan") or {}).get("vector_anchor_hits", []),
                "vector_short_circuited": bool(score_info.get("vector_short_circuited", False)),
                "negation_penalized": bool(score_info.get("negation_penalized", False)),
                "final_score": round(float(score_info["final_score"]), 6),
                "memory_key": node.memory_key,
            },
        }
        if lvl == "L0":
            result["overview"] = None
            result["content_md"] = None
        elif lvl in ("L0+L1", "L1"):
            result["overview"] = node.overview
            result["content_md"] = None
        else:
            # "FULL" or "L0+L1+L2"
            result["overview"] = node.overview
            result["content_md"] = self.filestore.read_content_md(ns, node)

        if follow_edges > 0 and self.graph is not None:
            result["neighbors"] = self.graph.neighbors(ns, node.memory_id, max_hops=follow_edges)

        return result
