"""Hybrid retrieval engine orchestrating query segmentation, vector scoring, and relational expansion."""
from __future__ import annotations

import sqlite3
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.models import MemoryNode, MemoryStatus
from ..core.namespace import Namespace
from ..core.registry import MemoryRegistry
from ..pipeline.updater import (
    compute_decayed_heat,
    compute_ebbinghaus_retention,
    normalize_ts_str,
)
from ..providers.embedding import EmbeddingProvider
from ..storage.filestore import FileStore
from ..storage.graph import GraphStore
from ..storage.index import (
    MemoryIndex,
    _cosine_similarity,
    build_search_text,
    time_matches_window,
)
from .query_segmentation import (
    _BIZ_KEYWORDS,
    _BIZ_PROTOTYPE_TEXTS,
    _CJK_CHARS_RE,
    _CLAUSE_SPLIT_RE,
    _QUERY_CN_PT_RE,
    _QUERY_CN_TIME_RE,
    _QUERY_TIME_RE,
    _SLOT_NEGATION_RE,
    _build_sliding_windows,
    _extract_positive_clauses,
    _fuzzy_phrase_match,
    _load_default_app_map,
    _parse_relative_date_window,
    _split_positive_and_negated_clauses,
)
from .relational_expander import mine_slots_and_resolve_two_hop
from .vector_scorer import (
    evaluate_node_Trigger_boost,
    format_search_result_item,
    project_slots_from_vector_hits,
)

__all__ = [
    "HybridSearchEngine",
    "SearchEngine",
    "_BIZ_KEYWORDS",
    "_BIZ_PROTOTYPE_TEXTS",
    "_CJK_CHARS_RE",
    "_CLAUSE_SPLIT_RE",
    "_QUERY_CN_PT_RE",
    "_QUERY_CN_TIME_RE",
    "_QUERY_TIME_RE",
    "_SLOT_NEGATION_RE",
    "_build_sliding_windows",
    "_extract_positive_clauses",
    "_fuzzy_phrase_match",
    "_load_default_app_map",
    "_parse_relative_date_window",
    "_split_positive_and_negated_clauses",
]


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
        """effective_heat = heat * 0.5^(delta_hours / half_life) (§2.3)."""
        return compute_decayed_heat(
            node.stats.heat,
            node.stats.last_heat_at,
            now_ts=now_ts,
            half_life_hours=self.heat_half_life_hours,
        )

    def _get_decay_lambda(self, subtype: str) -> float:
        """Read Ebbinghaus decay_lambda from registry (§2.4)."""
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
        except sqlite3.Error:
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
        slot_rows = self._get_active_slot_rows(ns)
        return mine_slots_and_resolve_two_hop(
            slot_rows=slot_rows,
            app_alias_map=self.app_alias_map,
            expand_apps_fn=self._expand_apps,
            query=query,
            now_ts=now_ts,
            query_slots=query_slots,
        )

    def _adsorb_slots_from_vector_hits(
        self,
        ns: Namespace,
        slot_plan: Dict[str, Any],
        vector_sim_map: Dict[str, float],
        q_emb: Sequence[float],
        window_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> bool:
        slot_rows = self._get_active_slot_rows(ns)
        return project_slots_from_vector_hits(
            slot_rows=slot_rows,
            slot_plan=slot_plan,
            vector_sim_map=vector_sim_map,
            q_emb=q_emb,
            window_embeddings=window_embeddings,
            biz_prototype_vecs=self._get_biz_prototype_vecs(),
            expand_apps_fn=self._expand_apps,
        )

    def _resolve_type_filters(
        self,
        subtypes: Optional[Sequence[str]],
        memory_types: Optional[Sequence[str]],
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        resolved_subtypes: Optional[List[str]] = list(subtypes) if subtypes else None
        resolved_mtypes: Optional[List[str]] = None
        if not memory_types:
            return resolved_subtypes, resolved_mtypes

        sub_list: List[str] = list(resolved_subtypes or [])
        mt_list: List[str] = []
        for mt in memory_types:
            if mt in self.registry:
                sub_list.append(mt)
            else:
                mt_list.append(mt)
        if sub_list and not mt_list:
            return sub_list, None
        if mt_list and not sub_list:
            return resolved_subtypes, mt_list
        for spec in self.registry.all_specs():
            if spec.memory_type in mt_list:
                sub_list.append(spec.subtype)
        return list(dict.fromkeys(sub_list)), None

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
        """Execute hybrid retrieval combining structured SQL SELECT, sliding-window vector recall, and FTS5."""
        ts = normalize_ts_str(now_ts)
        resolved_subtypes, resolved_mtypes = self._resolve_type_filters(subtypes, memory_types)

        sim_map: Dict[str, float] = {}
        select_sim_map: Dict[str, float] = {}
        vector_sim_map: Dict[str, float] = {}
        fts_sim_map: Dict[str, float] = {}
        structured_reasons: Dict[str, str] = {}

        # 0. Two-stage personalized expression resolution + keyword slot mining + SQL SELECT (§2.5)
        slot_plan = self._mine_slots_and_resolve_two_hop(
            ns, query, now_ts=now_ts, query_slots=query_slots
        )
        expanded_query = slot_plan.get("expanded_query") or query

        for alias_mid, alias_reason in (slot_plan.get("alias_hit_ids") or {}).items():
            sim_map[alias_mid] = max(sim_map.get(alias_mid, 0.0), 0.88)
            select_sim_map[alias_mid] = max(select_sim_map.get(alias_mid, 0.0), 0.88)
            structured_reasons[alias_mid] = alias_reason

        max_select_sim = 0.0
        if any(
            slot_plan.get(k)
            for k in ("applications", "businesses", "scenes", "time_windows", "relative_date_range")
        ):
            try:
                for mid, s_sim, s_reason in self.index.select_by_slots(
                    ns,
                    applications=slot_plan.get("applications"),
                    businesses=slot_plan.get("businesses"),
                    scenes=slot_plan.get("scenes"),
                    time_windows=slot_plan.get("time_windows"),
                    relative_date_range=slot_plan.get("relative_date_range"),
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
            except sqlite3.Error:
                pass

        vector_short_circuited = bool(cascade_short_circuit and max_select_sim >= 0.93)

        # 1. Dense vector recall (scoped by ns, using 6~10 char sliding windows + Max-Pooling)
        if not vector_short_circuited and self.embedding_provider is not None and expanded_query.strip():
            try:
                q_emb = self.embedding_provider.embed_one(expanded_query)
                sliding_windows = list(slot_plan.get("sliding_windows") or [])
                extra_windows = [w for w in sliding_windows if w and w != expanded_query]
                extra_embs = self.embedding_provider.embed(extra_windows) if extra_windows else []
                window_embs: List[Sequence[float]] = [q_emb] + list(extra_embs)

                for mid, v_sim in self.index.search_vector_multi_windows(
                    ns,
                    window_embs,
                    subtypes=resolved_subtypes,
                    memory_types=resolved_mtypes,
                    statuses=statuses,
                    top_k=max(top_k * 3, 30),
                ):
                    sim_map[mid] = max(sim_map.get(mid, 0.0), float(v_sim))
                    vector_sim_map[mid] = max(vector_sim_map.get(mid, 0.0), float(v_sim))

                # 1.5 Vector Similarity Retrieval & Row-Level Slot Projection -> Stage-2 Completion -> SQL SELECT
                if self._adsorb_slots_from_vector_hits(
                    ns, slot_plan, vector_sim_map, q_emb, window_embeddings=window_embs
                ):
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
                        relative_date_range=slot_plan.get("relative_date_range"),
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
            except sqlite3.Error:
                pass

        # 3. Also include scoped candidates from file/index if query is empty or keyword substring matches
        scoped_ids = self.index.list_scoped_ids(
            ns,
            subtypes=resolved_subtypes,
            memory_types=resolved_mtypes,
            statuses=statuses,
        )
        for mid in scoped_ids:
            sim_map.setdefault(mid, 0.0)

        if not sim_map:
            return []

        id_to_node: Dict[str, MemoryNode] = {}
        for node in self.filestore.iter_nodes(ns):
            if node.memory_id not in sim_map:
                continue
            if statuses:
                if node.status not in statuses:
                    continue
            elif node.status in ("archived", "expired", "superseded"):
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

            sim = evaluate_node_Trigger_boost(node, mid, base_sim, q_low, vector_sim_map, structured_reasons)
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
            final_score = raw_score * self.temporary_weight if node.validity.type == "temporary" else raw_score

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
            node = item["node"]
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
        return format_search_result_item(
            filestore=self.filestore,
            graph=self.graph,
            ns=ns,
            node=node,
            score_info=score_info,
            levels=levels,
            follow_edges=follow_edges,
        )


HybridSearchEngine = SearchEngine
