"""Gated multi-window max-pooling similarity, row-level slot projection, and hybrid candidate scoring."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from ..core.models import MemoryNode
from ..core.namespace import Namespace
from ..storage.filestore import FileStore
from ..storage.graph import GraphStore
from ..storage.index import _cosine_similarity, build_search_text, time_matches_window
from .query_segmentation import _BIZ_PROTOTYPE_TEXTS, _QUERY_TIME_RE
from .relational_expander import check_periodic_time_conflict, expand_relational_pointers


def _extract_expression_row_slots(
    row: Dict[str, Any],
    mid: str,
    v_sim: float,
    slot_rows: Sequence[Dict[str, Any]],
    slot_plan: Dict[str, Any],
    explicit_apps: Set[str],
    explicit_bizs: Set[str],
    negated_apps: Set[str],
    negated_bizs: Set[str],
    negated_aliases: Set[str],
    negated_tws: Sequence[str],
) -> bool:
    alias = str(row.get("alias_col") or row.get("memory_key") or "").strip()
    target = str(row.get("target_col") or "").strip()
    resolved_aliases: Dict[str, str] = slot_plan["resolved_aliases"]
    if not alias or not target or alias in resolved_aliases or alias in negated_aliases:
        return False
    bound_app = str(row.get("app_col") or "").strip()
    bound_biz = str(row.get("biz_col") or "").strip()
    bound_scene = str(row.get("scene_col") or "").strip()
    if (explicit_apps and bound_app and bound_app not in explicit_apps) or (
        explicit_bizs and bound_biz and bound_biz not in explicit_bizs
    ):
        return False
    if bound_app in negated_apps or bound_biz in negated_bizs:
        return False

    tw_matches = [m_tw.group(1) for m_tw in _QUERY_TIME_RE.finditer(target)]
    if check_periodic_time_conflict(
        slot_rows, tw_matches, explicit_apps, explicit_bizs, negated_apps, negated_bizs
    ):
        return False

    resolved_aliases[alias] = target
    scope_str = f" (作用域: {bound_app}|{bound_biz})" if bound_app or bound_biz else ""
    slot_plan["alias_hit_ids"][mid] = f"向量锚点吸附词典(v_sim={v_sim:.2f}): {alias} -> {target}{scope_str}"
    slot_plan.setdefault("vector_anchor_hits", []).append(
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
    changed = False
    for tw in tw_matches:
        if tw not in slot_plan["time_windows"] and not any(time_matches_window(tw, ntw) for ntw in negated_tws):
            slot_plan["time_windows"].append(tw)
            changed = True
    for val, dst in ((bound_app, slot_plan["applications"]), (bound_biz, slot_plan["businesses"]), (bound_scene, slot_plan["scenes"])):
        if val and val not in dst:
            dst.append(val)
            changed = True
    return changed


def _extract_hit_row_slots(
    row: Dict[str, Any],
    v_sim: float,
    slot_rows: Sequence[Dict[str, Any]],
    slot_plan: Dict[str, Any],
    explicit_apps: Set[str],
    explicit_bizs: Set[str],
    negated_apps: Set[str],
    negated_bizs: Set[str],
    negated_scenes: Set[str],
    negated_aliases: Set[str],
    negated_tws: Sequence[str],
) -> bool:
    """Extract row-level structured slot columns from a single high-similarity expression or scene row."""
    mid = row.get("memory_id") or ""
    st = row.get("subtype") or ""
    if st == "expression" and v_sim >= 0.52:
        return _extract_expression_row_slots(
            row, mid, v_sim, slot_rows, slot_plan, explicit_apps, explicit_bizs, negated_apps, negated_bizs, negated_aliases, negated_tws
        )

    if st in ("scene_preference", "environment", "working_record") and v_sim >= 0.42:
        sc = str(row.get("scene_col") or "").strip()
        if not sc or sc in slot_plan["scenes"] or sc in negated_scenes:
            return False
        s_app = str(row.get("app_col") or "").strip()
        s_biz = str(row.get("biz_col") or "").strip()
        if (explicit_apps and s_app and s_app not in explicit_apps) or (
            explicit_bizs and s_biz and s_biz not in explicit_bizs
        ):
            return False
        if s_app in negated_apps or s_biz in negated_bizs:
            return False
        slot_plan["scenes"].append(sc)
        if s_app and s_app not in slot_plan["applications"]:
            slot_plan["applications"].append(s_app)
        if s_biz and s_biz not in slot_plan["businesses"]:
            slot_plan["businesses"].append(s_biz)
        slot_plan.setdefault("vector_anchor_hits", []).append(
            {
                "memory_id": mid,
                "subtype": st,
                "anchor": sc,
                "target": f"{s_app}|{s_biz}",
                "vector_sim": round(v_sim, 4),
                "unlocked_slots": {"scene": sc, "application": s_app, "business": s_biz},
            }
        )
        return True

    return False


def _merge_projected_slot_candidates(
    slot_rows: Sequence[Dict[str, Any]],
    slot_plan: Dict[str, Any],
    all_q_embs: Sequence[Sequence[float]],
    biz_prototype_vecs: Dict[str, List[float]],
    expand_apps_fn: Callable[[Sequence[str]], List[str]],
    explicit_apps: Set[str],
    explicit_bizs: Set[str],
    negated_apps: Set[str],
    negated_bizs: Set[str],
    negated_tws: Sequence[str],
    row_changed: bool,
) -> bool:
    """Match business prototype vectors if needed and execute stage-2 relational pointer expansion."""
    bizs: List[str] = slot_plan["businesses"]
    vector_anchor_hits: List[Dict[str, Any]] = slot_plan.setdefault("vector_anchor_hits", [])
    changed = row_changed

    if not bizs and all_q_embs and biz_prototype_vecs:
        best_biz = ""
        best_sim = 0.0
        for b_code, p_vec in biz_prototype_vecs.items():
            if b_code in negated_bizs:
                continue
            b_sim = max((_cosine_similarity(w_e, p_vec) for w_e in all_q_embs), default=0.0)
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

    expand_relational_pointers(
        slot_rows=slot_rows,
        apps=slot_plan["applications"],
        bizs=bizs,
        time_windows=slot_plan["time_windows"],
        pointer_hops=slot_plan["pointer_hops"],
        explicit_apps=explicit_apps,
        explicit_bizs=explicit_bizs,
        negated_apps=negated_apps,
        negated_bizs=negated_bizs,
        negated_time_windows=negated_tws,
        include_reverse_periodic=False,
    )
    slot_plan["applications"] = expand_apps_fn(slot_plan["applications"])
    return True


def project_slots_from_vector_hits(
    slot_rows: Sequence[Dict[str, Any]],
    slot_plan: Dict[str, Any],
    vector_sim_map: Dict[str, float],
    q_emb: Sequence[float],
    window_embeddings: Optional[Sequence[Sequence[float]]],
    biz_prototype_vecs: Dict[str, List[float]],
    expand_apps_fn: Callable[[Sequence[str]], List[str]],
) -> bool:
    """Vector Similarity Retrieval & Row-Level Slot Projection -> Stage-2 Relational Completion."""
    if not slot_rows:
        return False

    all_q_embs: List[Sequence[float]] = [w for w in (window_embeddings or []) if w]
    if q_emb and not all_q_embs:
        all_q_embs = [q_emb]

    explicit_apps = set(slot_plan.get("explicit_applications") or [])
    explicit_bizs = set(slot_plan.get("explicit_businesses") or [])
    negated_apps = set(slot_plan.get("negated_applications") or [])
    negated_bizs = set(slot_plan.get("negated_businesses") or [])
    negated_scenes = set(slot_plan.get("negated_scenes") or [])
    negated_aliases = set(slot_plan.get("negated_aliases") or [])
    negated_tws = list(slot_plan.get("negated_time_windows") or [])

    row_changed = False
    for row in slot_rows:
        mid = row.get("memory_id") or ""
        v_sim = float(vector_sim_map.get(mid, 0.0))
        if v_sim <= 0.0 and row.get("embedding_json") and all_q_embs:
            try:
                r_emb = json.loads(row["embedding_json"])
                if r_emb:
                    v_sim = max((_cosine_similarity(w_e, r_emb) for w_e in all_q_embs), default=0.0)
            except (ValueError, TypeError):
                v_sim = 0.0

        if _extract_hit_row_slots(
            row=row,
            v_sim=v_sim,
            slot_rows=slot_rows,
            slot_plan=slot_plan,
            explicit_apps=explicit_apps,
            explicit_bizs=explicit_bizs,
            negated_apps=negated_apps,
            negated_bizs=negated_bizs,
            negated_scenes=negated_scenes,
            negated_aliases=negated_aliases,
            negated_tws=negated_tws,
        ):
            row_changed = True

    return _merge_projected_slot_candidates(
        slot_rows=slot_rows,
        slot_plan=slot_plan,
        all_q_embs=all_q_embs,
        biz_prototype_vecs=biz_prototype_vecs,
        expand_apps_fn=expand_apps_fn,
        explicit_apps=explicit_apps,
        explicit_bizs=explicit_bizs,
        negated_apps=negated_apps,
        negated_bizs=negated_bizs,
        negated_tws=negated_tws,
        row_changed=row_changed,
    )


def format_search_result_item(
    filestore: FileStore,
    graph: Optional[GraphStore],
    ns: Namespace,
    node: MemoryNode,
    score_info: Dict[str, Any],
    levels: str = "L0+L1",
    follow_edges: int = 0,
) -> Dict[str, Any]:
    """Format a scored MemoryNode into the tiered L0/L1/full explainable response dictionary."""
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
        result["overview"] = node.overview
        result["content_md"] = filestore.read_content_md(ns, node)

    if follow_edges > 0 and graph is not None:
        result["neighbors"] = graph.neighbors(ns, node.memory_id, max_hops=follow_edges)

    return result


def evaluate_node_Trigger_boost(
    node: MemoryNode,
    mid: str,
    base_sim: float,
    q_low: str,
    vector_sim_map: Dict[str, float],
    structured_reasons: Dict[str, str],
) -> float:
    """Boost similarity when query matches searchable text or experience card trigger_conditions."""
    sim = float(base_sim)
    if not q_low:
        return sim
    searchable = build_search_text(node).lower()
    if q_low in searchable:
        sim = max(sim, 0.85)
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
                if mid not in structured_reasons:
                    structured_reasons[mid] = (
                        f"经验触发条件匹配: keywords={matched_kws} (v_sim={vector_sim_map.get(mid, 0.0):.2f})"
                    )
    return sim


__all__ = [
    "_extract_hit_row_slots",
    "_merge_projected_slot_candidates",
    "evaluate_node_Trigger_boost",
    "format_search_result_item",
    "project_slots_from_vector_hits",
]
