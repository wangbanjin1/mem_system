"""Two-stage relational table completion via `service_to_app` and `periodic_preference`."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..storage.index import time_matches_window
from .query_segmentation import (
    _BIZ_KEYWORDS,
    _QUERY_CN_PT_RE,
    _QUERY_CN_TIME_RE,
    _QUERY_TIME_RE,
    _build_sliding_windows,
    _fuzzy_phrase_match,
    _parse_relative_date_window,
    _split_positive_and_negated_clauses,
)


def _resolve_service_to_app_hop(
    slot_rows: Sequence[Dict[str, Any]],
    apps: List[str],
    bizs: List[str],
    pointer_hops: List[str],
    negated_apps: Sequence[str] | Set[str],
    negated_bizs: Sequence[str] | Set[str],
) -> None:
    """Resolve bidirectional service_to_app relational hops between business and application."""
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


def _resolve_periodic_preference_hop(
    slot_rows: Sequence[Dict[str, Any]],
    apps: List[str],
    bizs: List[str],
    time_windows: List[str],
    pointer_hops: List[str],
    explicit_apps: Sequence[str] | Set[str],
    explicit_bizs: Sequence[str] | Set[str],
    negated_apps: Sequence[str] | Set[str],
    negated_bizs: Sequence[str] | Set[str],
    negated_time_windows: Sequence[str],
    include_reverse_periodic: bool = True,
) -> None:
    """Resolve forward and reverse periodic_preference relational hops between time_window and app/business."""
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
                    hop_str = f"periodic_preference[{cond_tw}] -> {p_app}|{p_biz}"
                    if hop_str not in pointer_hops:
                        pointer_hops.append(hop_str)

    if include_reverse_periodic and apps and bizs:
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
                hop_str = f"periodic_preference[{p_app}|{p_biz}] -> {cond_tw}"
                if hop_str not in pointer_hops:
                    pointer_hops.append(hop_str)


def expand_relational_pointers(
    slot_rows: Sequence[Dict[str, Any]],
    apps: List[str],
    bizs: List[str],
    time_windows: List[str],
    pointer_hops: List[str],
    explicit_apps: Sequence[str] | Set[str],
    explicit_bizs: Sequence[str] | Set[str],
    negated_apps: Sequence[str] | Set[str],
    negated_bizs: Sequence[str] | Set[str],
    negated_time_windows: Sequence[str],
    include_reverse_periodic: bool = True,
) -> None:
    """Execute stage-2 relational table completion (`service_to_app` & `periodic_preference` bidirectional links)."""
    _resolve_service_to_app_hop(slot_rows, apps, bizs, pointer_hops, negated_apps, negated_bizs)
    _resolve_periodic_preference_hop(
        slot_rows=slot_rows,
        apps=apps,
        bizs=bizs,
        time_windows=time_windows,
        pointer_hops=pointer_hops,
        explicit_apps=explicit_apps,
        explicit_bizs=explicit_bizs,
        negated_apps=negated_apps,
        negated_bizs=negated_bizs,
        negated_time_windows=negated_time_windows,
        include_reverse_periodic=include_reverse_periodic,
    )


def check_periodic_time_conflict(
    slot_rows: Sequence[Dict[str, Any]],
    tw_matches: Sequence[str],
    explicit_apps: Set[str],
    explicit_bizs: Set[str],
    negated_apps: Set[str],
    negated_bizs: Set[str],
) -> bool:
    """Return True if a candidate time_window's periodic_preference conflicts with explicit or negated apps/bizs."""
    if not tw_matches or not (explicit_apps or explicit_bizs or negated_apps or negated_bizs):
        return False
    for prow in slot_rows:
        if prow.get("subtype") != "periodic_preference":
            continue
        cond_tw = str(prow.get("time_window_col") or "").strip()
        if cond_tw and any(time_matches_window(tw_m, cond_tw) for tw_m in tw_matches):
            p_app = str(prow.get("app_col") or "").strip()
            p_biz = str(prow.get("biz_col") or "").strip()
            if (
                (explicit_apps and p_app and p_app not in explicit_apps)
                or (explicit_bizs and p_biz and p_biz not in explicit_bizs)
                or (p_app in negated_apps)
                or (p_biz in negated_bizs)
            ):
                return True
    return False


def _collect_seed_slots_from_hits(
    slot_rows: Sequence[Dict[str, Any]],
    sorted_app_aliases: List[Tuple[str, str]],
    expand_apps_fn: Callable[[Sequence[str]], List[str]],
    pos_text: str,
    neg_text: str,
    query_slots: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stage 1A: Collect explicit/negated app & business seeds and match personalized expression rows."""
    pos_low = pos_text.lower()
    neg_low = neg_text.lower()
    seed: Dict[str, Any] = {
        "resolved_aliases": {},
        "alias_hit_ids": {},
        "apps": [],
        "bizs": [],
        "scenes": [],
        "time_windows": [],
        "negated_apps": [],
        "negated_bizs": [],
        "negated_scenes": [],
        "negated_aliases": [],
        "negated_time_windows": [],
    }

    if isinstance(query_slots, dict):
        for src_k, dst_k in (
            ("applications", "apps"),
            ("businesses", "bizs"),
            ("scenes", "scenes"),
            ("time_windows", "time_windows"),
            ("negated_applications", "negated_apps"),
            ("negated_businesses", "negated_bizs"),
            ("negated_scenes", "negated_scenes"),
        ):
            for val in query_slots.get(src_k) or []:
                s_val = str(val).strip()
                if s_val and s_val not in seed[dst_k]:
                    seed[dst_k].append(s_val)

    if neg_low:
        for alias_low, canonical_app in sorted_app_aliases:
            if alias_low and alias_low in neg_low and canonical_app not in seed["negated_apps"]:
                seed["negated_apps"].append(canonical_app)
        for kw, canonical_biz in _BIZ_KEYWORDS:
            if kw in neg_low and canonical_biz not in seed["negated_bizs"]:
                seed["negated_bizs"].append(canonical_biz)

    explicit_apps = list(seed["apps"])
    for alias_low, canonical_app in sorted_app_aliases:
        if alias_low and alias_low in pos_low and canonical_app not in explicit_apps:
            explicit_apps.append(canonical_app)
    seed["explicit_apps"] = expand_apps_fn(explicit_apps)
    seed["negated_apps"] = expand_apps_fn(seed["negated_apps"])

    explicit_bizs = list(seed["bizs"])
    for kw, canonical_biz in _BIZ_KEYWORDS:
        if kw in pos_low and canonical_biz not in explicit_bizs:
            explicit_bizs.append(canonical_biz)
    seed["explicit_bizs"] = explicit_bizs

    _match_expression_rows_for_seed(slot_rows, pos_text, neg_text, seed)
    return seed


def _match_expression_rows_for_seed(
    slot_rows: Sequence[Dict[str, Any]],
    pos_text: str,
    neg_text: str,
    seed: Dict[str, Any],
) -> None:
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

        if neg_text and _fuzzy_phrase_match(alias, neg_text) and not _fuzzy_phrase_match(alias, pos_text):
            if alias not in seed["negated_aliases"]:
                seed["negated_aliases"].append(alias)
            for m_tw in _QUERY_TIME_RE.finditer(target):
                tw_neg = m_tw.group(1)
                if tw_neg not in seed["negated_time_windows"]:
                    seed["negated_time_windows"].append(tw_neg)
            if bound_app and bound_app not in seed["explicit_apps"] and bound_app not in seed["negated_apps"]:
                seed["negated_apps"].append(bound_app)
            if bound_biz and bound_biz not in seed["explicit_bizs"] and bound_biz not in seed["negated_bizs"]:
                seed["negated_bizs"].append(bound_biz)
            continue

        if _fuzzy_phrase_match(alias, pos_text):
            if seed["explicit_apps"] and bound_app and bound_app not in seed["explicit_apps"]:
                continue
            if seed["explicit_bizs"] and bound_biz and bound_biz not in seed["explicit_bizs"]:
                continue
            seed["resolved_aliases"][alias] = target
            scope_str = f" (作用域: {bound_app}|{bound_biz})" if bound_app or bound_biz else ""
            match_mode = "精确命中" if alias in pos_text else "近似语义命中"
            seed["alias_hit_ids"][row["memory_id"]] = f"{match_mode}个性化词典: {alias} -> {target}{scope_str}"
            if bound_app and bound_app not in seed["apps"]:
                seed["apps"].append(bound_app)
            if bound_biz and bound_biz not in seed["bizs"]:
                seed["bizs"].append(bound_biz)
            if bound_scene and bound_scene not in seed["scenes"]:
                seed["scenes"].append(bound_scene)


def _resolve_venue_and_qos_backfill(
    slot_rows: Sequence[Dict[str, Any]],
    sorted_app_aliases: List[Tuple[str, str]],
    expand_apps_fn: Callable[[Sequence[str]], List[str]],
    combined_pos_text: str,
    neg_text: str,
    now_ts: Optional[str],
    seed: Dict[str, Any],
) -> None:
    """Stage 1B: Extract time windows, physical venue/scene slots, and reference timestamp fallback."""
    combined_pos_low = combined_pos_text.lower()
    apps: List[str] = seed["apps"]
    bizs: List[str] = seed["bizs"]
    scenes: List[str] = seed["scenes"]
    time_windows: List[str] = seed["time_windows"]

    for alias_low, canonical_app in sorted_app_aliases:
        if alias_low and alias_low in combined_pos_low and canonical_app not in apps:
            apps.append(canonical_app)
    seed["apps"] = expand_apps_fn(apps)
    apps = seed["apps"]

    for kw, canonical_biz in _BIZ_KEYWORDS:
        if kw in combined_pos_low and canonical_biz not in bizs:
            bizs.append(canonical_biz)

    for m in _QUERY_TIME_RE.finditer(combined_pos_text):
        if m.group(1) not in time_windows:
            time_windows.append(m.group(1))
    for m in _QUERY_CN_TIME_RE.finditer(combined_pos_text):
        h1, m1, h2, m2 = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4) or 0)
        tw = f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
        if tw not in time_windows:
            time_windows.append(tw)
    for m in _QUERY_CN_PT_RE.finditer(combined_pos_text):
        hh, mm = int(m.group(1)), (30 if "半" in m.group(0) else int(m.group(2) or 0))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            pt = f"{hh:02d}:{mm:02d}"
            if pt not in time_windows and not any(pt in w for w in time_windows):
                time_windows.append(pt)

    for row in slot_rows:
        sc = str(row.get("scene_col") or "").strip()
        if not sc:
            continue
        if neg_text and _fuzzy_phrase_match(sc, neg_text) and not _fuzzy_phrase_match(sc, combined_pos_text):
            if sc not in seed["negated_scenes"]:
                seed["negated_scenes"].append(sc)
            continue
        if _fuzzy_phrase_match(sc, combined_pos_text) and sc not in scenes:
            scenes.append(sc)
            if row.get("subtype") == "scene_preference":
                s_app = str(row.get("app_col") or "").strip()
                s_biz = str(row.get("biz_col") or "").strip()
                if s_app and s_app not in apps and (not seed["explicit_apps"] or s_app in seed["explicit_apps"]):
                    if s_app not in seed["negated_apps"]:
                        apps.append(s_app)
                if s_biz and s_biz not in bizs and (not seed["explicit_bizs"] or s_biz in seed["explicit_bizs"]):
                    if s_biz not in seed["negated_bizs"]:
                        bizs.append(s_biz)

    if not time_windows and now_ts:
        m_ts = re.search(r"T(\d{2}:\d{2})", str(now_ts))
        m_ts_cn = re.search(r"(\d{1,2})时(\d{1,2})分", str(now_ts)) if not m_ts else None
        hh_mm = (
            m_ts.group(1)
            if m_ts
            else (f"{int(m_ts_cn.group(1)):02d}:{int(m_ts_cn.group(2)):02d}" if m_ts_cn else "")
        )
        if hh_mm and hh_mm != "00:00" and not any(
            time_matches_window(hh_mm, ntw) for ntw in seed["negated_time_windows"]
        ):
            time_windows.append(hh_mm)


def mine_slots_and_resolve_two_hop(
    slot_rows: Sequence[Dict[str, Any]],
    app_alias_map: Dict[str, str],
    expand_apps_fn: Callable[[Sequence[str]], List[str]],
    query: str,
    now_ts: Optional[str] = None,
    query_slots: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Two-stage personalized expression resolution & structured slot mining with negation clause filtering."""
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
    sorted_app_aliases = sorted(app_alias_map.items(), key=lambda x: len(x[0]), reverse=True)

    seed = _collect_seed_slots_from_hits(
        slot_rows, sorted_app_aliases, expand_apps_fn, pos_text, neg_text, query_slots
    )
    combined_pos_text = (pos_text + " " + " ".join(seed["resolved_aliases"].values())).strip()

    _resolve_venue_and_qos_backfill(
        slot_rows, sorted_app_aliases, expand_apps_fn, combined_pos_text, neg_text, now_ts, seed
    )

    pointer_hops: List[str] = []
    expand_relational_pointers(
        slot_rows=slot_rows,
        apps=seed["apps"],
        bizs=seed["bizs"],
        time_windows=seed["time_windows"],
        pointer_hops=pointer_hops,
        explicit_apps=seed["explicit_apps"],
        explicit_bizs=seed["explicit_bizs"],
        negated_apps=seed["negated_apps"],
        negated_bizs=seed["negated_bizs"],
        negated_time_windows=seed["negated_time_windows"],
        include_reverse_periodic=True,
    )

    apps = expand_apps_fn(seed["apps"])
    negated_apps = expand_apps_fn(seed["negated_apps"])
    return {
        "resolved_aliases": seed["resolved_aliases"],
        "applications": apps,
        "businesses": seed["bizs"],
        "scenes": seed["scenes"],
        "time_windows": seed["time_windows"],
        "relative_date_range": _parse_relative_date_window(combined_pos_text, now_ts),
        "explicit_applications": seed["explicit_apps"],
        "explicit_businesses": seed["explicit_bizs"],
        "negated_applications": [a for a in negated_apps if a not in apps],
        "negated_businesses": [b for b in seed["negated_bizs"] if b not in seed["bizs"]],
        "negated_scenes": [s for s in seed["negated_scenes"] if s not in seed["scenes"]],
        "negated_aliases": seed["negated_aliases"],
        "negated_time_windows": seed["negated_time_windows"],
        "pointer_hops": pointer_hops,
        "vector_anchor_hits": [],
        "positive_clauses": pos_clauses,
        "sliding_windows": _build_sliding_windows(pos_clauses or ([q_raw] if q_raw else []), win_size=8, step=4),
        "expanded_query": combined_pos_text if combined_pos_text else q_raw,
        "alias_hit_ids": seed["alias_hit_ids"],
    }


__all__ = [
    "_collect_seed_slots_from_hits",
    "_resolve_periodic_preference_hop",
    "_resolve_service_to_app_hop",
    "_resolve_venue_and_qos_backfill",
    "check_periodic_time_conflict",
    "expand_relational_pointers",
    "mine_slots_and_resolve_two_hop",
]
