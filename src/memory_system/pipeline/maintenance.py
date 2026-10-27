from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..core.models import MemoryNode, MemoryStatus, now_utc_str
from ..core.namespace import Namespace
from ..core.registry import MemoryRegistry
from ..storage.filestore import FileStore
from ..storage.index import MemoryIndex
from .updater import (
    compute_decayed_heat,
    compute_ebbinghaus_retention,
    normalize_ts_str,
    parse_ts_epoch,
)


class MaintenanceManager:
    """Periodic maintenance jobs:
    1. candidate_timeout: expire unreinforced candidates after candidate_timeout_days
    2. decay: update decayed heat / archive stale temporary nodes
    3. eviction: card exile method S(m) = w1*C + w2*R + w3*H + w4*U - w5*N
    4. timeline_compress: summarize node histories into timelines/{subtype}/timeline.json & timeline.md
    """

    def __init__(
        self,
        registry: MemoryRegistry,
        filestore: FileStore,
        index: Optional[MemoryIndex] = None,
        heat_half_life_hours: float = 168.0,
        candidate_timeout_days: int = 90,
        eviction_max_active: int = 500,
        eviction_score_threshold: float = 0.15,
    ) -> None:
        self.registry = registry
        self.filestore = filestore
        self.index = index
        self.heat_half_life_hours = float(heat_half_life_hours)
        self.candidate_timeout_days = int(candidate_timeout_days)
        self.eviction_max_active = int(eviction_max_active)
        self.eviction_score_threshold = float(eviction_score_threshold)

    def run_all(self, ns: Namespace, now_ts: Optional[str] = None) -> Dict[str, Any]:
        ts = now_ts or now_utc_str()
        timed_out = self.run_candidate_timeout(ns, now_ts=ts)
        decayed = self.run_decay(ns, now_ts=ts)
        evicted = self.run_eviction(ns, now_ts=ts)
        timelines = self.run_timeline_compress(ns, now_ts=ts)
        return {
            "candidate_timed_out": len(timed_out),
            "decayed_updated": len(decayed),
            "evicted": len(evicted),
            "timelines_compressed": len(timelines),
        }

    def run_candidate_timeout(self, ns: Namespace, now_ts: Optional[str] = None) -> List[str]:
        ts = now_ts or now_utc_str()
        t_now = parse_ts_epoch(ts)
        if t_now is None:
            return []

        expired_ids: List[str] = []
        for node in list(self.filestore.iter_nodes(ns)):
            if node.status != MemoryStatus.CANDIDATE.value:
                continue
            t_last = parse_ts_epoch(node.stats.last_reinforced_at or node.updated_at or node.created_at)
            if t_last is None:
                continue
            age_days = (t_now - t_last) / 86400.0
            if age_days >= self.candidate_timeout_days:
                node.status = MemoryStatus.ARCHIVED.value
                node.updated_at = ts
                self.filestore.save_node(
                    ns,
                    node,
                    history_entry={"ts": ts, "action": "candidate_timeout", "status": node.status},
                )
                if self.index is not None:
                    self.index.upsert(ns, node)
                expired_ids.append(node.memory_id)
        return expired_ids

    def apply_ebbinghaus_lifecycle_decay(
        self,
        ns: Namespace,
        now_ts: Optional[str] = None,
        candidate_expire_days: float = 14.0,
        candidate_min_retention: float = 0.75,
        reinforced_degrade_days: float = 90.0,
        reinforced_min_retention: float = 0.25,
    ) -> Dict[str, List[str]]:
        """Bidirectional Ebbinghaus lifecycle decay (§2.4):
        1. Unreinforced `candidate` nodes past `candidate_expire_days` (or R(t,0) < candidate_min_retention)
           automatically transition to `expired` (eliminating one-off implicit noise).
        2. Long-unused `reinforced` nodes past `reinforced_degrade_days` with R(t,n) < reinforced_min_retention
           automatically degrade back to `active`.
        """
        ts = normalize_ts_str(now_ts)
        t_now = parse_ts_epoch(ts)
        if t_now is None:
            return {"expired_candidates": [], "degraded_reinforced": []}

        expired_candidates: List[str] = []
        degraded_reinforced: List[str] = []

        for node in list(self.filestore.iter_nodes(ns)):
            if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value):
                continue
            t_last = parse_ts_epoch(node.stats.last_reinforced_at or node.updated_at or node.created_at)
            if t_last is None:
                continue
            age_days = (t_now - t_last) / 86400.0
            if age_days <= 0.0:
                continue

            spec = self.registry.get(node.subtype)
            base_lam = spec.decay_lambda if spec is not None else 0.02
            retention, stability_factor, eff_lam = compute_ebbinghaus_retention(
                decay_lambda=base_lam,
                last_reinforced_at=node.stats.last_reinforced_at or node.updated_at or node.created_at,
                now_ts=ts,
                positive_votes=node.stats.positive_votes,
                reuse_count=node.stats.reuse_count,
            )

            if node.status == MemoryStatus.CANDIDATE.value:
                if age_days >= candidate_expire_days or (base_lam > 0 and retention < candidate_min_retention):
                    prev_st = node.status
                    node.status = MemoryStatus.EXPIRED.value
                    node.updated_at = ts
                    self.filestore.save_node(
                        ns,
                        node,
                        history_entry={
                            "ts": ts,
                            "action": "ebbinghaus_candidate_expired",
                            "prev_status": prev_st,
                            "status": node.status,
                            "age_days": round(age_days, 2),
                            "retention": round(retention, 4),
                        },
                    )
                    if self.index is not None:
                        self.index.upsert(ns, node)
                    expired_candidates.append(node.memory_id)

            elif node.status == MemoryStatus.REINFORCED.value:
                if age_days >= reinforced_degrade_days and retention < reinforced_min_retention:
                    prev_st = node.status
                    node.status = MemoryStatus.ACTIVE.value
                    node.updated_at = ts
                    self.filestore.save_node(
                        ns,
                        node,
                        history_entry={
                            "ts": ts,
                            "action": "ebbinghaus_reinforced_degraded",
                            "prev_status": prev_st,
                            "status": node.status,
                            "age_days": round(age_days, 2),
                            "retention": round(retention, 4),
                            "stability_factor": round(stability_factor, 4),
                        },
                    )
                    if self.index is not None:
                        self.index.upsert(ns, node)
                    degraded_reinforced.append(node.memory_id)

        return {
            "expired_candidates": expired_candidates,
            "degraded_reinforced": degraded_reinforced,
        }

    def run_decay(self, ns: Namespace, now_ts: Optional[str] = None) -> List[str]:
        ts = normalize_ts_str(now_ts)
        updated_ids: List[str] = []
        for node in list(self.filestore.iter_nodes(ns)):
            if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value):
                continue
            new_heat = compute_decayed_heat(
                node.stats.heat,
                node.stats.last_heat_at,
                now_ts=ts,
                half_life_hours=self.heat_half_life_hours,
            )
            if abs(new_heat - node.stats.heat) > 1e-6:
                node.stats.heat = new_heat
                node.stats.last_heat_at = ts
                self.filestore.save_node(ns, node)
                if self.index is not None:
                    self.index.upsert(ns, node)
                updated_ids.append(node.memory_id)
        return updated_ids

    def compute_card_value(self, node: MemoryNode, now_ts: Optional[str] = None) -> float:
        """Card exile value formula from design doc:
        S(m) = w1*C_m + w2*R_m + w3*H_m + w4*U_m - w5*N_m
        """
        ts = normalize_ts_str(now_ts)
        spec = self.registry.get(node.subtype)
        lam = spec.decay_lambda if spec is not None else 0.02

        c_m = float(node.confidence)
        r_m, _, _ = compute_ebbinghaus_retention(
            decay_lambda=lam,
            last_reinforced_at=node.stats.last_reinforced_at or node.updated_at or node.created_at,
            now_ts=ts,
            positive_votes=node.stats.positive_votes,
            reuse_count=node.stats.reuse_count,
        )
        h_m = min(1.0, compute_decayed_heat(node.stats.heat, node.stats.last_heat_at, now_ts=ts, half_life_hours=self.heat_half_life_hours))
        u_m = min(1.0, node.stats.reuse_count / 5.0)
        n_m = min(1.0, node.stats.negative_votes / 3.0)

        return 0.35 * c_m + 0.25 * r_m + 0.20 * h_m + 0.20 * u_m - 0.25 * n_m

    def run_eviction(self, ns: Namespace, now_ts: Optional[str] = None) -> List[str]:
        """Exile lowest-value cards when count exceeds eviction_max_active.
        Never auto-exiles ACTIVE preferences or DISPUTED nodes unless over capacity.
        """
        ts = now_ts or now_utc_str()
        nodes = [
            n
            for n in self.filestore.iter_nodes(ns)
            if n.status not in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value)
        ]
        if len(nodes) <= self.eviction_max_active:
            return []

        # Sort eligible nodes (prefer exiling candidate/superseded/temporary first, then lowest card value)
        def _sort_key(n: MemoryNode) -> tuple[int, float]:
            protected = 1 if n.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value, MemoryStatus.DISPUTED.value) and n.validity.type != "temporary" else 0
            return (protected, self.compute_card_value(n, now_ts=ts))

        nodes_sorted = sorted(nodes, key=_sort_key)
        to_evict_count = max(0, len(nodes) - self.eviction_max_active)
        evicted_ids: List[str] = []

        for n in nodes_sorted[:to_evict_count]:
            n.status = MemoryStatus.ARCHIVED.value
            n.updated_at = ts
            self.filestore.save_node(
                ns,
                n,
                history_entry={"ts": ts, "action": "eviction", "status": n.status},
            )
            if self.index is not None:
                self.index.upsert(ns, n)
            evicted_ids.append(n.memory_id)
        return evicted_ids

    def run_timeline_compress(self, ns: Namespace, now_ts: Optional[str] = None) -> List[str]:
        """Generate per-subtype timeline summaries under users/{user_id}/timelines/{subtype}/."""
        ts = now_ts or now_utc_str()
        by_subtype: Dict[str, List[MemoryNode]] = {}
        for node in self.filestore.iter_nodes(ns):
            by_subtype.setdefault(node.subtype, []).append(node)

        compressed_subtypes: List[str] = []
        base_dir = self.filestore.data_root / "users" / ns.user_id / "timelines"
        for subtype, s_nodes in by_subtype.items():
            if not s_nodes:
                continue
            tdir = base_dir / subtype
            tdir.mkdir(parents=True, exist_ok=True)
            entries = []
            md_lines = [f"# Timeline Summary: {subtype}", ""]
            for n in sorted(s_nodes, key=lambda x: x.updated_at):
                hist = self.filestore.read_history(ns, n)
                entries.append(
                    {
                        "memory_id": n.memory_id,
                        "memory_key": n.memory_key,
                        "status": n.status,
                        "version": n.version,
                        "updated_at": n.updated_at,
                        "description": n.description,
                        "history_count": len(hist),
                    }
                )
                md_lines.append(f"- **{n.memory_key}** (`{n.status}`, v{n.version}): {n.description}")
            summary_obj = {
                "user_id": ns.user_id,
                "subtype": subtype,
                "compressed_at": ts,
                "node_count": len(entries),
                "entries": entries,
            }
            self.filestore.atomic_write_json(tdir / "timeline.json", summary_obj)
            self.filestore.atomic_write_text(tdir / "timeline.md", "\n".join(md_lines) + "\n")
            compressed_subtypes.append(subtype)
        return compressed_subtypes
