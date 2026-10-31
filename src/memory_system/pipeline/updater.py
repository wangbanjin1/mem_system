from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, Optional, Tuple

from ..core.logging_utils import JsonLinesLogger
from ..core.memory_key import build_memory_key
from ..core.models import (
    Edge,
    EdgeType,
    Event,
    EventType,
    MemoryNode,
    MemoryStatus,
    Stats,
    ValidityInfo,
    now_utc_str,
)
from ..core.namespace import Namespace
from ..core.registry import MemoryRegistry, MemoryTypeSpec
from ..providers.embedding import EmbeddingProvider
from ..storage.filestore import FileStore
from ..storage.graph import GraphStore
from ..storage.index import MemoryIndex, build_pure_phrase_text, build_search_text
from .sanitizer import CaseSanitizer


# Event Type -> (vote_delta, action, heat_delta) table (§2.2)
_VOTE: Dict[str, Tuple[int, str, float]] = {
    EventType.PREFERENCE_SET.value: (+1, "set", +0.2),
    EventType.EXPRESSION_ALIAS.value: (+1, "set", +0.2),
    EventType.CASE_RECORD.value: (+1, "set", +0.2),
    EventType.TRAJECTORY_RECORD.value: (+1, "set", +0.2),
    EventType.POLICY_RECORD.value: (+1, "set", +0.2),
    EventType.PREFERENCE_CONFIRM.value: (+1, "reinforce", +0.3),
    EventType.PREFERENCE_CORRECT.value: (+1, "correct", -0.3),
    EventType.PREFERENCE_CLEAR.value: (-1, "clear", -0.3),
    EventType.WORKING_RECORD.value: (+1, "set", +0.1),
    EventType.WORKING_RECORD_REUSE.value: (+1, "reinforce", +0.3),
    EventType.WORKING_RECORD_RESCHEDULE.value: (+1, "reinforce", +0.3),
    EventType.EXPERIENCE_APPLY.value: (+1, "reinforce", +0.3),
    EventType.EXPERIENCE_CRYSTALLIZE.value: (+1, "set", +0.2),
}


_CN_TS_RE = re.compile(
    r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[时:](\d{1,2})(?:分)?(?:(\d{1,2})秒?)?"
)


def normalize_ts_str(ts_str: Optional[str]) -> str:
    """Normalize Chinese timestamp ('2026年10月10日12时15分') or ISO string into 'YYYY-MM-DDTHH:MM:SSZ'."""
    if not ts_str or not isinstance(ts_str, str) or not ts_str.strip():
        return now_utc_str()
    s = ts_str.strip()
    m = _CN_TS_RE.search(s)
    if m:
        yr, mo, dy, hh, mm, ss = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
            int(m.group(6) or 0),
        )
        return f"{yr:04d}-{mo:02d}-{dy:02d}T{hh:02d}:{mm:02d}:{ss:02d}Z"
    if len(s) >= 19 and s[10] == " ":
        s = s[:10] + "T" + s[11:19] + "Z"
    elif len(s) == 19 and "T" in s and not s.endswith("Z"):
        s = s + "Z"
    return s


def parse_ts_epoch(ts_str: str) -> Optional[float]:
    """Parse YYYY-MM-DDTHH:MM:SSZ (or Chinese timestamp) using localtime mktime (§4 Pitfall 8).
    Never raises on malformed input; returns None on failure.
    """
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        norm = normalize_ts_str(ts_str)
        core = norm.strip()[:19]
        tm = time.strptime(core, "%Y-%m-%dT%H:%M:%S")
        return float(time.mktime(tm))
    except Exception:
        return None


def compute_ebbinghaus_retention(
    decay_lambda: float,
    last_reinforced_at: str,
    now_ts: Optional[str] = None,
    positive_votes: int = 1,
    reuse_count: int = 0,
    kappa: float = 0.8,
) -> Tuple[float, float, float]:
    """Compute Ebbinghaus spaced-repetition retention R(t, n) (§2.4):
      n = max(0, positive_votes - 1) + reuse_count
      S(n) = 1.0 + kappa * ln(1 + n)
      lambda_eff = decay_lambda / S(n)
      R(t, n) = exp(-lambda_eff * delta_days)
    Returns (retention, stability_factor, effective_lambda).
    """
    n_reinf = max(0, int(positive_votes) - 1) + max(0, int(reuse_count))
    stability_factor = 1.0 + float(kappa) * math.log(1.0 + n_reinf)
    base_lam = max(0.0, float(decay_lambda))
    eff_lam = base_lam / stability_factor if stability_factor > 0 else base_lam
    if eff_lam <= 0.0:
        return (1.0, stability_factor, 0.0)
    t_last = parse_ts_epoch(last_reinforced_at)
    t_now = parse_ts_epoch(now_ts or now_utc_str())
    if t_last is None or t_now is None:
        return (1.0, stability_factor, eff_lam)
    delta_days = (t_now - t_last) / 86400.0
    if delta_days <= 0.0:
        return (1.0, stability_factor, eff_lam)
    retention = math.exp(-eff_lam * delta_days)
    return (retention, stability_factor, eff_lam)


def compute_decayed_heat(
    heat: float,
    last_heat_at: str,
    now_ts: Optional[str] = None,
    half_life_hours: float = 168.0,
) -> float:
    """Compute effective_heat = heat * 0.5^(delta_hours / half_life_hours).
    Per §4 Pitfall 8: if clock skew (delta <= 0) or invalid timestamp, do not decay.
    """
    if heat <= 0.0 or half_life_hours <= 0.0:
        return max(0.0, float(heat))
    t_last = parse_ts_epoch(last_heat_at)
    t_now = parse_ts_epoch(now_ts or now_utc_str())
    if t_last is None or t_now is None:
        return max(0.0, float(heat))
    delta_hours = (t_now - t_last) / 3600.0
    if delta_hours <= 0.0:
        return max(0.0, float(heat))
    return float(heat) * (0.5 ** (delta_hours / float(half_life_hours)))


def _format_description(spec: MemoryTypeSpec, content: Dict[str, Any], memory_key: str) -> str:
    note = content.get("note") or content.get("summary") or content.get("title")
    base = ""
    if spec.description_template:
        try:
            safe_map = {k: (v if v is not None else "_") for k, v in content.items()}
            for f in spec.memory_key_fields:
                safe_map.setdefault(f, "_")
            base = spec.description_template.format_map(safe_map)
        except Exception:
            base = f"{spec.subtype}: {memory_key}"
    else:
        base = f"{spec.subtype}: {memory_key}"

    # Include key parameters in description so L0 reflects current content accurately (§4 Pitfall 4)
    extras = []
    for k in ("resolution", "priority", "latency", "time_window", "target", "value"):
        val = content.get(k)
        if val not in (None, ""):
            extras.append(f"{k}={val}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    if note:
        return f"{base}{suffix} - {note}"
    return f"{base}{suffix}"


def _format_overview(node: MemoryNode) -> str:
    kv = [f"{k}: {v}" for k, v in sorted(node.content.items()) if v not in (None, "", [], {})]
    return "; ".join(kv)


class MemoryUpdater:
    """Deterministic Python state machine for creating, merging, voting, transitioning,
    and activating memory nodes without LLM involvement.
    """

    def __init__(
        self,
        registry: MemoryRegistry,
        filestore: FileStore,
        index: Optional[MemoryIndex] = None,
        graph: Optional[GraphStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        logger: Optional[JsonLinesLogger] = None,
        heat_half_life_hours: float = 168.0,
    ) -> None:
        self.registry = registry
        self.filestore = filestore
        self.index = index
        self.graph = graph
        self.embedding_provider = embedding_provider
        self.logger = logger or JsonLinesLogger()
        self.heat_half_life_hours = float(heat_half_life_hours)

    def _find_by_key(
        self,
        ns: Namespace,
        spec: MemoryTypeSpec,
        memory_key: str,
    ) -> Optional[MemoryNode]:
        return self.filestore.find_by_key(ns, spec.memory_type, spec.subtype, memory_key)

    def apply_event(self, ns: Namespace, event: Event) -> MemoryNode:
        spec = self.registry.require(event.subtype)
        # Ensure key fields are populated from memory_key_fields or content
        combined_key_source = {**event.content, **event.memory_key_fields}
        memory_key = build_memory_key(spec, combined_key_source)

        lock = self.filestore.get_key_lock(ns, spec.subtype, memory_key)
        with lock:
            existing = self._find_by_key(ns, spec, memory_key)
            if existing is None:
                node = self._create(ns, spec, memory_key, event)
            else:
                node = self._update(ns, spec, existing, event)

        return node

    def _create(
        self,
        ns: Namespace,
        spec: MemoryTypeSpec,
        memory_key: str,
        event: Event,
    ) -> MemoryNode:
        now = event.event_time or now_utc_str()
        vote_delta, action, heat_delta = _VOTE.get(
            event.event_type, (+1, "set", +0.2)
        )
        content_data = {**event.memory_key_fields, **event.content}
        if ns.owner_type == "agent":
            content_data = CaseSanitizer.sanitize(content_data)

        val_type = str(event.validity or content_data.get("validity") or "long_term")
        pos_votes = max(0, vote_delta)
        neg_votes = max(0, -vote_delta)
        stats = Stats(
            votes=vote_delta,
            positive_votes=pos_votes,
            negative_votes=neg_votes,
            heat=max(0.0, heat_delta),
            last_heat_at=now,
            last_reinforced_at=now,
            reuse_count=1 if action == "reinforce" else 0,
        )
        node = MemoryNode(
            memory_type=spec.memory_type,
            subtype=spec.subtype,
            memory_key=memory_key,
            owner_type=ns.owner_type,
            user_id=ns.user_id,
            agent_id=ns.agent_id,
            tenant_id=ns.tenant_id,
            content=content_data,
            description=_format_description(spec, content_data, memory_key),
            status=MemoryStatus.CANDIDATE.value,
            confidence=float(event.confidence or 0.85),
            version=1,
            validity=ValidityInfo(type=val_type, valid_from=now),
            stats=stats,
            evidence=[event.event_id] if event.event_id else [],
            created_at=now,
            updated_at=now,
        )
        node.overview = _format_overview(node)
        if content_data.get("declaration_mode") == "implicit":
            effective_threshold = max(2, spec.vote_threshold)
        elif content_data.get("declaration_mode") == "explicit":
            effective_threshold = 1
        else:
            effective_threshold = spec.vote_threshold
        node.status = self._transition(
            current_status=MemoryStatus.CANDIDATE.value,
            votes=node.stats.votes,
            vote_threshold=effective_threshold,
            action=action,
            is_concession=event.is_concession,
            was_existing=False,
        )

        history_entry = {
            "ts": now,
            "action": "create",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "status": node.status,
            "votes": node.stats.votes,
            "heat": round(node.stats.heat, 6),
            "version": node.version,
        }
        self._persist_node_and_index(ns, node, history_entry, event)
        return node

    def _update(
        self,
        ns: Namespace,
        spec: MemoryTypeSpec,
        node: MemoryNode,
        event: Event,
    ) -> MemoryNode:
        # 1. Variant pollution defense (§4 Pitfall 2, updater.py:280-286):
        # Temporary event MUST NOT overwrite or vote on a non-temporary node!
        ev_validity = str(event.validity or event.content.get("validity") or "long_term")
        if ev_validity == "temporary" and node.validity.type != "temporary":
            self.logger.log(
                "event_ignored_temporary",
                user_id=ns.user_id,
                subtype=spec.subtype,
                memory_key=node.memory_key,
                memory_id=node.memory_id,
                event_id=event.event_id,
            )
            return node

        now = event.event_time or now_utc_str()
        vote_delta, action, heat_delta = _VOTE.get(
            event.event_type, (+1, "set", +0.2)
        )

        incoming_content = {**event.memory_key_fields, **event.content}
        if ns.owner_type == "agent":
            incoming_content = CaseSanitizer.sanitize(incoming_content)

        # 2. Check content changes BEFORE merging so description (L0) never freezes (§4 Pitfall 4)
        old_content_snapshot = dict(node.content)
        has_new_note_or_change = False
        for k, v in incoming_content.items():
            if v not in (None, "", [], {}) and old_content_snapshot.get(k) != v:
                has_new_note_or_change = True
                break

        # 3. Merge non-empty values into node.content
        # Protect already-active/reinforced nodes from having their core QoS/routing fields
        # overwritten by an implicit single-task event (unless it is an explicit PREFERENCE_CORRECT).
        is_implicit_non_corr = (
            node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value)
            and incoming_content.get("declaration_mode") == "implicit"
            and event.event_type != EventType.PREFERENCE_CORRECT.value
        )
        protected_fields = {"resolution", "latency", "time_window", "condition", "value"}
        for k, v in incoming_content.items():
            if v not in (None, "", [], {}):
                if is_implicit_non_corr and k in protected_fields and node.content.get(k) not in (None, ""):
                    continue
                node.content[k] = v

        if has_new_note_or_change or not node.description:
            node.description = _format_description(spec, node.content, node.memory_key)
        node.overview = _format_overview(node)

        # 4. Accumulate votes
        node.stats.votes += vote_delta
        if vote_delta > 0:
            node.stats.positive_votes += vote_delta
            node.stats.last_reinforced_at = now
        elif vote_delta < 0:
            node.stats.negative_votes += abs(vote_delta)
        if action == "reinforce":
            node.stats.reuse_count += 1

        # 5. State transition
        prev_status = node.status
        node.status = self._transition(
            current_status=prev_status,
            votes=node.stats.votes,
            vote_threshold=spec.vote_threshold,
            action=action,
            is_concession=event.is_concession,
            was_existing=True,
        )

        # 6. Heat decay + signal superposition
        self._apply_heat(node, heat_delta, now)

        # 7. Update metadata, version, confidence, evidence
        node.version += 1
        node.updated_at = now
        if action in ("set", "reinforce"):
            node.confidence = min(0.99, max(node.confidence, float(event.confidence or 0.85)) + 0.02)
        elif action == "correct" and not event.is_concession:
            node.confidence = max(0.1, node.confidence - 0.2)

        if event.event_id and event.event_id not in node.evidence:
            node.evidence.append(event.event_id)

        history_entry = {
            "ts": now,
            "action": action,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "prev_status": prev_status,
            "status": node.status,
            "votes": node.stats.votes,
            "heat": round(node.stats.heat, 6),
            "version": node.version,
        }
        self._persist_node_and_index(ns, node, history_entry, event)
        return node

    @staticmethod
    def _transition(
        current_status: str,
        votes: int,
        vote_threshold: int,
        action: str,
        is_concession: bool = False,
        was_existing: bool = True,
    ) -> str:
        """Pure state transition function (§2.2 updater.py:139-167)."""
        if action == "clear":
            return MemoryStatus.ARCHIVED.value

        if action == "correct":
            if is_concession:
                return (
                    MemoryStatus.ACTIVE.value
                    if votes >= vote_threshold
                    else MemoryStatus.CANDIDATE.value
                )
            return MemoryStatus.DISPUTED.value

        # action in ("set", "reinforce")
        if current_status in (
            MemoryStatus.ARCHIVED.value,
            MemoryStatus.EXPIRED.value,
            MemoryStatus.DISPUTED.value,
            MemoryStatus.SUPERSEDED.value,
        ):
            # Re-assertion revives node
            return (
                MemoryStatus.ACTIVE.value
                if votes >= max(1, vote_threshold)
                else MemoryStatus.CANDIDATE.value
            )

        if current_status == MemoryStatus.CANDIDATE.value:
            if votes >= vote_threshold:
                return MemoryStatus.ACTIVE.value
            return MemoryStatus.CANDIDATE.value

        if current_status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
            if was_existing and (action == "reinforce" or votes > vote_threshold):
                return MemoryStatus.REINFORCED.value
            return MemoryStatus.ACTIVE.value

        return current_status

    def _apply_heat(self, node: MemoryNode, heat_delta: float, now_ts: str) -> None:
        decayed = compute_decayed_heat(
            node.stats.heat,
            node.stats.last_heat_at,
            now_ts=now_ts,
            half_life_hours=self.heat_half_life_hours,
        )
        node.stats.heat = max(0.0, decayed + float(heat_delta))
        node.stats.last_heat_at = now_ts

    def bump_heat(
        self,
        ns: Namespace,
        node: MemoryNode,
        delta: float = 0.05,
        now_ts: Optional[str] = None,
    ) -> MemoryNode:
        """Bump heat on retrieval hit (§2.3): ONLY modifies heat & last_heat_at,
        does NOT alter votes, version, or status.
        """
        lock = self.filestore.get_key_lock(ns, node.subtype, node.memory_key)
        with lock:
            fresh = self.filestore.find_by_key(ns, node.memory_type, node.subtype, node.memory_key)
            target = fresh if fresh is not None else node
            ts = now_ts or now_utc_str()
            self._apply_heat(target, delta, ts)
            self.filestore.save_node(
                ns,
                target,
                history_entry={
                    "ts": ts,
                    "action": "bump_heat",
                    "delta": delta,
                    "heat": round(target.stats.heat, 6),
                    "status": target.status,
                    "votes": target.stats.votes,
                },
            )
            if self.index is not None:
                self.index.upsert(ns, target, embedding=None)
            node.stats.heat = target.stats.heat
            node.stats.last_heat_at = target.stats.last_heat_at
            return target

    def _persist_node_and_index(
        self,
        ns: Namespace,
        node: MemoryNode,
        history_entry: Dict[str, Any],
        event: Optional[Event] = None,
    ) -> None:
        self.filestore.save_node(ns, node, history_entry=history_entry)
        if self.index is not None:
            emb = None
            if self.embedding_provider is not None:
                try:
                    phrase_txt = build_pure_phrase_text(node) or build_search_text(node)
                    emb = self.embedding_provider.embed_one(phrase_txt)
                except Exception:
                    emb = None
            self.index.upsert(ns, node, embedding=emb)

        if self.graph is not None and event is not None and event.event_id:
            edge_type = EdgeType.SUPPORTS.value
            if event.event_type in (
                EventType.PREFERENCE_CONFIRM.value,
                EventType.WORKING_RECORD_REUSE.value,
                EventType.EXPERIENCE_APPLY.value,
            ):
                edge_type = EdgeType.REINFORCES.value
            elif event.event_type == EventType.PREFERENCE_CORRECT.value:
                edge_type = EdgeType.CONTRADICTS.value
            self.graph.add_edge(
                Edge(
                    source_id=event.event_id,
                    target_id=node.memory_id,
                    edge_type=edge_type,
                    user_id=ns.user_id,
                    owner_type=ns.owner_type,
                    agent_id=ns.agent_id,
                )
            )
