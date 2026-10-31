"""Working memory ledger handler and agent experience recording helper."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .core.memory_key import build_memory_key
from .core.models import Event, EventType, MemoryNode, MemoryStatus
from .core.namespace import Namespace, agent_namespace, user_namespace
from .core.registry import MemoryRegistry
from .core.working_rules import classify_working_record_reference
from .pipeline.sanitizer import CaseSanitizer
from .pipeline.updater import MemoryUpdater, normalize_ts_str
from .storage.filestore import FileStore


def record_agent_experience(
    registry: MemoryRegistry,
    updater: MemoryUpdater,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    subtype: str,
    content: Dict[str, Any],
    memory_key_fields: Optional[Dict[str, Any]] = None,
    confidence: float = 0.9,
) -> MemoryNode:
    """Record agent experience (case / trajectory / policy) isolated per user_id (§1.3)
    and sanitized via CaseSanitizer (§4 Pitfall 11).
    """
    ns = agent_namespace(user_id=user_id, agent_id=agent_id, tenant_id=tenant_id)
    spec = registry.require(subtype)
    sanitized_content = CaseSanitizer.sanitize(content)
    key_fields = dict(memory_key_fields or {})
    for kf in spec.memory_key_fields:
        if kf not in key_fields and kf in sanitized_content:
            key_fields[kf] = sanitized_content[kf]
    ev_type_map = {
        "case": EventType.CASE_RECORD.value,
        "trajectory": EventType.TRAJECTORY_RECORD.value,
        "policy": EventType.POLICY_RECORD.value,
    }
    ev = Event(
        event_type=ev_type_map.get(subtype, EventType.CASE_RECORD.value),
        subtype=subtype,
        memory_key_fields=key_fields,
        content=sanitized_content,
        validity="long_term",
        confidence=confidence,
        owner_type="agent",
        user_id=user_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
    )
    return updater.apply_event(ns, ev)


def ingest_working_records(
    registry: MemoryRegistry,
    filestore: FileStore,
    updater: MemoryUpdater,
    tenant_id: str,
    user_id: str,
    records: Sequence[Dict[str, Any]],
) -> List[MemoryNode]:
    """Batch ingest working memory records without LLM (§2.7).
    Idempotent by record_id: re-ingesting an existing record_id triggers WORKING_RECORD_REUSE (reinforce)
    instead of creating duplicate nodes.
    """
    ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
    spec = registry.require("working_record")
    results: List[MemoryNode] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_id = str(rec.get("record_id") or rec.get("id") or "").strip()
        if not rec_id:
            continue
        content = dict(rec)
        content["record_id"] = rec_id
        mkey = build_memory_key(spec, {"record_id": rec_id})
        existing = filestore.find_by_key(ns, spec.memory_type, spec.subtype, mkey)
        ev_type = (
            EventType.WORKING_RECORD_REUSE.value
            if existing is not None
            else EventType.WORKING_RECORD.value
        )
        ev_ts = normalize_ts_str(str(rec.get("event_time") or rec.get("created_at") or ""))
        ev_validity = str(rec.get("validity") or "long_term")
        ev = Event(
            event_type=ev_type,
            subtype="working_record",
            memory_key_fields={"record_id": rec_id},
            content=content,
            validity=ev_validity,
            confidence=float(rec.get("confidence", 0.95)),
            user_id=user_id,
            tenant_id=tenant_id,
            event_time=ev_ts,
        )
        node = updater.apply_event(ns, ev)
        results.append(node)

    return results


def append_session_working_record_if_present(
    ns: Namespace,
    metadata: Optional[Dict[str, Any]],
    session_ts: str,
    ingest_records_fn: Callable[[str, str, Sequence[Dict[str, Any]]], List[MemoryNode]],
) -> None:
    """Append completed session working_record if passed in metadata (skipping single-time temporary overrides)."""
    meta = metadata or {}
    wr_to_append = meta.get("working_record_to_append")
    if (
        isinstance(wr_to_append, dict)
        and wr_to_append.get("record_id")
        and not wr_to_append.get("is_temporary_override")
        and wr_to_append.get("validity") != "temporary"
    ):
        rec_copy = dict(wr_to_append)
        rec_copy.setdefault("event_time", session_ts)
        ingest_records_fn(ns.tenant_id, ns.user_id, [rec_copy])


def process_working_memory_turn(
    ns: Namespace,
    filestore: FileStore,
    ingest_records_fn: Callable[[str, str, Sequence[Dict[str, Any]]], List[MemoryNode]],
    u1_text: str,
    session_ts: str,
    raw_ref_ts: str,
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Handle working memory reuse, reschedule, or disambiguation against the user's ledger."""
    events_out: List[Dict[str, Any]] = []
    node_ids_out: List[str] = []

    wr_act = classify_working_record_reference(u1_text)
    if wr_act is None:
        return False, events_out, node_ids_out

    wr_nodes = [
        n
        for n in filestore.iter_nodes(ns, memory_type="working_memory", subtype="working_record")
        if n.status not in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value)
    ]
    if wr_nodes:
        wr_nodes.sort(key=lambda x: str(x.content.get("record_id") or x.created_at or ""))
        is_resched = wr_act == "reschedule"
        loc_hits = [
            n
            for n in wr_nodes
            if str(n.content.get("location") or "").strip()
            and str(n.content.get("location")).strip() in u1_text
        ]
        pool = loc_hits or wr_nodes
        target_wr = pool[-1] if is_resched else pool[0]
        rec_data = dict(target_wr.content)
        rec_data["event_time"] = session_ts
        if is_resched:
            rec_data["status"] = "rescheduled"
            if len(raw_ref_ts) >= 11:
                date_prefix = raw_ref_ts[:11]
                st_old = str(rec_data.get("start_timestamp") or "")
                et_old = str(rec_data.get("end_timestamp") or "")
                if len(st_old) >= 11:
                    rec_data["start_timestamp"] = date_prefix + st_old[11:]
                if len(et_old) >= 11:
                    rec_data["end_timestamp"] = date_prefix + et_old[11:]
        nodes = ingest_records_fn(ns.tenant_id, ns.user_id, [rec_data])
        for n in nodes:
            node_ids_out.append(n.memory_id)
            events_out.append(
                {
                    "event_type": (
                        EventType.WORKING_RECORD_RESCHEDULE.value
                        if is_resched
                        else EventType.WORKING_RECORD_REUSE.value
                    ),
                    "subtype": "working_record",
                    "memory_key_fields": {"record_id": str(rec_data.get("record_id") or "")},
                    "validity": "long_term",
                    "content": rec_data,
                }
            )
    return True, events_out, node_ids_out
