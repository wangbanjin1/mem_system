"""Facade and single public entry point for the Agent Memory System."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core.config import Config, load_config
from .core.logging_utils import JsonLinesLogger
from .core.models import Event, MemoryNode
from .core.namespace import Namespace, agent_namespace, user_namespace
from .core.registry import MemoryRegistry, load_registry
from .core.session_ingestor import (
    _TRANSIENT_SCENE_MARKERS,
    align_scene_with_known,
    decompose_to_orthogonal_events,
    extract_clean_venue,
    run_llm_extract_and_induce_with_events,
)
from .experience_handler import (
    _BUILTIN_EXPERIENCE_TEMPLATES,
    _STRATEGY_TO_DOMAIN_MAP,
    _ref_year_of,
    ingest_experience_cards as _ingest_experience_cards_helper,
    process_experience_turn,
)
from .pipeline.archiver import SessionArchiver
from .pipeline.extractor import Extractor
from .pipeline.fastpath import _CONCESSION_RE, _PARAM_SIGNAL_RE, FastPathProcessor
from .pipeline.induction import InductionEngine
from .pipeline.maintenance import MaintenanceManager
from .pipeline.queue import FileTaskQueue
from .pipeline.updater import MemoryUpdater, normalize_ts_str
from .providers.embedding import EmbeddingProvider
from .providers.llm import LLMProvider
from .retrieval.router import QueryRouter
from .retrieval.search import SearchEngine
from .storage.filestore import FileStore
from .storage.graph import GraphStore
from .storage.index import MemoryIndex, build_pure_phrase_text, build_search_text
from .working_memory_handler import (
    append_session_working_record_if_present,
    ingest_working_records as _ingest_working_records_helper,
    process_working_memory_turn,
    record_agent_experience as _record_agent_experience_helper,
)

__all__ = [
    "MemorySystem",
    "_BUILTIN_EXPERIENCE_TEMPLATES",
    "_STRATEGY_TO_DOMAIN_MAP",
    "_TRANSIENT_SCENE_MARKERS",
    "_ref_year_of",
]


class MemorySystem:
    """Facade and single public entry point for the Agent Memory System (api.py)."""

    def __init__(
        self,
        config: Optional[Config | Dict[str, Any] | str | Path] = None,
        data_root: Optional[str | Path] = None,
        registry_dir: Optional[str | Path] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        router_llm_provider: Optional[LLMProvider] = None,
        async_mode: Optional[bool] = None,
    ) -> None:
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = load_config(overrides=config)
        else:
            self.config = load_config(config_path=config)

        if data_root is not None:
            self.config.raw["data_root"] = str(data_root)
        if async_mode is not None:
            self.config.raw.setdefault("queue", {})["async_mode"] = bool(async_mode)

        self.data_root: Path = self.config.data_root
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.logger = JsonLinesLogger(self.data_root / "logs" / "memory_system.jsonl")
        self.registry: MemoryRegistry = load_registry(registry_dir)
        self.filestore = FileStore(self.data_root)
        self.index = MemoryIndex(self.data_root / "index" / "memory.db")
        self.graph = GraphStore(self.data_root)

        self.embedding_provider = embedding_provider or EmbeddingProvider.from_config(self.config.embedding)
        self.llm_provider = llm_provider or LLMProvider.from_config(self.config.llm)
        self.router_llm_provider = router_llm_provider or LLMProvider.from_config(self.config.llm_router)

        half_life = float(self.config.maintenance.get("heat_half_life_hours", 168.0))
        self.archiver = SessionArchiver(self.filestore)
        self.fastpath = FastPathProcessor(self.config.repo_root / "data")
        self.extractor = Extractor(
            registry=self.registry,
            llm_provider=self.llm_provider,
            data_dir=self.config.repo_root / "data",
            prompt_path=self.config.repo_root / "prompts" / "extract_generic.md",
            logger=self.logger,
        )
        self.updater = MemoryUpdater(
            registry=self.registry,
            filestore=self.filestore,
            index=self.index,
            graph=self.graph,
            embedding_provider=self.embedding_provider,
            logger=self.logger,
            heat_half_life_hours=half_life,
        )
        self.induction = InductionEngine(registry=self.registry, filestore=self.filestore, updater=self.updater)
        self.maintenance = MaintenanceManager(
            registry=self.registry,
            filestore=self.filestore,
            index=self.index,
            heat_half_life_hours=half_life,
            candidate_timeout_days=int(self.config.maintenance.get("candidate_timeout_days", 90)),
            eviction_max_active=int(self.config.maintenance.get("eviction_max_active", 500)),
            eviction_score_threshold=float(self.config.maintenance.get("eviction_score_threshold", 0.15)),
        )

        ret_cfg = self.config.retrieval
        self.search_engine = SearchEngine(
            registry=self.registry,
            filestore=self.filestore,
            index=self.index,
            embedding_provider=self.embedding_provider,
            graph=self.graph,
            heat_callback=lambda ns, node, delta: self.updater.bump_heat(ns, node, delta=delta),
            alpha=float(ret_cfg.get("alpha", 0.5)),
            beta=float(ret_cfg.get("beta", 0.2)),
            gamma=float(ret_cfg.get("gamma", 0.1)),
            delta=float(ret_cfg.get("delta", 0.2)),
            temporary_weight=float(ret_cfg.get("temporary_weight", 0.5)),
            heat_half_life_hours=half_life,
            heat_bump_on_hit=float(ret_cfg.get("heat_bump_on_hit", 0.05)),
        )
        self.router = QueryRouter(filestore=self.filestore, llm_provider=self.router_llm_provider)

        q_cfg = self.config.queue
        self.queue = FileTaskQueue(
            data_root=self.data_root,
            handler=self._handle_async_llm_task,
            max_retries=int(q_cfg.get("max_retries", 3)),
            poll_interval_sec=float(q_cfg.get("poll_interval_sec", 0.2)),
            logger=self.logger,
        )
        if q_cfg.get("async_mode", False):
            self.queue.start_worker()

    def record_session(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        turns: List[Dict[str, Any]],
        async_mode: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full write pipeline sequence (§1.2): decay -> archive -> cognitive tier -> fastpath -> LLM -> induction."""
        ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
        numbered_turns = self.extractor._ensure_turn_numbers(turns)

        raw_ref_ts = (
            reference_time
            or (metadata or {}).get("reference_time")
            or (numbered_turns[0].get("time") if numbered_turns else None)
        )
        session_ts = normalize_ts_str(str(raw_ref_ts)) if raw_ref_ts else normalize_ts_str(None)
        decay_summary = self.maintenance.apply_ebbinghaus_lifecycle_decay(ns, now_ts=session_ts)

        self.archiver.archive_session(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            turns=numbered_turns,
            metadata=metadata,
        )

        is_non_pref_session, cog_events, cog_node_ids = self._process_cognitive_tier_events(
            ns=ns,
            session_id=session_id,
            turns=numbered_turns,
            metadata=metadata,
            session_ts=session_ts,
            raw_ref_ts=str(raw_ref_ts or ""),
        )
        if is_non_pref_session:
            self.induction.run_all(ns)
            return {
                "task_id": f"TASK_{uuid.uuid4().hex[:12].upper()}",
                "status": "completed",
                "fastpath_nodes": cog_node_ids,
                "extracted_events": cog_events,
                "ebbinghaus_decay": decay_summary,
            }

        fp_result = self.fastpath.process(
            session_turns=numbered_turns, session_id=session_id, user_id=user_id, tenant_id=tenant_id
        )
        user_text_all = " ".join(
            str(t.get("content") or t.get("text") or "")
            for t in numbered_turns
            if str(t.get("role", "user")).lower() == "user"
        )
        has_param_signal = fp_result.has_param_signal or bool(_PARAM_SIGNAL_RE.search(user_text_all))
        has_concession = fp_result.has_concession or bool(_CONCESSION_RE.search(user_text_all))
        should_run_llm = (not fp_result.is_noise) and (
            fp_result.needs_llm or has_param_signal or has_concession
        )

        applied_fastpath_ids: List[str] = []
        applied_fastpath_events: List[Dict[str, Any]] = []
        deferred_fastpath_events: List[Event] = []
        for ev in fp_result.confirmed:
            ev.event_time = session_ts
            if not should_run_llm:
                node = self._apply_candidate(ns, ev)
                applied_fastpath_ids.append(node.memory_id)
                applied_fastpath_events.append(ev.to_dict())
            else:
                deferred_fastpath_events.append(ev)

        use_async = bool(self.config.queue.get("async_mode", False)) if async_mode is None else bool(async_mode)
        task_id = f"TASK_{uuid.uuid4().hex[:12].upper()}"
        extracted_events_out: List[Dict[str, Any]] = list(applied_fastpath_events)

        if should_run_llm:
            payload = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "turns": numbered_turns,
                "session_ts": session_ts,
                "has_concession": has_concession,
                "fastpath_events": [e.to_dict() for e in deferred_fastpath_events],
            }
            if use_async:
                self.queue.enqueue(payload, task_id=task_id)
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "fastpath_nodes": applied_fastpath_ids,
                    "ebbinghaus_decay": decay_summary,
                }
            _, llm_events = self._run_llm_extract_and_induce_with_events(payload)
            extracted_events_out.extend(llm_events)
        else:
            self.induction.run_all(ns)

        return {
            "task_id": task_id,
            "status": "completed",
            "fastpath_nodes": applied_fastpath_ids,
            "extracted_events": extracted_events_out,
            "ebbinghaus_decay": decay_summary,
        }

    def _handle_async_llm_task(self, payload: Dict[str, Any]) -> None:
        self._run_llm_extract_and_induce(payload)

    def _process_cognitive_tier_events(
        self,
        ns: Namespace,
        session_id: str,
        turns: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        session_ts: str,
        raw_ref_ts: str,
    ) -> tuple[bool, List[Dict[str, Any]], List[str]]:
        """Handle working_memory and experiential_memory events deterministically."""
        append_session_working_record_if_present(
            ns=ns, metadata=metadata, session_ts=session_ts, ingest_records_fn=self.ingest_working_records
        )
        user_turns_text = [
            str(t.get("content") or t.get("text") or "")
            for t in turns
            if str(t.get("role", "user")).lower() == "user"
        ]
        u1_text = user_turns_text[0] if user_turns_text else ""
        u_later_text = " ".join(user_turns_text[1:]) if len(user_turns_text) > 1 else ""

        matched_exp, exp_events, exp_node_ids = process_experience_turn(
            ns=ns,
            filestore=self.filestore,
            extractor=self.extractor,
            ingest_cards_fn=self.ingest_experience_cards,
            session_id=session_id,
            turns=turns,
            u1_text=u1_text,
            u_later_text=u_later_text,
            session_ts=session_ts,
            raw_ref_ts=raw_ref_ts,
        )
        if matched_exp:
            return True, exp_events, exp_node_ids

        matched_wr, wr_events, wr_node_ids = process_working_memory_turn(
            ns=ns,
            filestore=self.filestore,
            ingest_records_fn=self.ingest_working_records,
            u1_text=u1_text,
            session_ts=session_ts,
            raw_ref_ts=raw_ref_ts,
        )
        if matched_wr:
            return True, wr_events, wr_node_ids

        return False, [], []

    def _run_llm_extract_and_induce(self, payload: Dict[str, Any]) -> List[MemoryNode]:
        nodes, _ = self._run_llm_extract_and_induce_with_events(payload)
        return nodes

    def _run_llm_extract_and_induce_with_events(
        self, payload: Dict[str, Any]
    ) -> tuple[List[MemoryNode], List[Dict[str, Any]]]:
        return run_llm_extract_and_induce_with_events(
            extractor=self.extractor,
            filestore=self.filestore,
            induction=self.induction,
            decompose_fn=self._decompose_to_orthogonal_events,
            apply_llm_event_fn=self._apply_llm_event,
            payload=payload,
        )

    def _decompose_to_orthogonal_events(
        self, ns: Namespace, ev: Event, turns: List[Dict[str, Any]]
    ) -> List[Event]:
        return decompose_to_orthogonal_events(
            filestore=self.filestore,
            index=self.index,
            apply_llm_event_fn=self._apply_llm_event,
            ns=ns,
            ev=ev,
            turns=turns,
        )

    @staticmethod
    def _extract_clean_venue(user_text: str) -> str:
        return extract_clean_venue(user_text)

    def _align_scene_with_known(self, ns: Namespace, scene: str, user_text: str) -> str:
        return align_scene_with_known(self.filestore, ns, scene, user_text)

    def _apply_candidate(self, ns: Namespace, event: Event) -> MemoryNode:
        return self.updater.apply_event(ns, event)

    def _apply_llm_event(self, ns: Namespace, event: Event) -> MemoryNode:
        return self.updater.apply_event(ns, event)

    def record_agent_experience(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        subtype: str,
        content: Dict[str, Any],
        memory_key_fields: Optional[Dict[str, Any]] = None,
        confidence: float = 0.9,
    ) -> MemoryNode:
        """Record agent experience (case / trajectory / policy) isolated per user_id (§1.3)."""
        return _record_agent_experience_helper(
            registry=self.registry,
            updater=self.updater,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            subtype=subtype,
            content=content,
            memory_key_fields=memory_key_fields,
            confidence=confidence,
        )

    def ingest_working_records(
        self, tenant_id: str, user_id: str, records: Sequence[Dict[str, Any]]
    ) -> List[MemoryNode]:
        """Batch ingest working memory records without LLM (§2.7)."""
        return _ingest_working_records_helper(
            registry=self.registry,
            filestore=self.filestore,
            updater=self.updater,
            tenant_id=tenant_id,
            user_id=user_id,
            records=records,
        )

    def ingest_experience_cards(
        self, tenant_id: str, user_id: str, cards: Sequence[Dict[str, Any]]
    ) -> List[MemoryNode]:
        """Batch ingest experience memory cards without LLM (§2.7)."""
        return _ingest_experience_cards_helper(
            registry=self.registry,
            filestore=self.filestore,
            updater=self.updater,
            tenant_id=tenant_id,
            user_id=user_id,
            cards=cards,
        )

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        top_k: int = 10,
        levels: str = "L0+L1",
        subtypes: Optional[Sequence[str]] = None,
        memory_types: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        owner_type: str = "user",
        agent_id: str = "",
        follow_edges: int = 0,
        use_router: bool = False,
        bump_heat: bool = True,
        now_ts: Optional[str] = None,
        query_slots: Optional[Dict[str, Any]] = None,
        cascade_short_circuit: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search user or agent memories using hybrid retrieval."""
        ns = (
            agent_namespace(user_id=user_id, agent_id=agent_id, tenant_id=tenant_id)
            if owner_type == "agent"
            else user_namespace(user_id=user_id, tenant_id=tenant_id)
        )
        effective_query = query
        effective_mtypes = list(memory_types) if memory_types else None

        if use_router and owner_type == "user":
            route_info = self.router.route(ns, query)
            effective_query = route_info.get("expanded_query") or query
            if effective_mtypes is None and route_info.get("memory_types"):
                effective_mtypes = route_info["memory_types"]

        return self.search_engine.search(
            ns=ns,
            query=effective_query,
            top_k=top_k,
            levels=levels,
            subtypes=subtypes,
            memory_types=effective_mtypes,
            statuses=statuses,
            follow_edges=follow_edges,
            bump_heat=bump_heat,
            now_ts=now_ts,
            query_slots=query_slots,
            cascade_short_circuit=cascade_short_circuit,
        )

    def get_node(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        owner_type: str = "user",
        agent_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a single MemoryNode by memory_id along with its markdown body and event history."""
        ns = (
            agent_namespace(user_id=user_id, agent_id=agent_id, tenant_id=tenant_id)
            if owner_type == "agent"
            else user_namespace(user_id=user_id, tenant_id=tenant_id)
        )
        node = self.filestore.find_by_id(ns, memory_id)
        if node is None:
            return None
        res = node.to_dict()
        res["content_md"] = self.filestore.read_content_md(ns, node)
        res["history"] = self.filestore.read_history(ns, node)
        return res

    def rebuild_index(self) -> int:
        """Rebuild the entire SQLite vector + FTS5 index from authoritative files on disk."""
        self.index.clear_all()
        count = 0
        for ns, node in self.filestore.iter_all_namespaces_and_nodes():
            emb = None
            if self.embedding_provider is not None:
                try:
                    phrase_txt = build_pure_phrase_text(node) or build_search_text(node)
                    emb = self.embedding_provider.embed_one(phrase_txt)
                except Exception:
                    emb = None
            self.index.upsert(ns, node, embedding=emb)
            count += 1
        return count

    def run_maintenance(
        self,
        tenant_id: str = "default",
        user_id: Optional[str] = None,
        now_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run periodic maintenance (candidate timeout, Ebbinghaus decay, capacity eviction, timeline compression)."""
        if user_id:
            ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
            return self.maintenance.run_all(ns, now_ts=now_ts)

        totals = {"candidate_timed_out": 0, "decayed_updated": 0, "evicted": 0, "timelines_compressed": 0}
        seen_users = set()
        for ns, _ in self.filestore.iter_all_namespaces_and_nodes():
            if ns.owner_type == "user" and ns.user_id not in seen_users:
                seen_users.add(ns.user_id)
                r = self.maintenance.run_all(ns, now_ts=now_ts)
                for k in totals:
                    totals[k] += int(r.get(k, 0))
        return totals

    def close(self) -> None:
        """Stop background task worker and close SQLite index connection."""
        self.queue.stop_worker()
        self.index.close()
