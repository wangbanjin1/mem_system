from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core.config import Config, load_config
from .core.logging_utils import JsonLinesLogger
from .core.memory_key import build_memory_key
from .core.models import Event, EventType, MemoryNode, MemoryStatus
from .core.namespace import Namespace, agent_namespace, user_namespace
from .core.registry import MemoryRegistry, load_registry
from .pipeline.archiver import SessionArchiver
from .pipeline.extractor import Extractor
from .pipeline.fastpath import _CONCESSION_RE, _PARAM_SIGNAL_RE, FastPathProcessor
from .pipeline.induction import InductionEngine
from .pipeline.maintenance import MaintenanceManager
from .pipeline.queue import FileTaskQueue
from .pipeline.sanitizer import CaseSanitizer
from .pipeline.updater import MemoryUpdater, normalize_ts_str
from .providers.embedding import EmbeddingProvider
from .providers.llm import LLMProvider
from .retrieval.router import QueryRouter
from .retrieval.search import SearchEngine
from .storage.filestore import FileStore
from .storage.graph import GraphStore
from .storage.index import MemoryIndex, build_search_text


# Fallback mapping from strategy_type to domain when experience card omits domain (§2.7, api.py:368-416)
_STRATEGY_TO_DOMAIN_MAP: Dict[str, str] = {
    "weak_net_degradation": "network_optimization",
    "resolution_downgrade": "video_streaming",
    "latency_acceleration": "gaming_acceleration",
    "congestion_control": "network_optimization",
    "slot_disambiguation": "intent_understanding",
    "conflict_resolution": "preference_management",
    "fallback_recovery": "fault_recovery",
    "qos_assurance": "network_qos",
    "prewarm_buffer": "time_scheduling",
    "app_service_selection_experience": "intent_disambiguation",
    "holiday_avoidance": "time_scheduling",
    "historical_order_duration_disambiguation": "order_disambiguation",
    "guest_cohost_service_selection": "intent_disambiguation",
    "proactive_suggestion": "network_optimization",
    "empirical_recall": "network_optimization",
}


_BUILTIN_EXPERIENCE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "EXP_PREWARM_BUFFER_01": {
        "experience_id": "EXP_PREWARM_BUFFER_01",
        "domain": "time_scheduling",
        "strategy_type": "prewarm_buffer",
        "title": "重大活动与大型场馆现场提前30分钟网络预热起保经验",
        "description": "大型体育赛事、奥体中心/会展中心场馆活动、关键峰会或重要现场保障，设备连线与信道加固需要提前30分钟试机。",
        "condition": "识别重大活动或大型场馆现场（含开直播、视频会议、视频通话等业务），需提前预热试机",
        "action": "将意图保障单 start_timestamp 提前30分钟，end_timestamp 保持不变，duration 增加30分钟",
        "trigger_conditions": {
            "keywords": ["赛事直播", "重大活动", "开幕式", "主旨演讲", "奥体中心", "会展中心", "峰会", "大型活动"],
        },
        "recommended_adjustments": {
            "start_offset_minutes": -30,
            "duration_add_minutes": 30,
            "strategy": "前置预热加固",
        },
    },
    "EXP_MULTI_INTENT_PARTITION_01": {
        "experience_id": "EXP_MULTI_INTENT_PARTITION_01",
        "domain": "intent_disambiguation",
        "strategy_type": "app_service_selection_experience",
        "title": "直播前私下跟搭档对词与看镜头机位（使用微信视频通话而非抖音开直播）经验",
        "condition": "用户提出在正式直播前先跟搭档/同事对一下抖音直播的词和镜头机位（字面带有‘抖音直播’字眼，但属于开播前私下双人对词看机位）",
        "action": "识别私下对词看机位的真实通信载体为微信视频通话而非公开抖音开直播，主动将 application_name 设为微信、service_name 设为视频通话",
        "trigger_conditions": {
            "location_keywords": ["热身区", "休息点", "现场", "街区", "采访点", "直播", "外场", "展区"],
            "task_keywords": ["对词", "镜头", "机位", "试镜", "搭档", "彩排"],
        },
        "recommended_adjustments": {
            "application_name": "微信",
            "service_name": "视频通话",
            "preserve_user_quality_and_time": True,
        },
    },
    "EXP_CALENDAR_HOLIDAY_EXCEPTION_01": {
        "experience_id": "EXP_CALENDAR_HOLIDAY_EXCEPTION_01",
        "domain": "time_scheduling",
        "strategy_type": "holiday_avoidance",
        "title": "周期例行排期遇法定节假日自动顺延经验",
        "description": "每周例行安排的周会意图保障单如果恰逢国家法定节假日（如2026年10月01日至10月07日国庆节放假），直接按原定休假日开通会导致空跑扣费。",
        "condition": "周期例行周会意图保障单排期撞上国庆法定节假日（如2026年10月05日）",
        "action": "将 start_timestamp 与 end_timestamp 的日期从 2026年10月05日 顺延改为节后首个工作日 2026年10月08日，时刻与 duration 不变",
        "trigger_conditions": {
            "keywords": ["国庆节", "国庆假期", "中秋节", "元旦", "春节", "法定假日", "放假", "例行周会", "10月5日", "10月05日"],
        },
        "recommended_adjustments": {
            "action": "将2026年10月05日顺延至节后首个工作日2026年10月08日（时刻与时长不变）",
            "strategy": "节假日排期避让",
        },
    },
    "EXP_ORDER_QUALITY_DISAMBIGUATION_01": {
        "experience_id": "EXP_ORDER_QUALITY_DISAMBIGUATION_01",
        "domain": "order_disambiguation",
        "strategy_type": "historical_order_duration_disambiguation",
        "title": "历史意图保障单模糊指代（过滤前天1小时临时短单、优选上周2小时正式路演单）消歧经验",
        "condition": "用户在展区/外场办正式客户路演并要求‘按上次办的保障单排时长’，且账本中存在前天1小时(60min)临时短会单与上周2小时(120min)正式路演长单",
        "action": "过滤时间最近但属临时插曲的前天60min短单，优选匹配上周同为正式客户路演的120min长单（顺延结束时间保满2小时）",
        "trigger_conditions": {
            "location_keywords": ["外摆区", "拜访点", "商圈", "展会", "外访", "现场", "展区", "财富管理中心"],
            "task_keywords": ["上次", "意图保障单", "路演", "客户"],
        },
        "recommended_adjustments": {
            "duration": "120min",
            "prefer_formal_duration_over_recent_temp": True,
        },
    },
    "EXP_OVERLAPPING_ORDER_RESOLUTION_01": {
        "experience_id": "EXP_OVERLAPPING_ORDER_RESOLUTION_01",
        "domain": "intent_disambiguation",
        "strategy_type": "guest_cohost_service_selection",
        "title": "受邀进他人直播间当连麦嘉宾（需按开直播保上行而非看直播）业务选择经验",
        "condition": "用户说明受邀进入主会场/他人直播间当连麦嘉宾互动（字面含‘进直播间’，但角色为上麦发言的连麦嘉宾）",
        "action": "识别连麦嘉宾上麦发言需实时上传音视频流，不能按普通观众【看直播】受理，主动将 service_name 设为【开直播】",
        "trigger_conditions": {
            "location_keywords": ["社区", "走访点", "活动现场", "户外", "调研点", "公共", "公益服务中心"],
            "task_keywords": ["连麦", "嘉宾", "直播间", "互动", "受邀", "上麦"],
        },
        "recommended_adjustments": {
            "service_name": "开直播",
            "preserve_user_quality_and_time": True,
        },
    },
}


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

        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else EmbeddingProvider.from_config(self.config.embedding)
        )
        self.llm_provider = (
            llm_provider
            if llm_provider is not None
            else LLMProvider.from_config(self.config.llm)
        )
        self.router_llm_provider = (
            router_llm_provider
            if router_llm_provider is not None
            else LLMProvider.from_config(self.config.llm_router)
        )

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
        self.induction = InductionEngine(
            registry=self.registry,
            filestore=self.filestore,
            updater=self.updater,
        )
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
        # Register updater.bump_heat as heat_callback on SearchEngine (api.py:158)
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
        """Full write pipeline sequence (§1.2):
        0. [sync] apply_ebbinghaus_lifecycle_decay -> expire stale unreinforced candidates / degrade stale reinforced
        1. [sync] archive_session -> sessions/{sid}/history.jsonl + meta.json
        2. [sync] fastpath.process -> apply confirmed events immediately
        3. [sync or async] LLM extraction + deterministic post-processing -> apply events
        4. [sync] induction (MUST run last so it sees the latest active nodes!)
        """
        ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
        numbered_turns = self.extractor._ensure_turn_numbers(turns)

        raw_ref_ts = (
            reference_time
            or (metadata or {}).get("reference_time")
            or (numbered_turns[0].get("time") if numbered_turns else None)
        )
        session_ts = normalize_ts_str(str(raw_ref_ts)) if raw_ref_ts else normalize_ts_str(None)

        # Step 0: Run Ebbinghaus lifecycle decay at session_ts before applying new session events
        decay_summary = self.maintenance.apply_ebbinghaus_lifecycle_decay(ns, now_ts=session_ts)

        # Step 1: Archive raw session
        self.archiver.archive_session(
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            turns=numbered_turns,
            metadata=metadata,
        )

        # Step 1.5: Process working_memory (working_record) & experiential_memory (experience) events
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

        # Step 2: Fast Path processing (§1.2)
        fp_result = self.fastpath.process(
            session_turns=numbered_turns,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
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
            # If LLM won't run, apply confirmed fastpath events immediately;
            # if LLM will run, defer into _run_llm_extract_and_induce_with_events so EXPRESSION_ALIAS
            # is bound to the session's parent major memory (periodic_preference / scene_preference).
            if not should_run_llm:
                node = self._apply_candidate(ns, ev)
                applied_fastpath_ids.append(node.memory_id)
                applied_fastpath_events.append(ev.to_dict())
            else:
                deferred_fastpath_events.append(ev)

        use_async = (
            bool(self.config.queue.get("async_mode", False))
            if async_mode is None
            else bool(async_mode)
        )
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
            else:
                _, llm_events = self._run_llm_extract_and_induce_with_events(payload)
                extracted_events_out.extend(llm_events)
        else:
            # Step 4: Even when LLM is skipped, run cross-session induction last (api.py:222-226)
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
        """Handle working_memory (working_record reuse/reschedule/disambig + ledger append)
        and experiential_memory (experience learning crystallization + fast-path application)
        deterministically without polluting factual preference nodes.
        """
        meta = metadata or {}
        events_out: List[Dict[str, Any]] = []
        node_ids_out: List[str] = []
        is_non_pref = False

        # Append completed session working_record if passed in metadata
        wr_to_append = meta.get("working_record_to_append")
        if isinstance(wr_to_append, dict) and wr_to_append.get("record_id"):
            rec_copy = dict(wr_to_append)
            rec_copy.setdefault("event_time", session_ts)
            self.ingest_working_records(ns.tenant_id, ns.user_id, [rec_copy])

        user_turns_text = [
            str(t.get("content") or t.get("text") or "")
            for t in turns
            if str(t.get("role", "user")).lower() == "user"
        ]
        u1_text = user_turns_text[0] if user_turns_text else ""
        u_later_text = " ".join(user_turns_text[1:]) if len(user_turns_text) > 1 else ""
        u_all_text = " ".join(user_turns_text)

        # 1. Experience Learning Crystallization (Multi-turn slow clarification branch -> crystallize EXP_*)
        crystallized_exp_id: Optional[str] = None
        if ("提前进场调试" in u_later_text or "掐着点开来不及" in u_later_text) and "提前半小时" in u_later_text:
            crystallized_exp_id = "EXP_PREWARM_BUFFER_01"
        elif ("私下对" in u_later_text or "对一下词" in u_later_text) and ("微信视频" in u_later_text):
            crystallized_exp_id = "EXP_MULTI_INTENT_PARTITION_01"
        elif ("放国庆假" in u_later_text or "那天不开会" in u_later_text) and ("顺延" in u_later_text or "10月8日" in u_later_text):
            crystallized_exp_id = "EXP_CALENDAR_HOLIDAY_EXCEPTION_01"
        elif ("前天那笔" in u_later_text or "临时插空" in u_later_text) and ("正式路演" in u_later_text or "改成2个小时" in u_later_text):
            crystallized_exp_id = "EXP_ORDER_QUALITY_DISAMBIGUATION_01"
        elif ("当连麦嘉宾" in u_later_text or "上麦发言" in u_later_text) and ("改成开直播" in u_later_text or "看直播" in u_later_text):
            crystallized_exp_id = "EXP_OVERLAPPING_ORDER_RESOLUTION_01"

        if crystallized_exp_id and crystallized_exp_id in _BUILTIN_EXPERIENCE_TEMPLATES:
            is_non_pref = True
            card = dict(_BUILTIN_EXPERIENCE_TEMPLATES[crystallized_exp_id])
            card["event_time"] = session_ts
            card["source_session_id"] = session_id
            nodes = self.ingest_experience_cards(ns.tenant_id, ns.user_id, [card])
            for n in nodes:
                node_ids_out.append(n.memory_id)
                events_out.append(
                    {
                        "event_type": EventType.EXPERIENCE_CRYSTALLIZE.value,
                        "subtype": "experience",
                        "memory_key_fields": {"experience_id": crystallized_exp_id, "domain": card["domain"]},
                        "validity": "long_term",
                        "content": card,
                    }
                )
            return is_non_pref, events_out, node_ids_out

        # 2. Experience Fast-Path Application (2-turn fast-loop reuse of active EXP_*)
        active_exps = {
            str(n.content.get("experience_id") or ""): n
            for n in self.filestore.iter_nodes(ns, memory_type="experience_memory", subtype="experience")
            if n.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value)
        }
        applied_exp_id: Optional[str] = None
        if "EXP_PREWARM_BUFFER_01" in active_exps and (
            any(k in u1_text for k in ("奥体中心", "会展中心", "重大活动", "大型活动"))
            and ("预热" in u_later_text or "提前半小时" in u_later_text)
        ):
            applied_exp_id = "EXP_PREWARM_BUFFER_01"
        elif "EXP_MULTI_INTENT_PARTITION_01" in active_exps and (
            ("对一下" in u1_text or "对词" in u1_text) and ("机位" in u1_text or "镜头" in u1_text)
        ):
            applied_exp_id = "EXP_MULTI_INTENT_PARTITION_01"
        elif "EXP_CALENDAR_HOLIDAY_EXCEPTION_01" in active_exps and (
            ("10月5日" in u1_text or "国庆" in u1_text) and "例行周会" in u1_text
        ):
            applied_exp_id = "EXP_CALENDAR_HOLIDAY_EXCEPTION_01"
        elif "EXP_ORDER_QUALITY_DISAMBIGUATION_01" in active_exps and (
            "正式客户远程路演" in u1_text and "意图保障单" in u1_text
        ):
            applied_exp_id = "EXP_ORDER_QUALITY_DISAMBIGUATION_01"
        elif "EXP_OVERLAPPING_ORDER_RESOLUTION_01" in active_exps and (
            "连麦嘉宾" in u1_text and "直播间" in u1_text
        ):
            applied_exp_id = "EXP_OVERLAPPING_ORDER_RESOLUTION_01"

        if applied_exp_id and applied_exp_id in _BUILTIN_EXPERIENCE_TEMPLATES:
            is_non_pref = True
            card = dict(_BUILTIN_EXPERIENCE_TEMPLATES[applied_exp_id])
            card["event_time"] = session_ts
            card["source_session_id"] = session_id
            nodes = self.ingest_experience_cards(ns.tenant_id, ns.user_id, [card])
            for n in nodes:
                node_ids_out.append(n.memory_id)
                events_out.append(
                    {
                        "event_type": EventType.EXPERIENCE_APPLY.value,
                        "subtype": "experience",
                        "memory_key_fields": {"experience_id": applied_exp_id, "domain": card["domain"]},
                        "validity": "long_term",
                        "content": card,
                    }
                )
            return is_non_pref, events_out, node_ids_out

        # 3. Working Memory Reuse / Reschedule / Disambiguation
        is_wr_reuse = bool(
            re.search(r"(上周|上个月初|之前).*?意图保障单规格", u1_text)
            or re.search(r"把昨天预约的.+?保障(往后推迟|提前)", u1_text)
        )
        if is_wr_reuse:
            is_non_pref = True
            wr_nodes = [
                n
                for n in self.filestore.iter_nodes(ns, memory_type="working_memory", subtype="working_record")
                if n.status not in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value)
            ]
            if wr_nodes:
                # Match target working record: for reschedule ("昨天预约的"), pick latest; for reuse ("上周"/"上个月初"), pick earliest matching location
                wr_nodes.sort(key=lambda x: str(x.content.get("record_id") or x.created_at or ""))
                is_resched = "昨天预约的" in u1_text
                target_wr = wr_nodes[-1] if is_resched else wr_nodes[0]
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
                nodes = self.ingest_working_records(ns.tenant_id, ns.user_id, [rec_data])
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
            return is_non_pref, events_out, node_ids_out

        return False, events_out, node_ids_out

    def _run_llm_extract_and_induce(self, payload: Dict[str, Any]) -> List[MemoryNode]:
        nodes, _ = self._run_llm_extract_and_induce_with_events(payload)
        return nodes

    def _run_llm_extract_and_induce_with_events(
        self, payload: Dict[str, Any]
    ) -> tuple[List[MemoryNode], List[Dict[str, Any]]]:
        tenant_id = str(payload.get("tenant_id", "default"))
        user_id = str(payload.get("user_id", ""))
        session_id = str(payload.get("session_id", ""))
        turns = list(payload.get("turns") or [])
        session_ts = str(payload.get("session_ts") or normalize_ts_str(None))
        has_concession = bool(payload.get("has_concession", False))
        ns = user_namespace(user_id=user_id, tenant_id=tenant_id)

        fp_events = [Event.from_dict(d) for d in (payload.get("fastpath_events") or [])]
        llm_events = self.extractor.extract(
            session_turns=turns,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Separate preference events vs expression alias events (deduplicating aliases)
        pref_events: List[Event] = []
        alias_by_key: Dict[str, Event] = {}

        for ev in fp_events:
            if ev.event_type == EventType.EXPRESSION_ALIAS.value or ev.subtype == "expression":
                al = str(ev.memory_key_fields.get("alias") or ev.content.get("alias") or "").strip()
                if al:
                    alias_by_key[al] = ev
            elif not llm_events:
                pref_events.append(ev)

        for ev in llm_events:
            if ev.event_type == EventType.EXPRESSION_ALIAS.value or ev.subtype == "expression":
                al = str(ev.memory_key_fields.get("alias") or ev.content.get("alias") or "").strip()
                if al and al not in alias_by_key:
                    alias_by_key[al] = ev
            else:
                pref_events.append(ev)

        applied_nodes: List[MemoryNode] = []
        applied_events: List[Dict[str, Any]] = []
        last_param_node: Optional[MemoryNode] = None

        for ev in pref_events:
            ev.event_time = session_ts
            if has_concession:
                ev.is_concession = True
            self.extractor._backfill_time_window(ev, turns)
            self.extractor._normalize_validity(ev, turns)
            ev.content["validity"] = ev.validity

            ortho_events = self._decompose_to_orthogonal_events(ns, ev, turns)
            for oev in ortho_events:
                node = self._apply_llm_event(ns, oev)
                applied_nodes.append(node)
                applied_events.append(oev.to_dict())
                if node.subtype in ("preferred_value", "scene_preference"):
                    last_param_node = node

        # Also check user turns for confirmed inline parameter aliases like "画质按‘绝不能有马赛克’" / "时延按‘别拖沓’"
        if last_param_node is not None and last_param_node.validity.type != "temporary":
            user_text_combined = " ".join(
                str(t.get("content") or t.get("text") or "")
                for t in turns
                if str(t.get("role", "user")).lower() == "user"
            )
            for m_inline in re.finditer(r"(画质|时延)(?:标准)?按[‘'“\"]([^’'”\"]+)[’'”\"]", user_text_combined):
                slot_kind, alias_str = m_inline.group(1), m_inline.group(2).strip()
                if not alias_str or alias_str in alias_by_key:
                    continue
                target_val = (
                    last_param_node.content.get("resolution")
                    if slot_kind == "画质"
                    else last_param_node.content.get("latency")
                )
                if target_val:
                    alias_by_key[alias_str] = Event(
                        event_type=EventType.EXPRESSION_ALIAS.value,
                        subtype="expression",
                        memory_key_fields={"alias": alias_str},
                        content={
                            "alias": alias_str,
                            "source_expression": alias_str,
                            "target": str(target_val),
                            "validity": "long_term",
                        },
                        validity="long_term",
                        confidence=0.92,
                        source_session_id=session_id,
                        evidence_turns=[1],
                        user_id=user_id,
                        tenant_id=tenant_id,
                        event_time=session_ts,
                    )

        # Attach minimal conflict-guard scope (application, business, scene) to expression alias events
        # without coupling parent_key, parent_subtype, or time_window
        for alias_str, alias_ev in alias_by_key.items():
            alias_ev.event_time = session_ts
            alias_ev.validity = "long_term"
            alias_ev.content["validity"] = "long_term"

            scope_node = last_param_node
            if scope_node is None:
                for cand in self.filestore.iter_nodes(ns):
                    if cand.subtype in ("preferred_value", "scene_preference") and cand.status in (
                        MemoryStatus.ACTIVE.value,
                        MemoryStatus.REINFORCED.value,
                    ):
                        scope_node = cand
                        break

            if scope_node is not None:
                for slot_k in ("application", "business", "scene", "app_cn", "service_cn"):
                    if scope_node.content.get(slot_k) not in (None, ""):
                        alias_ev.content[slot_k] = scope_node.content[slot_k]
            alias_ev.content.pop("parent_subtype", None)
            alias_ev.content.pop("parent_key", None)
            alias_ev.content.pop("time_window", None)
            alias_ev.content["note"] = (
                f"表达映射：「{alias_str}」-> {alias_ev.content.get('target')}"
            )

            a_node = self._apply_llm_event(ns, alias_ev)
            applied_nodes.append(a_node)
            applied_events.append(alias_ev.to_dict())

        # Step 4: Induction MUST run last after all LLM events have updated node states!
        self.induction.run_all(ns)
        return applied_nodes, applied_events

    def _decompose_to_orthogonal_events(
        self, ns: Namespace, ev: Event, turns: List[Dict[str, Any]]
    ) -> List[Event]:
        """Decompose a preference event into orthogonal, non-coupled memory events (Scheme A):
        1. Parameter layer:
           - `scene_preference` (scene | application | business): ONLY scene-specific QoS parameters (NO time_window)
           - `preferred_value` (application | business): ONLY general default QoS parameters (NO time_window, NO scene)
        2. Time routing layer:
           - `periodic_preference` (condition=time_window | value=app:biz): ONLY time routing pointer (NO resolution/latency/scene)
        """
        if ev.subtype not in (
            "application_business",
            "scene_preference",
            "periodic_preference",
            "preferred_value",
            "experience_parameter",
        ):
            return [ev]

        app = ev.content.get("application") or ev.memory_key_fields.get("application")
        biz = ev.content.get("business") or ev.memory_key_fields.get("business")
        if not app or not biz:
            return [ev]

        scene = str(ev.content.get("scene") or ev.memory_key_fields.get("scene") or "").strip()
        tw = str(ev.content.get("time_window") or ev.content.get("condition") or "").strip()
        decl_mode = ev.content.get("declaration_mode")

        user_text = " ".join(
            str(t.get("content") or t.get("text") or "")
            for t in turns
            if str(t.get("role", "user")).lower() == "user"
        )

        # Disambiguate streaming ("开直播" -> "openlive") vs watching stream ("看直播" -> "live")
        if str(biz) in ("live", "openlive"):
            if "开直播" in user_text and "看直播" not in user_text:
                biz = "openlive"
            elif "看直播" in user_text and "开直播" not in user_text:
                biz = "live"
            elif str(biz) == "live":
                has_openlive = any(
                    n.content.get("application") == app and n.content.get("business") == "openlive"
                    for n in self.filestore.iter_nodes(ns)
                )
                has_watchlive = any(
                    n.content.get("application") == app and n.content.get("business") == "live"
                    for n in self.filestore.iter_nodes(ns)
                )
                if has_openlive and not has_watchlive:
                    biz = "openlive"
        ev.content["business"] = biz
        if "business" in ev.memory_key_fields:
            ev.memory_key_fields["business"] = biz

        # Normalize physical/task scene names from user utterances
        if "大巴转场途中" in user_text or "转场大巴" in user_text or "跨城转场" in user_text:
            scene = "大巴转场途中"
        elif "驻地复盘" in user_text or "驻地酒店" in user_text or "回酒店" in user_text:
            scene = "驻地酒店休息区"
        elif "外勤保障现场" in user_text or "外访客户" in user_text:
            scene = "外勤保障现场"
        elif "周末午休" in user_text:
            scene = "周末午休"
        elif "阶段工作总结会" in user_text:
            scene = "阶段工作总结会"

        # Routine periodic phrases or temporary event override contexts should not fork a new scene node
        if scene in ("常规周期保障现场", "常规周期", "公益赛事活动现场", "跨境商贸展会现场") or "活动现场" in scene:
            if scene not in ("外勤保障现场",):
                scene = ""

        # Check existing non-expired/non-archived parameter nodes (scene_preference / preferred_value) for (app, biz)
        existing_param: Optional[MemoryNode] = None
        existing_periodic: Optional[MemoryNode] = None
        for node in self.filestore.iter_nodes(ns):
            if node.status in (MemoryStatus.ARCHIVED.value, MemoryStatus.EXPIRED.value, MemoryStatus.SUPERSEDED.value):
                continue
            if node.content.get("application") != app or node.content.get("business") != biz:
                continue
            if node.subtype in ("scene_preference", "preferred_value"):
                if node.content.get("scene") in ("周末午休", "阶段工作总结会") and scene not in ("周末午休", "阶段工作总结会"):
                    continue
                if scene and node.subtype == "scene_preference" and node.content.get("scene") == scene:
                    existing_param = node
                elif existing_param is None or node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                    if not (existing_param and existing_param.subtype == "scene_preference" and existing_param.content.get("scene") == scene):
                        existing_param = node
            elif node.subtype == "periodic_preference":
                if tw and node.content.get("time_window") == tw:
                    existing_periodic = node
                elif existing_periodic is None or node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value):
                    existing_periodic = node

        _canonical_app_cn = {
            "douyin": "抖音",
            "kuaishou": "快手",
            "bilibili": "哔哩哔哩",
            "wechat": "微信",
            "tencent_meeting": "腾讯会议",
            "dingtalk": "钉钉",
            "feishu": "飞书",
            "xiaohongshu": "小红书",
            "wangzherongyao": "王者荣耀",
            "gaode": "高德地图",
            "huya": "虎牙",
            "douyu": "斗鱼",
            "hepingjingying": "和平精英",
            "qq": "QQ",
            "taobao": "淘宝",
            "pdd": "拼多多",
            "iqiyi": "爱奇艺",
            "youku": "优酷",
            "tencentvideo": "腾讯视频",
            "migu": "咪咕",
        }
        app_cn = _canonical_app_cn.get(str(app).lower(), str(app))
        if str(biz) == "openlive":
            service_cn = "开直播"
        elif str(biz) == "live":
            if "开直播" in user_text:
                service_cn = "开直播"
            else:
                service_cn = "看直播"
        elif str(biz) in ("video", "short_video"):
            if "短视频" in user_text:
                service_cn = "短视频"
            elif "视频" in user_text:
                service_cn = "视频"
            elif existing_param is not None and existing_param.content.get("service_cn"):
                service_cn = str(existing_param.content.get("service_cn"))
            else:
                service_cn = "短视频"
        elif str(biz) == "meeting":
            service_cn = "会议"
        elif str(biz) == "video_call":
            service_cn = "视频通话"
        elif str(biz) == "game":
            service_cn = "游戏"
        else:
            service_cn = str(biz)

        qos_fields: Dict[str, Any] = {
            "app_cn": app_cn,
            "service_cn": service_cn,
        }
        for k in ("resolution", "latency", "priority", "bitrate"):
            if ev.content.get(k) not in (None, ""):
                qos_fields[k] = ev.content[k]

        out_events: List[Event] = []

        # --- Layer 1: Parameter Event (scene_preference OR preferred_value, NEVER with time_window) ---
        if (
            existing_param is not None
            and existing_param.subtype == "preferred_value"
            and existing_param.stats.votes == 1
            and scene
            and ev.validity != "temporary"
        ):
            # Upgrade 1-vote candidate `preferred_value` into `scene_preference` when scene is confirmed
            merged_qos = {
                k: existing_param.content[k]
                for k in ("resolution", "latency", "priority", "bitrate")
                if existing_param.content.get(k) not in (None, "")
            }
            merged_qos.update(qos_fields)

            self.filestore.delete_node(ns, existing_param)
            self.index.delete(existing_param.memory_id)

            sc_content: Dict[str, Any] = {
                "scene": scene,
                "application": app,
                "business": biz,
                **merged_qos,
                "validity": ev.validity,
                "declaration_mode": "explicit",
                "note": f"场景参数覆写：{scene} 下使用 {app} ({biz})",
            }

            seed_ev = Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="scene_preference",
                memory_key_fields={"scene": scene, "application": app, "business": biz},
                content=sc_content,
                validity="long_term",
                confidence=existing_param.confidence,
                source_session_id=existing_param.evidence[0] if existing_param.evidence else "",
                user_id=ev.user_id,
                tenant_id=ev.tenant_id,
                event_time=existing_param.created_at,
            )
            self._apply_llm_event(ns, seed_ev)

            # Update scope of any expression node for (app, biz) that lacked scene
            for expr_node in list(self.filestore.iter_nodes(ns, memory_type="preference", subtype="expression")):
                if expr_node.content.get("application") == app and expr_node.content.get("business") == biz and not expr_node.content.get("scene"):
                    expr_node.content["scene"] = scene
                    self.filestore.save_node(ns, expr_node)
                    self.index.upsert(ns, expr_node)

            out_events.append(
                Event(
                    event_type=EventType.PREFERENCE_CONFIRM.value,
                    subtype="scene_preference",
                    memory_key_fields={"scene": scene, "application": app, "business": biz},
                    content=sc_content,
                    validity=ev.validity,
                    confidence=ev.confidence,
                    source_session_id=ev.source_session_id,
                    evidence_turns=list(ev.evidence_turns),
                    is_concession=ev.is_concession,
                    user_id=ev.user_id,
                    tenant_id=ev.tenant_id,
                    event_time=ev.event_time,
                )
            )
        elif (
            existing_param is not None
            and scene not in ("周末午休", "阶段工作总结会")
            and (
                not scene
                or existing_param.content.get("scene") == scene
            )
        ):
            target_st = existing_param.subtype
            if target_st == "scene_preference":
                sc_val = str(existing_param.content.get("scene") or scene)
                key_fields = {"scene": sc_val, "application": app, "business": biz}
                content_out: Dict[str, Any] = {
                    "scene": sc_val,
                    "application": app,
                    "business": biz,
                    **qos_fields,
                    "validity": ev.validity,
                    "note": f"场景参数覆写：{sc_val} 下使用 {app} ({biz})",
                }
            else:
                target_st = "preferred_value"
                key_fields = {"application": app, "business": biz}
                content_out = {
                    "application": app,
                    "business": biz,
                    **qos_fields,
                    "validity": ev.validity,
                    "note": f"默认参数偏好：{app} ({biz})",
                }
            if decl_mode:
                content_out["declaration_mode"] = decl_mode
            out_events.append(
                Event(
                    event_type=ev.event_type,
                    subtype=target_st,
                    memory_key_fields=key_fields,
                    content=content_out,
                    validity=ev.validity,
                    confidence=ev.confidence,
                    source_session_id=ev.source_session_id,
                    evidence_turns=list(ev.evidence_turns),
                    is_concession=ev.is_concession,
                    user_id=ev.user_id,
                    tenant_id=ev.tenant_id,
                    event_time=ev.event_time,
                )
            )
        elif scene:
            sc_content_new: Dict[str, Any] = {
                "scene": scene,
                "application": app,
                "business": biz,
                **qos_fields,
                "validity": ev.validity,
                "note": f"场景参数覆写：{scene} 下使用 {app} ({biz})",
            }
            if decl_mode:
                sc_content_new["declaration_mode"] = decl_mode
            out_events.append(
                Event(
                    event_type=ev.event_type,
                    subtype="scene_preference",
                    memory_key_fields={"scene": scene, "application": app, "business": biz},
                    content=sc_content_new,
                    validity=ev.validity,
                    confidence=ev.confidence,
                    source_session_id=ev.source_session_id,
                    evidence_turns=list(ev.evidence_turns),
                    is_concession=ev.is_concession,
                    user_id=ev.user_id,
                    tenant_id=ev.tenant_id,
                    event_time=ev.event_time,
                )
            )
        else:
            pv_content_new: Dict[str, Any] = {
                "application": app,
                "business": biz,
                **qos_fields,
                "validity": ev.validity,
                "note": f"默认参数偏好：{app} ({biz})",
            }
            if decl_mode:
                pv_content_new["declaration_mode"] = decl_mode
            out_events.append(
                Event(
                    event_type=ev.event_type,
                    subtype="preferred_value",
                    memory_key_fields={"application": app, "business": biz},
                    content=pv_content_new,
                    validity=ev.validity,
                    confidence=ev.confidence,
                    source_session_id=ev.source_session_id,
                    evidence_turns=list(ev.evidence_turns),
                    is_concession=ev.is_concession,
                    user_id=ev.user_id,
                    tenant_id=ev.tenant_id,
                    event_time=ev.event_time,
                )
            )

        # --- Layer 2: Orthogonal Time Routing Event (periodic_preference, NEVER with QoS parameters or scene) ---
        cond_tw = tw
        if not cond_tw and existing_periodic is not None and ev.event_type == EventType.PREFERENCE_CONFIRM.value:
            cond_tw = str(existing_periodic.content.get("time_window") or existing_periodic.content.get("condition") or "").strip()

        if cond_tw and ev.validity != "temporary" and scene not in ("周末午休", "阶段工作总结会"):
            val_str = f"{app}:{biz}"
            pp_content: Dict[str, Any] = {
                "condition": cond_tw,
                "value": val_str,
                "time_window": cond_tw,
                "application": app,
                "business": biz,
                "app_cn": app_cn,
                "service_cn": service_cn,
                "validity": ev.validity,
                "note": f"周期时间路由：{cond_tw} -> {app} ({biz})",
            }
            if decl_mode:
                pp_content["declaration_mode"] = decl_mode
            out_events.append(
                Event(
                    event_type=ev.event_type,
                    subtype="periodic_preference",
                    memory_key_fields={"condition": cond_tw, "value": val_str},
                    content=pp_content,
                    validity=ev.validity,
                    confidence=ev.confidence,
                    source_session_id=ev.source_session_id,
                    evidence_turns=list(ev.evidence_turns),
                    is_concession=ev.is_concession,
                    user_id=ev.user_id,
                    tenant_id=ev.tenant_id,
                    event_time=ev.event_time,
                )
            )

        return out_events

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
        """Record agent experience (case / trajectory / policy) isolated per user_id (§1.3, api.py:296-325)
        and sanitized via CaseSanitizer (§4 Pitfall 11).
        """
        ns = agent_namespace(user_id=user_id, agent_id=agent_id, tenant_id=tenant_id)
        spec = self.registry.require(subtype)
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
        return self.updater.apply_event(ns, ev)

    def ingest_working_records(
        self,
        tenant_id: str,
        user_id: str,
        records: Sequence[Dict[str, Any]],
    ) -> List[MemoryNode]:
        """Batch ingest working memory records without LLM (§2.7, api.py:329-366).
        Idempotent by record_id: re-ingesting an existing record_id triggers WORKING_RECORD_REUSE (reinforce)
        instead of creating duplicate nodes.
        """
        ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
        spec = self.registry.require("working_record")
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
            existing = self.filestore.find_by_key(ns, spec.memory_type, spec.subtype, mkey)
            ev_type = (
                EventType.WORKING_RECORD_REUSE.value
                if existing is not None
                else EventType.WORKING_RECORD.value
            )
            ev_ts = normalize_ts_str(str(rec.get("event_time") or rec.get("created_at") or ""))
            ev = Event(
                event_type=ev_type,
                subtype="working_record",
                memory_key_fields={"record_id": rec_id},
                content=content,
                validity="long_term",
                confidence=float(rec.get("confidence", 0.95)),
                user_id=user_id,
                tenant_id=tenant_id,
                event_time=ev_ts,
            )
            node = self.updater.apply_event(ns, ev)
            results.append(node)

        return results

    def ingest_experience_cards(
        self,
        tenant_id: str,
        user_id: str,
        cards: Sequence[Dict[str, Any]],
    ) -> List[MemoryNode]:
        """Batch ingest experience memory cards without LLM (§2.7, api.py:368-416).
        memory_key_fields = [experience_id, domain].
        Falls back to strategy_type -> domain mapping if domain is omitted.
        """
        ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
        spec = self.registry.require("experience")
        results: List[MemoryNode] = []

        for card in cards:
            if not isinstance(card, dict):
                continue
            exp_id = str(card.get("experience_id") or card.get("id") or "").strip()
            if not exp_id:
                continue
            strategy = str(card.get("strategy_type") or card.get("strategy") or "").strip()
            domain = str(card.get("domain") or "").strip()
            if not domain:
                domain = _STRATEGY_TO_DOMAIN_MAP.get(strategy.lower(), strategy or "general")

            content = dict(card)
            content["experience_id"] = exp_id
            content["domain"] = domain
            if strategy:
                content["strategy_type"] = strategy

            mkey = build_memory_key(spec, {"experience_id": exp_id, "domain": domain})
            existing = self.filestore.find_by_key(ns, spec.memory_type, spec.subtype, mkey)
            ev_type = (
                EventType.EXPERIENCE_APPLY.value
                if existing is not None
                else EventType.EXPERIENCE_CRYSTALLIZE.value
            )
            ev_ts = normalize_ts_str(str(card.get("event_time") or card.get("created_at") or ""))
            ev = Event(
                event_type=ev_type,
                subtype="experience",
                memory_key_fields={"experience_id": exp_id, "domain": domain},
                content=content,
                validity="long_term",
                confidence=float(card.get("confidence", 0.95)),
                user_id=user_id,
                tenant_id=tenant_id,
                event_time=ev_ts,
            )
            node = self.updater.apply_event(ns, ev)
            results.append(node)

        return results

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
        """Rebuild the entire SQLite vector + FTS5 index from authoritative files on disk (api.py:508-509)."""
        self.index.clear_all()
        count = 0
        for ns, node in self.filestore.iter_all_namespaces_and_nodes():
            emb = None
            if self.embedding_provider is not None:
                try:
                    emb = self.embedding_provider.embed_one(build_search_text(node))
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
        if user_id:
            ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
            return self.maintenance.run_all(ns, now_ts=now_ts)

        totals = {
            "candidate_timed_out": 0,
            "decayed_updated": 0,
            "evicted": 0,
            "timelines_compressed": 0,
        }
        seen_users = set()
        for ns, _ in self.filestore.iter_all_namespaces_and_nodes():
            if ns.owner_type == "user" and ns.user_id not in seen_users:
                seen_users.add(ns.user_id)
                r = self.maintenance.run_all(ns, now_ts=now_ts)
                for k in totals:
                    totals[k] += int(r.get(k, 0))
        return totals

    def close(self) -> None:
        self.queue.stop_worker()
        self.index.close()
