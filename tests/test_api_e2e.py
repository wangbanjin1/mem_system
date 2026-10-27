from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from memory_system.api import MemorySystem
from memory_system.core.models import Event, EventType, MemoryStatus
from memory_system.core.namespace import user_namespace
from memory_system.pipeline.sanitizer import CaseSanitizer
from memory_system.providers.embedding import EmbeddingProvider, MockEmbeddingProvider
from memory_system.providers.llm import MockLLMProvider
from server.app import create_app


def _make_system(tmp_path: Path, responder: Any = None) -> MemorySystem:
    return MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(responder=responder),
        router_llm_provider=MockLLMProvider(),
    )


def test_cross_user_isolation_full_flow(tmp_path: Path) -> None:
    """Verify §1.3 & §3 Step 3: User isolation is keyed by user_id across file storage, index, and agent experience."""
    ms = _make_system(tmp_path)

    ms.record_session(
        tenant_id="TENANT_1",
        user_id="USER_001",
        agent_id="AGENT_X",
        session_id="S_U1",
        turns=[{"turn": 1, "role": "user", "content": "以后我说“开播保障”就是指抖音直播"}],
    )
    ms.record_session(
        tenant_id="TENANT_1",
        user_id="USER_002",
        agent_id="AGENT_X",
        session_id="S_U2",
        turns=[{"turn": 1, "role": "user", "content": "以后我说“开黑加速”就是指王者荣耀游戏"}],
    )

    # Agent experience is also isolated per user_id
    ms.record_agent_experience(
        tenant_id="TENANT_1",
        user_id="USER_001",
        agent_id="AGENT_X",
        subtype="case",
        content={
            "intent_category": "live_assurance",
            "initial_state_signature": "missing_app",
            "solution_signature": "fill_from_alias",
            "user_id": "USER_001",
            "session_id": "S_U1",
            "note": "为用户 USER_001 处理 S_U1 直播补全",
        },
    )

    u1_hits = ms.search(tenant_id="TENANT_2", user_id="USER_001", query="开播保障 开黑加速")
    u2_hits = ms.search(tenant_id="TENANT_2", user_id="USER_002", query="开播保障 开黑加速")

    u1_keys = {h["memory_key"] for h in u1_hits}
    u2_keys = {h["memory_key"] for h in u2_hits}
    assert "开播保障" in u1_keys
    assert "开黑加速" not in u1_keys
    assert "开黑加速" in u2_keys
    assert "开播保障" not in u2_keys

    # Verify agent experience isolation between USER_001 and USER_002
    a1_hits = ms.search(
        tenant_id="TENANT_1",
        user_id="USER_001",
        owner_type="agent",
        agent_id="AGENT_X",
        query="live_assurance",
    )
    a2_hits = ms.search(
        tenant_id="TENANT_1",
        user_id="USER_002",
        owner_type="agent",
        agent_id="AGENT_X",
        query="live_assurance",
    )
    assert len(a1_hits) == 1
    assert len(a2_hits) == 0

    # Verify CaseSanitizer stripped identity fields
    case_content = a1_hits[0]["content"]
    assert "user_id" not in case_content
    assert "session_id" not in case_content
    assert "USER_001" not in case_content.get("note", "")
    ms.close()


def test_cross_session_induction_all_three_types(tmp_path: Path) -> None:
    """Verify §2.8: induce_preferred_values, induce_service_to_app, induce_periodic_preferences."""
    ms = _make_system(
        tmp_path,
        responder=lambda _: {
            "events": [
                {
                    "event_type": "PREFERENCE_SET",
                    "subtype": "application_business",
                    "memory_key_fields": {"application": "douyin", "business": "live"},
                    "content": {
                        "application": "douyin",
                        "business": "live",
                        "resolution": "1080p",
                        "priority": "stability",
                    },
                    "validity": "long_term",
                    "evidence_turns": [1],
                }
            ]
        },
    )

    # Session 1 & Session 2 in same time window 12:30-13:30
    for sid in ("S_IND_1", "S_IND_2"):
        ms.record_session(
            tenant_id="default",
            user_id="U_IND",
            agent_id="A1",
            session_id=sid,
            turns=[
                {
                    "turn": 1,
                    "role": "user",
                    "content": "以后抖音直播默认1080p稳定优先，12时30分开始、13时30分结束。",
                }
            ],
        )

    ns = user_namespace("U_IND")
    subtypes_present = {n.subtype: n for n in ms.filestore.iter_nodes(ns)}
    assert "preferred_value" in subtypes_present
    assert "periodic_preference" in subtypes_present
    assert "service_to_app" in subtypes_present
    assert "application_business" not in subtypes_present

    pv_node = subtypes_present["preferred_value"]
    assert pv_node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value)
    assert pv_node.memory_key == "douyin | live"
    assert pv_node.content["resolution"] == "1080p"
    assert "time_window" not in pv_node.content

    pp_node = subtypes_present["periodic_preference"]
    assert pp_node.status in (MemoryStatus.ACTIVE.value, MemoryStatus.REINFORCED.value)
    assert pp_node.memory_key == "12:30-13:30 | douyin:live"
    assert pp_node.content["time_window"] == "12:30-13:30"
    assert "resolution" not in pp_node.content
    ms.close()


def test_rebuild_index_restores_identical_search_results(tmp_path: Path) -> None:
    ms = _make_system(tmp_path)
    ms.ingest_working_records(
        tenant_id="default",
        user_id="U_REBUILD",
        records=[{"record_id": "REC_99", "summary": "王者荣耀排位赛网络加速工单"}],
    )
    before = ms.search(tenant_id="default", user_id="U_REBUILD", query="王者荣耀", bump_heat=False)
    assert len(before) == 1

    ms.index.clear_all()
    assert ms.search(tenant_id="default", user_id="U_REBUILD", query="王者荣耀", bump_heat=False) == []

    rebuilt_count = ms.rebuild_index()
    assert rebuilt_count == 1
    after = ms.search(tenant_id="default", user_id="U_REBUILD", query="王者荣耀", bump_heat=False)
    assert len(after) == 1
    assert after[0]["memory_id"] == before[0]["memory_id"]
    assert math.isclose(after[0]["score"], before[0]["score"], rel_tol=1e-5) if "math" in globals() else (abs(after[0]["score"] - before[0]["score"]) < 1e-5)
    ms.close()


def test_maintenance_candidate_timeout_eviction_and_timeline(tmp_path: Path) -> None:
    ms = _make_system(tmp_path)
    ms.maintenance.eviction_max_active = 2
    ns = user_namespace("U_MAINT")

    # 1. Stale candidate node from 2020
    n_stale = ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="periodic_preference",
            memory_key_fields={"condition": "08:00-09:00", "value": "douyin:live"},
            content={"condition": "08:00-09:00", "value": "douyin:live"},
            event_time="2020-01-01T00:00:00Z",
        ),
    )
    assert n_stale.status == MemoryStatus.CANDIDATE.value

    # 2. Three more nodes to trigger eviction (max_active=2)
    for i in range(3):
        ms.ingest_working_records(
            tenant_id="default",
            user_id="U_MAINT",
            records=[{"record_id": f"REC_M_{i}", "summary": f"工单 {i}"}],
        )

    stats = ms.run_maintenance(user_id="U_MAINT")
    assert stats["candidate_timed_out"] == 1
    assert stats["evicted"] == 1
    assert stats["timelines_compressed"] >= 1
    assert (tmp_path / "users" / "U_MAINT" / "timelines" / "working_record" / "timeline.json").exists()
    ms.close()


def test_async_queue_and_dead_letter_logging(tmp_path: Path) -> None:
    ms = _make_system(tmp_path)
    res = ms.record_session(
        tenant_id="default",
        user_id="U_ASYNC",
        agent_id="A1",
        session_id="S_ASYNC_1",
        turns=[{"turn": 1, "role": "user", "content": "以后抖音直播用1080p"}],
        async_mode=True,
    )
    assert res["status"] == "queued"
    task_obj = ms.queue.process_one(res["task_id"])
    assert task_obj is not None
    assert task_obj["status"] == "done"

    # Test dead-letter logging when handler raises
    def _failing_handler(_: Dict[str, Any]) -> None:
        raise RuntimeError("simulated worker failure")

    ms.queue.handler = _failing_handler
    ms.queue.max_retries = 2
    bad_tid = ms.queue.enqueue({"test": 1})
    ms.queue.process_one(bad_tid)
    final_state = ms.queue.process_one(bad_tid)
    assert final_state is not None
    assert final_state["status"] == "dead"
    log_text = (tmp_path / "logs" / "memory_system.jsonl").read_text(encoding="utf-8")
    assert "task_dead" in log_text
    ms.close()


def test_embedding_count_mismatch_raises() -> None:
    """Verify §4 Pitfall 10: embedding count mismatch must raise ValueError."""
    emb = EmbeddingProvider()
    emb._post_raw = lambda url, headers, payload: json.dumps(  # type: ignore[method-assign]
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}
    ).encode("utf-8")
    with pytest.raises(ValueError, match="Embedding count mismatch"):
        emb.embed(["text1", "text2"])


def test_fastapi_server_routes_direct_invocation(tmp_path: Path) -> None:
    ms = _make_system(tmp_path)
    app = create_app(ms)
    routes = {getattr(r, "path", "") for r in app.routes}
    for expected_path in (
        "/health",
        "/sessions",
        "/search",
        "/ingest/working_records",
        "/ingest/experience_cards",
        "/agent_experience",
        "/maintenance",
        "/rebuild_index",
    ):
        assert expected_path in routes
    ms.close()


def test_cognitive_three_tier_working_and_experience_flow(tmp_path: Path) -> None:
    """Verify Working Memory (working_record) and Experience Memory (experience) crystallization & slot search."""
    ms = _make_system(tmp_path)

    # 1. Record a session with working_record_to_append and experience crystallization
    res = ms.record_session(
        tenant_id="default",
        user_id="USER_COG_01",
        agent_id="intent_agent",
        session_id="S-USER_001-05",
        turns=[
            {
                "turn": 1,
                "user": {"utterance": "今天在高山露天训练场有重要现场保障，活动12时30分正式开始，帮我开抖音开直播保障。"},
                "agent": {"utterance": "收到，确认直接开通吗？"},
            },
            {
                "turn": 2,
                "user": {"utterance": "等等，大型活动现场还得提前进场调试设备呢，掐着点开来不及！帮我把开始时间往前提前半小时吧。"},
                "agent": {"utterance": "明白！已为您将意图保障单起保时间提前30分钟。"},
            },
        ],
        metadata={
            "reference_time": "2026年09月25日12时15分",
            "working_record_to_append": {
                "record_id": "REC_USER_COG_01_20260925_05",
                "session_id": "S-USER_001-05",
                "created_at": "2026年09月25日12时15分",
                "application_name": "抖音",
                "service_name": "开直播",
                "resolution": "1080p",
                "rtt": "50ms",
                "start_timestamp": "2026年09月25日12时30分",
                "end_timestamp": "2026年09月25日13时30分",
                "duration": "60min",
                "location": "高山露天训练场",
                "status": "completed",
            },
        },
        reference_time="2026年09月25日12时15分",
    )
    assert res["status"] == "completed"
    assert any(e["subtype"] == "experience" for e in res["extracted_events"])

    # 2. Verify working_record and experience are indexed with structured columns and retrievable
    hits = ms.search(
        tenant_id="default",
        user_id="USER_COG_01",
        query="今天在奥体中心多功能馆有重大活动保障，帮我开抖音开直播意图保障单",
        top_k=10,
    )
    subtypes = {h["subtype"] for h in hits}
    assert "working_record" in subtypes
    assert "experience" in subtypes
    exp_hit = next(h for h in hits if h["subtype"] == "experience")
    assert exp_hit["content"]["experience_id"] == "EXP_PREWARM_BUFFER_01"

    # 3. Verify rescheduling updates the working_record
    res2 = ms.record_session(
        tenant_id="default",
        user_id="USER_COG_01",
        agent_id="intent_agent",
        session_id="S-USER_001-13",
        turns=[
            {
                "turn": 1,
                "user": {"utterance": "把昨天预约的抖音开直播保障往后推迟一个小时。"},
                "agent": {"utterance": "收到，新的保障时段为12时30分至13时30分，确认进行改期吗？"},
            },
            {
                "turn": 2,
                "user": {"utterance": "对，确认修改。"},
                "agent": {"utterance": "好的，已为您成功办理意图保障单延期变更。"},
            },
        ],
        metadata={"reference_time": "2026年10月23日12时15分"},
        reference_time="2026年10月23日12时15分",
    )
    assert res2["status"] == "completed"
    assert any(e["subtype"] == "working_record" for e in res2["extracted_events"])
    ns = user_namespace("USER_COG_01")
    wr_node = ms.filestore.find_by_key(ns, "working_memory", "working_record", "REC_USER_COG_01_20260925_05")
    assert wr_node is not None
    assert wr_node.content["status"] == "rescheduled"
    assert wr_node.content["start_timestamp"] == "2026年10月23日12时30分"
    ms.close()

