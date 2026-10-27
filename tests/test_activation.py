from __future__ import annotations

import math
import time
from pathlib import Path

from memory_system.api import MemorySystem
from memory_system.core.models import Event, EventType, MemoryStatus
from memory_system.core.namespace import user_namespace
from memory_system.core.registry import load_registry
from memory_system.pipeline.updater import MemoryUpdater, compute_decayed_heat
from memory_system.providers.embedding import MockEmbeddingProvider
from memory_system.providers.llm import MockLLMProvider
from memory_system.storage.filestore import FileStore
from memory_system.storage.index import MemoryIndex


def _fmt_local_ts(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.localtime(epoch))


def test_initial_heat_values_per_event_type(tmp_path: Path) -> None:
    reg = load_registry()
    fs = FileStore(tmp_path)
    updater = MemoryUpdater(registry=reg, filestore=fs)
    ns = user_namespace("U_HEAT")

    # PREFERENCE_SET -> +0.2
    n_pref = updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live"},
        ),
    )
    assert math.isclose(n_pref.stats.heat, 0.2, abs_tol=1e-5)

    # WORKING_RECORD -> +0.1
    n_work = updater.apply_event(
        ns,
        Event(
            event_type=EventType.WORKING_RECORD.value,
            subtype="working_record",
            memory_key_fields={"record_id": "R1"},
            content={"record_id": "R1", "summary": "test"},
        ),
    )
    assert math.isclose(n_work.stats.heat, 0.1, abs_tol=1e-5)

    # EXPERIENCE_CRYSTALLIZE -> +0.2
    n_exp = updater.apply_event(
        ns,
        Event(
            event_type=EventType.EXPERIENCE_CRYSTALLIZE.value,
            subtype="experience",
            memory_key_fields={"experience_id": "E1", "domain": "net"},
            content={"experience_id": "E1", "domain": "net"},
        ),
    )
    assert math.isclose(n_exp.stats.heat, 0.2, abs_tol=1e-5)


def test_heat_half_life_decay_exact(tmp_path: Path) -> None:
    t0 = 1760000000.0
    ts0 = _fmt_local_ts(t0)
    ts_one_half_life = _fmt_local_ts(t0 + 168.0 * 3600.0)
    ts_two_half_lives = _fmt_local_ts(t0 + 336.0 * 3600.0)

    h1 = compute_decayed_heat(0.8, ts0, now_ts=ts_one_half_life, half_life_hours=168.0)
    assert math.isclose(h1, 0.4, rel_tol=1e-4)

    h2 = compute_decayed_heat(0.8, ts0, now_ts=ts_two_half_lives, half_life_hours=168.0)
    assert math.isclose(h2, 0.2, rel_tol=1e-4)


def test_heat_clock_rollback_and_invalid_ts_safety() -> None:
    t0 = 1760000000.0
    ts0 = _fmt_local_ts(t0)
    ts_earlier = _fmt_local_ts(t0 - 3600.0)
    # Clock skew <= 0 should not decay
    assert math.isclose(compute_decayed_heat(0.5, ts0, now_ts=ts_earlier, half_life_hours=168.0), 0.5)
    # Malformed timestamp should not raise
    assert math.isclose(compute_decayed_heat(0.5, "not-a-timestamp", now_ts=ts0, half_life_hours=168.0), 0.5)


def test_bump_heat_on_search_hit_does_not_change_votes_or_status(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_BUMP")
    # Create a candidate node (periodic_preference has vote_threshold=2)
    node = ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="periodic_preference",
            memory_key_fields={"condition": "20:00-21:00", "value": "douyin:live"},
            content={"condition": "20:00-21:00", "value": "douyin:live"},
        ),
    )
    assert node.status == MemoryStatus.CANDIDATE.value
    assert node.stats.votes == 1
    initial_heat = node.stats.heat

    hits = ms.search(tenant_id="default", user_id="U_BUMP", query="douyin:live", bump_heat=True)
    assert len(hits) == 1

    reloaded = ms.filestore.find_by_id(ns, node.memory_id)
    assert reloaded is not None
    assert reloaded.status == MemoryStatus.CANDIDATE.value
    assert reloaded.stats.votes == 1
    assert reloaded.version == 1
    assert math.isclose(reloaded.stats.heat, initial_heat + 0.05, abs_tol=1e-4)
    ms.close()


def test_ebbinghaus_decay_lambda_per_subtype() -> None:
    reg = load_registry()
    assert math.isclose(reg.require("application_business").decay_lambda, 0.02)
    assert math.isclose(reg.require("periodic_preference").decay_lambda, 0.02)
    assert math.isclose(reg.require("preferred_value").decay_lambda, 0.01)
    assert math.isclose(reg.require("expression").decay_lambda, 0.0)
    assert math.isclose(reg.require("working_record").decay_lambda, 0.0)
    assert math.isclose(reg.require("experience").decay_lambda, 0.0)
    assert math.isclose(reg.require("environment").decay_lambda, 0.05)
    assert math.isclose(reg.require("time_pattern").decay_lambda, 0.05)


def test_working_record_and_experience_do_not_decay_over_time(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ms.ingest_working_records(
        tenant_id="default",
        user_id="U_DECAY",
        records=[{"record_id": "REC_100", "summary": "抖音直播工单记录"}],
    )
    ms.ingest_experience_cards(
        tenant_id="default",
        user_id="U_DECAY",
        cards=[{"experience_id": "EXP_100", "domain": "live_qos", "title": "弱网直播掉帧处理"}],
    )

    future_ts = _fmt_local_ts(time.time() + 180 * 86400)
    w_hits = ms.search(
        tenant_id="default",
        user_id="U_DECAY",
        query="抖音直播工单记录",
        memory_types=["working_record"],
        now_ts=future_ts,
    )
    e_hits = ms.search(
        tenant_id="default",
        user_id="U_DECAY",
        query="弱网直播掉帧处理",
        memory_types=["experience"],
        now_ts=future_ts,
    )
    assert math.isclose(w_hits[0]["explain"]["freshness"], 1.0, abs_tol=1e-6)
    assert math.isclose(e_hits[0]["explain"]["freshness"], 1.0, abs_tol=1e-6)
    ms.close()


def test_ingest_working_records_idempotent_reinforces(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    n1 = ms.ingest_working_records(
        tenant_id="default",
        user_id="U_WORK",
        records=[{"record_id": "REC_001", "summary": "首次工单"}],
    )[0]
    assert n1.status == MemoryStatus.ACTIVE.value
    assert n1.stats.votes == 1

    # Re-ingesting the same record_id reinforces rather than creating a second node
    n2 = ms.ingest_working_records(
        tenant_id="default",
        user_id="U_WORK",
        records=[{"record_id": "REC_001", "summary": "复用工单"}],
    )[0]
    assert n2.memory_id == n1.memory_id
    assert n2.status == MemoryStatus.REINFORCED.value
    assert n2.stats.votes == 2
    assert len(list(ms.filestore.iter_nodes(user_namespace("U_WORK"), subtype="working_record"))) == 1
    ms.close()


def test_ingest_experience_cards_fallback_domain_and_filter(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    nodes = ms.ingest_experience_cards(
        tenant_id="default",
        user_id="U_EXP",
        cards=[
            {
                "experience_id": "EXP_001",
                "strategy_type": "weak_net_degradation",
                "title": "弱网降级策略卡",
                "trigger_conditions": {"keywords": ["弱网", "掉帧"]},
            }
        ],
    )
    assert len(nodes) == 1
    assert nodes[0].status == MemoryStatus.ACTIVE.value
    assert nodes[0].content["domain"] == "network_optimization"
    assert nodes[0].memory_key == "EXP_001 | network_optimization"

    # Search isolated by memory_types
    exp_hits = ms.search(
        tenant_id="default",
        user_id="U_EXP",
        query="弱网掉帧怎么处理",
        memory_types=["experience"],
    )
    assert len(exp_hits) == 1
    assert exp_hits[0]["subtype"] == "experience"

    work_hits = ms.search(
        tenant_id="default",
        user_id="U_EXP",
        query="弱网掉帧怎么处理",
        memory_types=["working_record"],
    )
    assert len(work_hits) == 0
    ms.close()
