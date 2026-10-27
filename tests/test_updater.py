from __future__ import annotations

import time
from pathlib import Path

from memory_system.core.memory_key import build_memory_key, parse_memory_key
from memory_system.core.models import Event, EventType, MemoryStatus
from memory_system.core.namespace import user_namespace
from memory_system.core.registry import load_registry
from memory_system.pipeline.updater import MemoryUpdater
from memory_system.providers.embedding import MockEmbeddingProvider
from memory_system.storage.filestore import FileStore
from memory_system.storage.graph import GraphStore
from memory_system.storage.index import MemoryIndex


def _make_updater(tmp_path: Path) -> tuple[MemoryUpdater, FileStore, MemoryIndex, GraphStore]:
    reg = load_registry()
    fs = FileStore(tmp_path)
    idx = MemoryIndex(tmp_path / "index" / "memory.db")
    graph = GraphStore(tmp_path)
    emb = MockEmbeddingProvider(dimension=32)
    updater = MemoryUpdater(registry=reg, filestore=fs, index=idx, graph=graph, embedding_provider=emb)
    return updater, fs, idx, graph


def test_registry_loads_15_subtypes() -> None:
    reg = load_registry()
    assert len(reg) == 15
    for expected in (
        "application_business",
        "preferred_value",
        "periodic_preference",
        "expression",
        "service_to_app",
        "experience_parameter",
        "time_pattern",
        "environment",
        "scene_preference",
        "constraint",
        "case",
        "trajectory",
        "policy",
        "working_record",
        "experience",
    ):
        assert expected in reg


def test_memory_key_build_and_parse_roundtrip() -> None:
    reg = load_registry()
    spec = reg.require("application_business")
    key = build_memory_key(spec, {"application": "douyin", "business": "live"})
    assert key == "douyin | live"
    parsed = parse_memory_key(spec, key)
    assert parsed == {"application": "douyin", "business": "live"}


def test_memory_key_missing_field_renders_underscore() -> None:
    reg = load_registry()
    spec = reg.require("experience_parameter")
    key = build_memory_key(spec, {"application": "douyin", "business": "live"})
    assert key == "douyin | live | _"
    parsed = parse_memory_key(spec, key)
    assert parsed == {"application": "douyin", "business": "live", "scene": None}


def test_candidate_to_active_and_reinforced_with_threshold_2(tmp_path: Path) -> None:
    updater, fs, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U001")
    # periodic_preference has vote_threshold=2
    ev1 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="periodic_preference",
        memory_key_fields={"condition": "12:30-13:30", "value": "douyin:live"},
        content={"condition": "12:30-13:30", "value": "douyin:live", "resolution": "1080p"},
    )
    n1 = updater.apply_event(ns, ev1)
    assert n1.status == MemoryStatus.CANDIDATE.value
    assert n1.stats.votes == 1

    ev2 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="periodic_preference",
        memory_key_fields={"condition": "12:30-13:30", "value": "douyin:live"},
        content={"condition": "12:30-13:30", "value": "douyin:live", "resolution": "1080p"},
    )
    n2 = updater.apply_event(ns, ev2)
    assert n2.memory_id == n1.memory_id
    assert n2.status == MemoryStatus.ACTIVE.value
    assert n2.stats.votes == 2

    ev3 = Event(
        event_type=EventType.PREFERENCE_CONFIRM.value,
        subtype="periodic_preference",
        memory_key_fields={"condition": "12:30-13:30", "value": "douyin:live"},
        content={"condition": "12:30-13:30", "value": "douyin:live"},
    )
    n3 = updater.apply_event(ns, ev3)
    assert n3.status == MemoryStatus.REINFORCED.value
    assert n3.stats.votes == 3
    assert len(fs.read_history(ns, n3)) == 3


def test_correct_transitions_to_disputed_unless_concession(tmp_path: Path) -> None:
    updater, _, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U001")
    ev1 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p"},
    )
    n1 = updater.apply_event(ns, ev1)
    assert n1.status == MemoryStatus.ACTIVE.value

    ev_corr = Event(
        event_type=EventType.PREFERENCE_CORRECT.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "720p"},
        is_concession=False,
    )
    n2 = updater.apply_event(ns, ev_corr)
    assert n2.status == MemoryStatus.DISPUTED.value

    # Concession correction does NOT enter disputed
    ev_conc = Event(
        event_type=EventType.PREFERENCE_CORRECT.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p"},
        is_concession=True,
    )
    n3 = updater.apply_event(ns, ev_conc)
    assert n3.status == MemoryStatus.ACTIVE.value


def test_clear_archives_and_reassertion_revives(tmp_path: Path) -> None:
    updater, _, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U001")
    ev1 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "bilibili", "business": "video"},
        content={"application": "bilibili", "business": "video", "resolution": "1080p"},
    )
    n1 = updater.apply_event(ns, ev1)
    assert n1.status == MemoryStatus.ACTIVE.value

    ev_clear = Event(
        event_type=EventType.PREFERENCE_CLEAR.value,
        subtype="application_business",
        memory_key_fields={"application": "bilibili", "business": "video"},
        content={"application": "bilibili", "business": "video"},
    )
    n2 = updater.apply_event(ns, ev_clear)
    assert n2.status == MemoryStatus.ARCHIVED.value

    # Re-assertion revives archived node
    ev_revive = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "bilibili", "business": "video"},
        content={"application": "bilibili", "business": "video", "resolution": "4k"},
    )
    n3 = updater.apply_event(ns, ev_revive)
    assert n3.status == MemoryStatus.ACTIVE.value
    assert n3.content["resolution"] == "4k"


def test_description_updates_when_content_or_note_changes(tmp_path: Path) -> None:
    """Verify §4 Pitfall 4: description (L0) is updated when content/note changes before merge."""
    updater, _, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U001")
    ev1 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "720p", "note": "初始720p"},
    )
    n1 = updater.apply_event(ns, ev1)
    assert "初始720p" in n1.description
    assert "720p" in n1.description

    ev2 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p", "note": "升级到1080p高清"},
    )
    n2 = updater.apply_event(ns, ev2)
    assert "升级到1080p高清" in n2.description
    assert "1080p" in n2.description
    assert "初始720p" not in n2.description


def test_merge_preserves_old_value_when_new_field_is_empty(tmp_path: Path) -> None:
    updater, _, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U001")
    ev1 = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p", "priority": "stability"},
    )
    updater.apply_event(ns, ev1)

    ev2 = Event(
        event_type=EventType.PREFERENCE_CONFIRM.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "", "priority": None, "latency": "20ms"},
    )
    n2 = updater.apply_event(ns, ev2)
    assert n2.content["resolution"] == "1080p"
    assert n2.content["priority"] == "stability"
    assert n2.content["latency"] == "20ms"


def test_three_piece_files_and_graph_edges_written(tmp_path: Path) -> None:
    updater, fs, _, graph = _make_updater(tmp_path)
    ns = user_namespace("U001")
    ev = Event(
        event_id="EVT_TEST_01",
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p"},
    )
    node = updater.apply_event(ns, ev)
    ndir = fs.node_dir(ns, node.memory_type, node.subtype, node.memory_key)
    assert (ndir / "memory.json").exists()
    assert (ndir / "content.md").exists()
    assert (ndir / "history.jsonl").exists()

    edges = graph.get_edges(ns, node.memory_id)
    assert len(edges) == 1
    assert edges[0].source_id == "EVT_TEST_01"
    assert edges[0].target_id == node.memory_id


import pytest


@pytest.mark.parametrize(
    "subtype,sample_fields",
    [
        ("application_business", {"application": "douyin", "business": "live"}),
        ("preferred_value", {"application": "douyin", "business": "live"}),
        ("periodic_preference", {"condition": "12:30-13:30", "value": "douyin:live"}),
        ("expression", {"alias": "开播保障", "target": "抖音直播"}),
        ("service_to_app", {"business": "live", "application": "douyin"}),
        ("experience_parameter", {"application": "douyin", "business": "live", "scene": "outdoor_live"}),
        ("time_pattern", {"application": "douyin", "business": "live", "period_type": "weekday_noon"}),
        ("environment", {"environment": "weak_coverage", "application": "douyin", "business": "live"}),
        ("scene_preference", {"scene": "commute", "application": "bilibili", "business": "video"}),
        ("constraint", {"application": "douyin", "constraint_type": "max_bitrate"}),
        ("case", {"intent_category": "live", "initial_state_signature": "s1", "solution_signature": "sol1"}),
        ("trajectory", {"intent_category": "live", "state_pattern": "p1"}),
        ("policy", {"intent_category": "live", "applicable_state_signature": "as1"}),
        ("working_record", {"record_id": "REC_TEST_1"}),
        ("experience", {"experience_id": "EXP_TEST_1", "domain": "qos"}),
    ],
)
def test_all_15_subtypes_create_and_roundtrip_key(tmp_path: Path, subtype: str, sample_fields: dict) -> None:
    updater, fs, _, _ = _make_updater(tmp_path)
    reg = load_registry()
    spec = reg.require(subtype)
    from memory_system.core.namespace import agent_namespace

    ns = agent_namespace("U001", "A001") if spec.owner_type == "agent" else user_namespace("U001")
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype=subtype,
        memory_key_fields={k: sample_fields[k] for k in spec.memory_key_fields},
        content=dict(sample_fields),
    )
    node = updater.apply_event(ns, ev)
    loaded = fs.find_by_key(ns, spec.memory_type, subtype, node.memory_key)
    assert loaded is not None
    assert loaded.memory_id == node.memory_id
    parsed = parse_memory_key(spec, node.memory_key)
    for k in spec.memory_key_fields:
        assert parsed[k] == sample_fields[k]


@pytest.mark.parametrize(
    "event_type,expected_vote_delta",
    [
        (EventType.PREFERENCE_SET.value, 1),
        (EventType.PREFERENCE_CONFIRM.value, 1),
        (EventType.WORKING_RECORD_REUSE.value, 1),
        (EventType.EXPERIENCE_APPLY.value, 1),
        (EventType.PREFERENCE_CLEAR.value, -1),
    ],
)
def test_vote_deltas_for_event_types(tmp_path: Path, event_type: str, expected_vote_delta: int) -> None:
    updater, _, _, _ = _make_updater(tmp_path)
    ns = user_namespace("U_VOTE")
    n = updater.apply_event(
        ns,
        Event(
            event_type=event_type,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live"},
        ),
    )
    assert n.stats.votes == expected_vote_delta

