from __future__ import annotations

from pathlib import Path

from memory_system.api import MemorySystem
from memory_system.core.models import Edge, EdgeType, Event, EventType
from memory_system.core.namespace import user_namespace
from memory_system.providers.embedding import MockEmbeddingProvider
from memory_system.providers.llm import MockLLMProvider


def test_tiered_loading_levels_l0_l1_full(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_SEARCH")
    ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live", "resolution": "1080p", "priority": "stability"},
        ),
    )

    h_l0 = ms.search(tenant_id="default", user_id="U_SEARCH", query="douyin", levels="L0")[0]
    assert h_l0["description"]
    assert h_l0["overview"] is None
    assert h_l0["content_md"] is None

    h_l1 = ms.search(tenant_id="default", user_id="U_SEARCH", query="douyin", levels="L0+L1")[0]
    assert h_l1["description"]
    assert h_l1["overview"] is not None
    assert h_l1["content_md"] is None

    h_full = ms.search(tenant_id="default", user_id="U_SEARCH", query="douyin", levels="full")[0]
    assert h_full["description"]
    assert h_full["overview"] is not None
    assert h_full["content_md"] is not None
    assert "## Overview (L1)" in h_full["content_md"]
    assert "## Details (L2)" in h_full["content_md"]
    ms.close()


def test_hybrid_scoring_explain_fields(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_EXPLAIN")
    ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live", "resolution": "1080p"},
        ),
    )

    hits = ms.search(tenant_id="default", user_id="U_EXPLAIN", query="douyin live 1080p")
    assert len(hits) == 1
    exp = hits[0]["explain"]
    for key in ("sim", "freshness", "heat", "confidence", "decay_lambda", "validity", "final_score", "memory_key"):
        assert key in exp
    assert exp["memory_key"] == "douyin | live"
    assert exp["final_score"] > 0.0
    ms.close()


def test_follow_edges_1_and_2_hops(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_GRAPH")
    node = ms.updater.apply_event(
        ns,
        Event(
            event_id="EVT_G1",
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live"},
        ),
    )
    ms.graph.add_edge(
        Edge(
            source_id=node.memory_id,
            target_id="MEM_OLD_01",
            edge_type=EdgeType.SUPERSEDES.value,
            user_id="U_GRAPH",
        )
    )
    ms.graph.add_edge(
        Edge(
            source_id="MEM_OLD_01",
            target_id="EVT_OLD_01",
            edge_type=EdgeType.DERIVED_FROM.value,
            user_id="U_GRAPH",
        )
    )

    hit_1hop = ms.search(tenant_id="default", user_id="U_GRAPH", query="douyin", follow_edges=1)[0]
    assert "EVT_G1" in hit_1hop["neighbors"]
    assert "MEM_OLD_01" in hit_1hop["neighbors"]
    assert "EVT_OLD_01" not in hit_1hop["neighbors"]

    hit_2hop = ms.search(tenant_id="default", user_id="U_GRAPH", query="douyin", follow_edges=2)[0]
    assert "EVT_OLD_01" in hit_2hop["neighbors"]
    ms.close()


def test_router_alias_short_circuit_and_expansion(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_ROUTER")
    ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.EXPRESSION_ALIAS.value,
            subtype="expression",
            memory_key_fields={"alias": "开播保障"},
            content={"alias": "开播保障", "target": "抖音户外直播", "application": "douyin", "business": "live"},
        ),
    )
    route_res = ms.router.route(ns, "今晚给我开播保障")
    assert route_res["short_circuit"] is True
    assert "抖音户外直播" in route_res["expanded_query"]
    assert len(route_res["matched_aliases"]) == 1
    ms.close()
