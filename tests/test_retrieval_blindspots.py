from __future__ import annotations

from pathlib import Path
from typing import List

from memory_system.api import MemorySystem
from memory_system.core.models import Event, EventType
from memory_system.core.namespace import user_namespace
from memory_system.providers.embedding import MockEmbeddingProvider


class CountingEmbeddingProvider(MockEmbeddingProvider):
    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.call_count = 0

    def embed_one(self, text: str) -> List[float]:
        self.call_count += 1
        return super().embed_one(text)


def _seed_user001_orthogonal_memories(ms: MemorySystem, tenant_id: str = "t_blind", user_id: str = "u_001") -> None:
    ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
    events = [
        # Main storyline: 抖音 + openlive (午间例行 -> 12:30-13:30, 大巴转场途中)
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="preferred_value",
            memory_key_fields={"application": "抖音", "business": "openlive"},
            content={
                "application": "抖音",
                "business": "openlive",
                "resolution": "1080p",
                "max_delay": "50ms",
                "bit_rate": "15Mbps",
            },
            validity="long_term",
            confidence=0.95,
        ),
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="scene_preference",
            memory_key_fields={"application": "抖音", "business": "openlive", "scene": "大巴转场途中"},
            content={
                "application": "抖音",
                "business": "openlive",
                "scene": "大巴转场途中",
                "resolution": "720p",
                "max_delay": "100ms",
                "bit_rate": "10Mbps",
            },
            validity="long_term",
            confidence=0.92,
        ),
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="periodic_preference",
            memory_key_fields={"condition": "12:30-13:30", "value": "抖音:openlive"},
            content={
                "condition": "12:30-13:30",
                "value": "抖音:openlive",
                "time_window": "12:30-13:30",
                "application": "抖音",
                "business": "openlive",
            },
            validity="long_term",
            confidence=0.92,
        ),
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="service_to_app",
            memory_key_fields={"business": "openlive", "application": "抖音"},
            content={"business": "openlive", "application": "抖音"},
            validity="long_term",
            confidence=0.90,
        ),
        Event(
            event_type=EventType.EXPRESSION_ALIAS.value,
            subtype="expression",
            memory_key_fields={"alias": "午间例行"},
            content={
                "alias": "午间例行",
                "target": "12:30-13:30",
                "category": "time_window",
            },
            validity="long_term",
            confidence=0.95,
        ),
        Event(
            event_type=EventType.EXPRESSION_ALIAS.value,
            subtype="expression",
            memory_key_fields={"alias": "保个基本画质"},
            content={
                "alias": "保个基本画质",
                "target": "720p",
                "category": "qos_degraded",
                "application": "抖音",
                "business": "openlive",
            },
            validity="long_term",
            confidence=0.92,
        ),
        # Sub storyline: 钉钉 + meeting (阶段工作总结会)
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="preferred_value",
            memory_key_fields={"application": "钉钉", "business": "meeting"},
            content={
                "application": "钉钉",
                "business": "meeting",
                "resolution": "4K",
                "max_delay": "100ms",
                "bit_rate": "10Mbps",
            },
            validity="long_term",
            confidence=0.90,
        ),
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="scene_preference",
            memory_key_fields={"application": "钉钉", "business": "meeting", "scene": "阶段工作总结会"},
            content={
                "application": "钉钉",
                "business": "meeting",
                "scene": "阶段工作总结会",
                "resolution": "720p",
                "max_delay": "100ms",
                "bit_rate": "5Mbps",
            },
            validity="temporary",
            confidence=0.88,
        ),
    ]
    for ev in events:
        ms.updater.apply_event(ns, ev)


def test_blindspot_1_clause_negation_and_contrast(tmp_path: Path) -> None:
    ms = MemorySystem(data_root=tmp_path / "data", embedding_provider=MockEmbeddingProvider(dimension=64))
    try:
        _seed_user001_orthogonal_memories(ms)

        # Case A: Negating 抖音+开直播 while switching to 钉钉+开会
        q1 = "今天不用抖音开直播了，改在阶段工作总结会上用钉钉开个会"
        hits1 = ms.search(tenant_id="t_blind", user_id="u_001", query=q1, top_k=6)
        assert hits1
        assert hits1[0]["memory_key"] == "钉钉 | meeting"
        ding_keys = {h["memory_key"] for h in hits1[:2]}
        assert "钉钉 | meeting" in ding_keys
        assert "阶段工作总结会 | 钉钉 | meeting" in ding_keys

        # Verify 抖音 | openlive is marked negation_penalized
        douyin_hits = [h for h in hits1 if "抖音" in h["memory_key"] and "openlive" in h["memory_key"]]
        for dh in douyin_hits:
            assert dh["explain"]["negation_penalized"] is True
            assert dh["score"] < hits1[0]["score"] * 0.4

        # Case B: Negating the routine alias "午间例行"
        q2 = "别按午间例行那套来，今天用钉钉开会"
        hits2 = ms.search(tenant_id="t_blind", user_id="u_001", query=q2, top_k=5)
        assert hits2
        assert hits2[0]["memory_key"] == "钉钉 | meeting"
    finally:
        ms.close()


def test_blindspot_2_paraphrased_split_token_alias_and_scene(tmp_path: Path) -> None:
    ms = MemorySystem(data_root=tmp_path / "data", embedding_provider=MockEmbeddingProvider(dimension=64))
    try:
        _seed_user001_orthogonal_memories(ms)

        # Paraphrased alias: "中午那个例行档期" -> matches "午间例行" -> 12:30-13:30 -> 抖音 | openlive
        q_alias = "按咱们中午那个例行档期准备一下"
        hits_alias = ms.search(tenant_id="t_blind", user_id="u_001", query=q_alias, top_k=5)
        top_keys = [h["memory_key"] for h in hits_alias[:3]]
        assert "抖音 | openlive" in top_keys

        # Paraphrased scene + paraphrased QoS alias: "在坐大巴车转场的路上" + "画质保个基本档位"
        q_scene = "现在在坐大巴车转场的路上，画质保个基本档位就行"
        hits_scene = ms.search(tenant_id="t_blind", user_id="u_001", query=q_scene, top_k=5)
        top_scene_keys = [h["memory_key"] for h in hits_scene[:3]]
        assert "大巴转场途中 | 抖音 | openlive" in top_scene_keys
        assert "保个基本画质" in [h["memory_key"] for h in hits_scene]
    finally:
        ms.close()


def test_blindspot_3_implicit_service_collocations(tmp_path: Path) -> None:
    ms = MemorySystem(data_root=tmp_path / "data", embedding_provider=MockEmbeddingProvider(dimension=64))
    try:
        _seed_user001_orthogonal_memories(ms)

        # Implicit openlive collocation ("上麦开播推流") without saying "开直播"
        q_live = "准备上麦开播推流了，帮我看下默认参数"
        hits_live = ms.search(tenant_id="t_blind", user_id="u_001", query=q_live, top_k=4)
        assert hits_live[0]["memory_key"] == "抖音 | openlive"
        assert hits_live[0]["explain"]["select_sim"] >= 0.93

        # Implicit meeting collocation ("线上投屏" / "远程碰头") with 钉钉
        q_meet = "一会儿在钉钉搞个线上投屏远程碰头"
        hits_meet = ms.search(tenant_id="t_blind", user_id="u_001", query=q_meet, top_k=4)
        assert hits_meet[0]["memory_key"] == "钉钉 | meeting"
        assert hits_meet[0]["explain"]["select_sim"] >= 0.93
    finally:
        ms.close()


def test_cascading_short_circuit_skips_vector_when_select_locks_param(tmp_path: Path) -> None:
    counting_emb = CountingEmbeddingProvider(dimension=64)
    ms = MemorySystem(data_root=tmp_path / "data", embedding_provider=counting_emb)
    try:
        _seed_user001_orthogonal_memories(ms)
        ms.ingest_working_records(
            tenant_id="t_blind",
            user_id="u_001",
            records=[
                {
                    "record_id": "WR_BLIND_01",
                    "summary": "排查蜂窝基站切换引发的TCP重传毛刺",
                    "metrics": {"rtt_ms": 180},
                }
            ],
        )
        calls_before = counting_emb.call_count

        # 1. With cascade_short_circuit=True on a slot-complete query -> skips embed_one!
        hits_fast = ms.search(
            tenant_id="t_blind",
            user_id="u_001",
            query="午间例行在大巴转场途中用抖音开直播",
            top_k=4,
            cascade_short_circuit=True,
        )
        assert hits_fast
        assert hits_fast[0]["explain"]["vector_short_circuited"] is True
        assert counting_emb.call_count == calls_before

        # 2. With cascade_short_circuit=True on an unstructured troubleshooting query -> automatically falls back to vector!
        hits_fallback = ms.search(
            tenant_id="t_blind",
            user_id="u_001",
            query="基站切换引发的TCP重传毛刺怎么处理",
            top_k=4,
            cascade_short_circuit=True,
        )
        assert hits_fallback
        assert hits_fallback[0]["explain"]["vector_short_circuited"] is False
        assert counting_emb.call_count == calls_before + 1
    finally:
        ms.close()


class SemanticClusterEmbeddingProvider(MockEmbeddingProvider):
    """Simulates a dense semantic embedding model where zero-character-overlap
    colloquial phrases map to the same semantic cluster vector as their anchor nodes."""

    def embed_one(self, text: str) -> list[float]:
        t = text or ""
        # Cluster 1: Noon routine ("午间例行" <-> "白日半晌那场固定功课")
        if "午间例行" in t or "白日半晌那场固定功课" in t:
            vec = [0.0] * self.dimension
            vec[0] = 0.95
            vec[1] = 0.3122
            return vec
        # Cluster 2: Coach transfer scene ("大巴转场途中" <-> "车队颠簸赶路那会儿")
        if "大巴转场途中" in t or "车队颠簸赶路那会儿" in t:
            vec = [0.0] * self.dimension
            vec[2] = 0.95
            vec[3] = 0.3122
            return vec
        return super().embed_one(text)


def test_vector_anchor_adsorption_triggers_two_hop_and_sql_select(tmp_path: Path) -> None:
    """Even when a query has ZERO character/keyword overlap with any rule dictionary or alias,
    Vector-Anchor Adsorption hits the expression/scene anchor row via cosine similarity,
    reads its structured slot columns (target_col/app_col/biz_col/scene_col), executes Hop 2
    pointer resolution, and triggers SQL SELECT to lock onto the parameter card."""
    ms = MemorySystem(
        data_root=tmp_path / "data",
        embedding_provider=SemanticClusterEmbeddingProvider(dimension=64),
    )
    try:
        _seed_user001_orthogonal_memories(ms)

        # Case A: Zero literal overlap with "午间例行" -> Vector hits expression row -> unlocks 12:30-13:30
        # -> Hop 2 periodic_preference unlocks 抖音 + openlive -> SQL SELECT locks onto "抖音 | openlive" (0.93)
        q_zero_kw = "按白日半晌那场固定功课准备一下"
        hits_a = ms.search(tenant_id="t_blind", user_id="u_001", query=q_zero_kw, top_k=5)
        by_key_a = {h["memory_key"]: h for h in hits_a}
        assert "抖音 | openlive" in by_key_a
        card_a = by_key_a["抖音 | openlive"]
        assert card_a["explain"]["select_sim"] >= 0.93
        assert "向量锚点吸附" in card_a["explain"]["structured_match"]
        assert card_a["explain"]["vector_anchor_hits"]

        # Case B: Zero literal overlap with "大巴转场途中" -> Vector hits scene_preference row
        # -> unlocks scene=大巴转场途中, app=抖音, biz=openlive -> SQL SELECT locks onto scene card (0.98) and baseline card (0.93)
        q_zero_scene = "车队颠簸赶路那会儿怎么配"
        hits_b = ms.search(tenant_id="t_blind", user_id="u_001", query=q_zero_scene, top_k=5)
        assert hits_b[0]["memory_key"] == "大巴转场途中 | 抖音 | openlive"
        assert hits_b[0]["explain"]["select_sim"] >= 0.98
        assert "向量锚点吸附" in hits_b[0]["explain"]["structured_match"]
    finally:
        ms.close()

