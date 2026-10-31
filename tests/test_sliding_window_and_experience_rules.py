from __future__ import annotations

from pathlib import Path
from typing import List

from memory_system.api import MemorySystem
from memory_system.core.models import Event, EventType
from memory_system.core.namespace import user_namespace
from memory_system.providers.embedding import MockEmbeddingProvider
from memory_system.storage.index import build_pure_phrase_text


class WindowSensitiveEmbeddingProvider(MockEmbeddingProvider):
    """Embedding provider where a full 28-char query vector is diluted (cos=0.31),
    while an 8-char sliding window ('车队颠簸赶路途中') aligns with the pure phrase vector
    ('坐大巴转场途中', cos=0.88)."""

    def embed_one(self, text: str) -> List[float]:
        t = (text or "").strip()
        # Stored pure phrase vector (7 chars)
        if t == "坐大巴转场途中":
            vec = [0.0] * self.dimension
            vec[0] = 0.94
            vec[1] = 0.3411
            return vec
        # Short sliding window (<= 10 chars) containing the core semantic span
        if len(t) <= 10 and "颠簸赶路" in t:
            vec = [0.0] * self.dimension
            vec[0] = 0.92
            vec[1] = 0.3919
            return vec
        # Full long query (> 15 chars): diluted across many clauses so full-query cos < 0.35
        if len(t) > 15 and "颠簸赶路" in t:
            vec = [0.0] * self.dimension
            vec[0] = 0.30
            vec[2] = 0.9539
            return vec
        return super().embed_one(text)


def test_pure_phrase_text_indexing_and_sqlite_column(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path / "data",
        embedding_provider=MockEmbeddingProvider(dimension=64),
    )
    try:
        ns = user_namespace(user_id="u_phrase", tenant_id="t_phrase")
        events = [
            Event(
                event_type=EventType.EXPRESSION_ALIAS.value,
                subtype="expression",
                memory_key_fields={"alias": "午间例行"},
                content={"alias": "午间例行", "target": "12:30-13:30"},
                validity="long_term",
                confidence=0.95,
            ),
            Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="scene_preference",
                memory_key_fields={"scene": "坐大巴转场途中", "application": "抖音", "business": "openlive"},
                content={
                    "scene": "坐大巴转场途中",
                    "application": "抖音",
                    "business": "openlive",
                    "resolution": "720p",
                    "max_delay": "100ms",
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
                    "time_window": "12:30-13:30",
                    "value": "抖音:openlive",
                    "application": "抖音",
                    "business": "openlive",
                },
                validity="long_term",
                confidence=0.92,
            ),
        ]
        nodes = [ms.updater.apply_event(ns, ev) for ev in events]

        assert build_pure_phrase_text(nodes[0]) == "午间例行"
        assert build_pure_phrase_text(nodes[1]) == "坐大巴转场途中"
        assert build_pure_phrase_text(nodes[2]) == "12:30-13:30"

        rows = {r["subtype"]: r for r in ms.index.list_active_slot_rows(ns)}
        assert rows["expression"]["phrase_text"] == "午间例行"
        assert rows["scene_preference"]["phrase_text"] == "坐大巴转场途中"
        assert rows["periodic_preference"]["phrase_text"] == "12:30-13:30"
    finally:
        ms.close()


def test_sliding_window_max_pooling_unlocks_short_phrase_in_long_query(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path / "data",
        embedding_provider=WindowSensitiveEmbeddingProvider(dimension=64),
    )
    try:
        ns = user_namespace(user_id="u_win", tenant_id="t_win")
        ms.updater.apply_event(
            ns,
            Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="preferred_value",
                memory_key_fields={"application": "抖音", "business": "openlive"},
                content={
                    "application": "抖音",
                    "business": "openlive",
                    "resolution": "1080p",
                    "max_delay": "50ms",
                },
                validity="long_term",
                confidence=0.95,
            ),
        )
        ms.updater.apply_event(
            ns,
            Event(
                event_type=EventType.PREFERENCE_SET.value,
                subtype="scene_preference",
                memory_key_fields={"scene": "坐大巴转场途中", "application": "抖音", "business": "openlive"},
                content={
                    "scene": "坐大巴转场途中",
                    "application": "抖音",
                    "business": "openlive",
                    "resolution": "720p",
                    "max_delay": "100ms",
                },
                validity="long_term",
                confidence=0.92,
            ),
        )

        # Full query is 24 chars (diluted full vector cos ~ 0.28), but 8-char sliding window hits '颠簸赶路' (cos > 0.85)
        long_query = "今天下午我们正处在车队颠簸赶路途中准备处理工作单"
        hits = ms.search(tenant_id="t_win", user_id="u_win", query=long_query, top_k=4)
        assert hits
        assert hits[0]["memory_key"] == "坐大巴转场途中 | 抖音 | openlive"
        assert hits[0]["explain"]["vector_sim"] >= 0.85
        assert hits[0]["explain"]["select_sim"] >= 0.98
        assert hits[0]["content"]["resolution"] == "720p"
    finally:
        ms.close()


def test_slow_path_experience_extraction_three_rule_types_and_novel_scenario(tmp_path: Path) -> None:
    ms = MemorySystem(
        data_root=tmp_path / "data",
        embedding_provider=MockEmbeddingProvider(dimension=64),
    )
    try:
        # 1. Rule Type 1: temporal_calendar_adjustment with custom 45-minute lead time
        res_t1 = ms.record_session(
            tenant_id="t_exp",
            user_id="u_exp",
            agent_id="a_exp",
            session_id="S_EXP_T1",
            reference_time="2026-10-12T09:00:00Z",
            turns=[
                {"turn": 1, "role": "user", "content": "今天下午14点到16点在国际会展中心有场重大活动保障，用抖音开直播"},
                {"turn": 2, "role": "assistant", "content": "已按14:00-16:00为您预填抖音开直播保障单。"},
                {"turn": 3, "role": "user", "content": "大型场馆掐着点开来不及调试设备，以后这种重大活动都得提前45分钟进场试机预热"},
            ],
        )
        assert res_t1["extracted_events"]
        ev_t1 = res_t1["extracted_events"][0]["content"]
        assert ev_t1["rule_type"] == "temporal_calendar_adjustment"
        assert ev_t1["recommended_adjustments"]["start_offset_minutes"] == -45
        assert ev_t1["recommended_adjustments"]["duration_add_minutes"] == 45

        # 2. Rule Type 2: novel literal_trap_override scenario (抖音直播彩排走位 -> 飞书会议)
        for sid in ("S_NOVEL_1", "S_NOVEL_2"):
            res_novel = ms.record_session(
                tenant_id="t_exp",
                user_id="u_exp",
                agent_id="a_exp",
                session_id=sid,
                reference_time="2026-10-13T10:00:00Z",
                turns=[
                    {"turn": 1, "role": "user", "content": "明天下午在文化艺术大剧院有场音乐会直播，先跟导演彩排走位"},
                    {"turn": 2, "role": "assistant", "content": "请问是按抖音开直播保障吗？"},
                    {"turn": 3, "role": "user", "content": "不是正式开播，平时这种私下彩排走位我们都是用飞书开视频会议沟通，帮我改成飞书视频会议"},
                ],
            )
        ev_novel = res_novel["extracted_events"][0]["content"]
        assert ev_novel["rule_type"] == "literal_trap_override"
        assert ev_novel["recommended_adjustments"]["application_name"] == "飞书"
        assert ev_novel["recommended_adjustments"]["service_name"] == "会议"
        assert ev_novel["phrase_text"] == "彩排走位"

        # Fast-path application in a subsequent 2-turn session should match the newly active novel card
        res_apply = ms.record_session(
            tenant_id="t_exp",
            user_id="u_exp",
            agent_id="a_exp",
            session_id="S_NOVEL_APPLY",
            reference_time="2026-10-14T10:00:00Z",
            turns=[
                {"turn": 1, "role": "user", "content": "下午在剧院跟导演彩排走位，帮我把保障单排上"},
                {"turn": 2, "role": "user", "content": "对，就按飞书会议来"},
            ],
        )
        assert res_apply["extracted_events"]
        assert res_apply["extracted_events"][0]["event_type"] == EventType.EXPERIENCE_APPLY.value
        assert res_apply["extracted_events"][0]["content"]["experience_id"] == ev_novel["experience_id"]

        # 3. Rule Type 3: ledger_disambiguation_policy
        res_ledger = ms.record_session(
            tenant_id="t_exp",
            user_id="u_exp",
            agent_id="a_exp",
            session_id="S_EXP_LEDGER",
            reference_time="2026-10-15T10:00:00Z",
            turns=[
                {"turn": 1, "role": "user", "content": "下午在财富管理中心办客户路演，按上次那笔意图保障单排时长"},
                {"turn": 2, "role": "assistant", "content": "已按前天30分钟短单为您排期。"},
                {"turn": 3, "role": "user", "content": "不对，前天那笔是临时插空的短会，办客户路演要按上周那笔正式路演单的时长改成120分钟"},
            ],
        )
        ev_ledger = res_ledger["extracted_events"][0]["content"]
        assert ev_ledger["rule_type"] == "ledger_disambiguation_policy"
        assert ev_ledger["recommended_adjustments"]["prefer_formal_duration_over_recent_temp"] is True
    finally:
        ms.close()
