from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from memory_system.api import MemorySystem
from memory_system.core.models import Event, EventType, MemoryStatus
from memory_system.core.namespace import user_namespace
from memory_system.core.registry import load_registry
from memory_system.pipeline.extractor import Extractor
from memory_system.providers.embedding import MockEmbeddingProvider
from memory_system.providers.llm import MockLLMProvider


def test_normalize_validity_temporary_veto_overrides_long_term() -> None:
    """Case 1: '临时' has one-vote veto even if sentence also says '老规矩' or '以后'."""
    ext = Extractor(registry=load_registry())
    turns = [
        {"turn": 1, "role": "user", "content": "平时按老规矩1080p，但今天临时调成720p吧"},
    ]
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "720p", "validity": "long_term"},
        validity="long_term",
        evidence_turns=[1],
    )
    ext._normalize_validity(ev, turns)
    assert ev.validity == "temporary"
    assert ev.content["validity"] == "temporary"


def test_normalize_validity_strong_long_term_words() -> None:
    """Case 2: Strong long-term words ('以后', '默认', '一直') force long_term."""
    ext = Extractor(registry=load_registry())
    turns = [
        {"turn": 1, "role": "user", "content": "以后我用抖音直播默认都用1080p"},
    ]
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p", "validity": "temporary"},
        validity="temporary",
        evidence_turns=[1],
    )
    ext._normalize_validity(ev, turns)
    assert ev.validity == "long_term"
    assert ev.content["validity"] == "long_term"


def test_normalize_validity_weak_long_term_words() -> None:
    """Case 3: Weak long-term words ('老规矩', '照旧', '平时', '惯例') force long_term."""
    ext = Extractor(registry=load_registry())
    turns = [
        {"turn": 1, "role": "user", "content": "按老规矩给我开播，照旧用1080p"},
    ]
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p"},
        validity="temporary",
        evidence_turns=[1],
    )
    ext._normalize_validity(ev, turns)
    assert ev.validity == "long_term"
    assert ev.content["validity"] == "long_term"


def test_normalize_validity_ignores_agent_utterance_pollution() -> None:
    """Case 4: Agent repeating '长期配置' or '默认' in assistant turn must NOT override user's '临时'."""
    ext = Extractor(registry=load_registry())
    turns = [
        {"turn": 1, "role": "user", "content": "今天临时用720p"},
        {"turn": 2, "role": "assistant", "content": "好的，不修改您的以后默认长期配置，本次临时使用720p。"},
    ]
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "720p"},
        validity="long_term",
        evidence_turns=[1, 2],
    )
    ext._normalize_validity(ev, turns)
    assert ev.validity == "temporary"
    assert ev.content["validity"] == "temporary"


def test_updater_ignores_temporary_event_on_long_term_node(tmp_path: Path) -> None:
    """Updater guardrail: temporary event on non-temporary node is a strict no-op and logs event_ignored_temporary."""
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_POLLUTE")
    ev_long = Event(
        event_id="EVT_LONG_1",
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "1080p", "time_window": "12:30-13:30"},
        validity="long_term",
    )
    n1 = ms.updater.apply_event(ns, ev_long)
    assert n1.content["resolution"] == "1080p"
    assert n1.stats.votes == 1
    assert n1.version == 1

    ev_temp = Event(
        event_id="EVT_TEMP_2",
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "resolution": "720p", "time_window": "20:00-21:00"},
        validity="temporary",
    )
    n2 = ms.updater.apply_event(ns, ev_temp)
    assert n2.content["resolution"] == "1080p"
    assert n2.content["time_window"] == "12:30-13:30"
    assert n2.stats.votes == 1
    assert n2.version == 1

    log_text = (tmp_path / "logs" / "memory_system.jsonl").read_text(encoding="utf-8")
    assert "event_ignored_temporary" in log_text
    ms.close()


def test_temporary_node_retrieval_score_penalized_by_half(tmp_path: Path) -> None:
    """Search guardrail: if first session created a temporary node, its retrieval score is multiplied by 0.5."""
    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(),
    )
    ns = user_namespace("U_SCORE_PENALTY")
    ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live", "resolution": "720p"},
            validity="temporary",
        ),
    )
    ms.updater.apply_event(
        ns,
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "kuaishou", "business": "live"},
            content={"application": "kuaishou", "business": "live", "resolution": "1080p"},
            validity="long_term",
        ),
    )

    hits = ms.search(tenant_id="default", user_id="U_SCORE_PENALTY", query="live", bump_heat=False)
    by_app = {h["content"]["application"]: h for h in hits}
    assert by_app["douyin"]["validity"] == "temporary"
    assert by_app["kuaishou"]["validity"] == "long_term"
    assert by_app["douyin"]["score"] < by_app["kuaishou"]["score"] * 0.65
    ms.close()


def test_three_session_variant_pollution_regression(tmp_path: Path) -> None:
    """Three-session regression (§4 Pitfall 2):
    Session A (老规矩 1080p, 12:30-13:30)
    -> Session B (今天临时调 720p, even if LLM wrongly labels it long_term!)
    -> Session C (老规矩轮检索): must still return 1080p and 12:30-13:30!
    """
    call_idx = {"n": 0}

    def _mock_llm(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            # Session A: LLM returns long_term 1080p (omits time_window to test backfill too!)
            return {
                "events": [
                    {
                        "event_type": "PREFERENCE_SET",
                        "subtype": "application_business",
                        "memory_key_fields": {"application": "douyin", "business": "live"},
                        "content": {"application": "douyin", "business": "live", "resolution": "1080p"},
                        "validity": "long_term",
                        "evidence_turns": [1],
                    }
                ]
            }
        # Session B: LLM mistakenly labels "今天临时调720p" as validity="long_term"
        return {
            "events": [
                {
                    "event_type": "PREFERENCE_SET",
                    "subtype": "application_business",
                    "memory_key_fields": {"application": "douyin", "business": "live"},
                    "content": {
                        "application": "douyin",
                        "business": "live",
                        "resolution": "720p",
                        "validity": "long_term",
                    },
                    "validity": "long_term",
                    "evidence_turns": [1],
                }
            ]
        }

    ms = MemorySystem(
        data_root=tmp_path,
        embedding_provider=MockEmbeddingProvider(dimension=32),
        llm_provider=MockLLMProvider(responder=_mock_llm),
    )

    # Session A: Establish long-term preference
    ms.record_session(
        tenant_id="default",
        user_id="U_REG",
        agent_id="A1",
        session_id="SESS_A",
        turns=[
            {"turn": 1, "role": "user", "content": "以后抖音直播按老规矩用1080p，12时30分开始、13时30分结束。"},
        ],
    )

    # Session B: Temporary variant
    ms.record_session(
        tenant_id="default",
        user_id="U_REG",
        agent_id="A1",
        session_id="SESS_B",
        turns=[
            {"turn": 1, "role": "user", "content": "今天抖音直播临时调成720p，网络太差了。"},
            {"turn": 2, "role": "assistant", "content": "已为您临时调整配置，不影响您的长期配置。"},
        ],
    )

    # Session C: Retrieve for "老规矩"
    hits = ms.search(tenant_id="default", user_id="U_REG", query="抖音直播老规矩配置", top_k=5)
    pv_hit = next(h for h in hits if h["subtype"] == "preferred_value")
    assert pv_hit["content"]["resolution"] == "1080p"
    assert "time_window" not in pv_hit["content"]
    assert pv_hit["validity"] == "long_term"

    pp_hit = next(h for h in hits if h["subtype"] == "periodic_preference")
    assert pp_hit["content"]["time_window"] == "12:30-13:30"
    assert "resolution" not in pp_hit["content"]
    assert pp_hit["validity"] == "long_term"
    ms.close()
