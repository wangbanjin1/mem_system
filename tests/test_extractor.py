from __future__ import annotations

from pathlib import Path

import pytest

from memory_system.core.logging_utils import JsonLinesLogger
from memory_system.core.models import Event, EventType
from memory_system.core.registry import load_registry
from memory_system.pipeline.extractor import Extractor, extract_time_window_from_text
from memory_system.providers.llm import MockLLMProvider, extract_json_from_text


@pytest.mark.parametrize(
    "utterance,expected_tw",
    [
        ("每天12时30分开始、13时30分结束直播", "12:30-13:30"),
        ("从12点半到1点半给我开播保障", "12:30-13:30"),
        ("晚上18:45至19:45进行视频会议", "18:45-19:45"),
        ("12点半到14点15分用抖音", "12:30-14:15"),
        ("12点15分到1点半直播", "12:15-13:30"),
        ("夜里23点开始持续2小时打游戏", "23:00-01:00"),
        ("从20点到22点看直播", "20:00-22:00"),
    ],
)
def test_seven_regex_patterns_for_time_window(utterance: str, expected_tw: str) -> None:
    assert extract_time_window_from_text(utterance) == expected_tw


def test_backfill_time_window_prefers_evidence_turns_then_fallback() -> None:
    ext = Extractor(registry=load_registry())
    turns = [
        {"turn": 1, "role": "user", "content": "上午09:00至10:00开会"},
        {"turn": 2, "role": "user", "content": "中午12时30分开始、13时30分结束用抖音直播"},
    ]
    ev = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live"},
        evidence_turns=[2],
    )
    ext._backfill_time_window(ev, turns)
    assert ev.content["time_window"] == "12:30-13:30"

    # Does not overwrite if LLM already extracted time_window
    ev_existing = Event(
        event_type=EventType.PREFERENCE_SET.value,
        subtype="application_business",
        memory_key_fields={"application": "douyin", "business": "live"},
        content={"application": "douyin", "business": "live", "time_window": "15:00-16:00"},
        evidence_turns=[2],
    )
    ext._backfill_time_window(ev_existing, turns)
    assert ev_existing.content["time_window"] == "15:00-16:00"


def test_validate_rejects_invalid_events_and_logs_discard(tmp_path: Path) -> None:
    log_file = tmp_path / "discard.jsonl"
    ext = Extractor(registry=load_registry(), logger=JsonLinesLogger(log_file))
    turns = [{"turn": 1, "role": "user", "content": "测试对话"}]

    # 1. Invalid event_type
    assert not ext.validate(
        Event(
            event_type="UNKNOWN_EVENT",
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live"},
            evidence_turns=[1],
        ),
        turns,
    )

    # 2. Unknown subtype
    assert not ext.validate(
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="nonexistent_subtype",
            memory_key_fields={"a": "b"},
            content={"a": "b"},
            evidence_turns=[1],
        ),
        turns,
    )

    # 3. Missing memory_key_fields
    assert not ext.validate(
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": None},
            content={"application": "douyin"},
            evidence_turns=[1],
        ),
        turns,
    )

    # 4. Invalid resolution domain
    assert not ext.validate(
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live", "resolution": "9999p"},
            evidence_turns=[1],
        ),
        turns,
    )

    # 5. Unsupported app-business pair in support_matrix (yuanshen does not support meeting)
    assert not ext.validate(
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "yuanshen", "business": "meeting"},
            content={"application": "yuanshen", "business": "meeting"},
            evidence_turns=[1],
        ),
        turns,
    )

    # 6. Non-existent evidence_turns
    assert not ext.validate(
        Event(
            event_type=EventType.PREFERENCE_SET.value,
            subtype="application_business",
            memory_key_fields={"application": "douyin", "business": "live"},
            content={"application": "douyin", "business": "live"},
            evidence_turns=[99],
        ),
        turns,
    )

    log_content = log_file.read_text(encoding="utf-8")
    assert log_content.count("extract_discard") == 6


def test_extractor_normalizes_chinese_app_names_and_extracts() -> None:
    mock_llm = MockLLMProvider(
        responder=lambda _: {
            "events": [
                {
                    "event_type": "PREFERENCE_SET",
                    "subtype": "application_business",
                    "memory_key_fields": {"application": "三角洲行动", "business": "游戏"},
                    "content": {"application": "三角洲行动", "business": "游戏", "resolution": "1080P"},
                    "validity": "long_term",
                    "evidence_turns": [1],
                }
            ]
        }
    )
    ext = Extractor(registry=load_registry(), llm_provider=mock_llm)
    turns = [{"turn": 1, "role": "user", "content": "以后玩三角洲行动默认用1080p，20:00至21:30。"}]
    events = ext.extract(turns, session_id="S1", user_id="U1")
    assert len(events) == 1
    ev = events[0]
    assert ev.memory_key_fields["application"] == "sanjiaozhou"
    assert ev.memory_key_fields["business"] == "game"
    assert ev.content["resolution"] == "1080p"
    assert ev.content["time_window"] == "20:00-21:30"


def test_extract_json_from_text_handles_think_tags_and_fences() -> None:
    raw = "<think>internal reasoning</think>\n```json\n{\"events\": [{\"ok\": 1}]}\n```"
    parsed = extract_json_from_text(raw)
    assert parsed == {"events": [{"ok": 1}]}


@pytest.mark.parametrize(
    "app_alias,expected_canonical",
    [
        ("抖音", "douyin"),
        ("快手直播", "kuaishou"),
        ("哔哩哔哩", "bilibili"),
        ("王者荣耀", "wangzherongyao"),
        ("和平精英", "hepingjingying"),
        ("三角洲行动", "sanjiaozhou"),
        ("原神", "yuanshen"),
        ("腾讯会议", "tencent_meeting"),
        ("微信视频", "wechat"),
    ],
)
def test_extractor_app_alias_normalization(app_alias: str, expected_canonical: str) -> None:
    ext = Extractor(registry=load_registry())
    assert ext.normalize_app(app_alias) == expected_canonical

