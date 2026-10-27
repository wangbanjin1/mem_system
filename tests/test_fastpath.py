from __future__ import annotations

from memory_system.core.models import EventType
from memory_system.pipeline.fastpath import FastPathProcessor


def test_fastpath_noise_filtering_skips_llm() -> None:
    fp = FastPathProcessor()
    res = fp.process(
        [
            {"turn": 1, "role": "user", "content": "你好"},
            {"turn": 2, "role": "user", "content": "好的，谢谢！"},
        ]
    )
    assert res.is_noise is True
    assert res.needs_llm is False
    assert res.confirmed == []


def test_fastpath_alias_mapping_without_param_skips_llm() -> None:
    fp = FastPathProcessor()
    res = fp.process(
        [
            {"turn": 1, "role": "user", "content": "以后我说“开播保障”就是指抖音直播"},
        ]
    )
    assert len(res.confirmed) == 1
    ev = res.confirmed[0]
    assert ev.event_type == EventType.EXPRESSION_ALIAS.value
    assert ev.subtype == "expression"
    assert ev.memory_key_fields["alias"] == "开播保障"
    assert ev.content["application"] == "douyin"
    assert ev.content["business"] == "live"
    assert res.needs_llm is False


def test_fastpath_param_signal_forces_llm() -> None:
    """Verify §4 Pitfall 3: if dialogue has parameter signals (1080p, 老规矩, etc.), fastpath MUST NOT skip LLM."""
    fp = FastPathProcessor()
    res = fp.process(
        [
            {"turn": 1, "role": "user", "content": "以后我说“开播保障”就是抖音直播，画质默认1080p"},
        ]
    )
    assert len(res.confirmed) == 1
    assert res.has_param_signal is True
    assert res.needs_llm is True


def test_fastpath_concession_forces_llm() -> None:
    """Verify §4 Pitfall 3: concession turn ('行，那就按X办吧') forces LLM."""
    fp = FastPathProcessor()
    res = fp.process(
        [
            {"turn": 1, "role": "user", "content": "行，那就按你说的抖音直播办吧"},
        ]
    )
    assert res.has_concession is True
    assert res.needs_llm is True


def test_fastpath_correction_and_clear_events() -> None:
    fp = FastPathProcessor()
    res_corr = fp.process(
        [
            {"turn": 1, "role": "user", "content": "不是快手直播，而是抖音直播"},
        ]
    )
    assert len(res_corr.confirmed) == 1
    assert res_corr.confirmed[0].event_type == EventType.PREFERENCE_CORRECT.value
    assert res_corr.confirmed[0].content["application"] == "douyin"

    res_clear = fp.process(
        [
            {"turn": 1, "role": "user", "content": "取消我之前抖音直播的默认偏好设置"},
        ]
    )
    assert len(res_clear.confirmed) == 1
    assert res_clear.confirmed[0].event_type == EventType.PREFERENCE_CLEAR.value


import pytest


@pytest.mark.parametrize(
    "utterance,expected_app,expected_biz",
    [
        ("以后用抖音看短视频", "douyin", "video"),
        ("默认用快手看直播", "kuaishou", "live"),
        ("平时用B站看视频", "bilibili", "video"),
        ("打王者荣耀游戏", "wangzherongyao", "game"),
        ("玩三角洲行动游戏", "sanjiaozhou", "game"),
        ("用腾讯会议开视频会议", "tencent_meeting", "meeting"),
        ("用微信打视频通话", "wechat", "video_call"),
    ],
)
def test_fastpath_app_and_business_slot_extraction(
    utterance: str, expected_app: str, expected_biz: str
) -> None:
    fp = FastPathProcessor()
    res = fp.process([{"turn": 1, "role": "user", "content": utterance}])
    assert len(res.confirmed) == 1
    assert res.confirmed[0].content["application"] == expected_app
    assert res.confirmed[0].content["business"] == expected_biz

