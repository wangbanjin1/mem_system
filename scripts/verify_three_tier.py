from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from memory_system.api import MemorySystem


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="verify_three_tier_"))
    print(f"[Verify 3-Tier] Using data_root: {tmp_dir}")
    try:
        ms = MemorySystem(data_root=tmp_dir)

        # Tier 1: Fact / Preference memory
        ms.record_session(
            tenant_id="default",
            user_id="USER_001",
            agent_id="AGENT_001",
            session_id="SESS_FACT_1",
            turns=[
                {"turn": 1, "role": "user", "content": "以后我说“开播保障”就是指抖音直播。"},
                {"turn": 2, "role": "user", "content": "以后抖音直播默认用1080p，稳定优先，18:45至19:45。"},
            ],
        )

        # Tier 2: Working Memory (Historical ticket/work order records)
        ms.ingest_working_records(
            tenant_id="default",
            user_id="USER_001",
            records=[
                {
                    "record_id": "WO_20261026_001",
                    "application": "sanjiaozhou",
                    "business": "game",
                    "scene": "campus",
                    "status": "resolved",
                    "summary": "三角洲行动晚高峰校园网丢包工单：开启游戏专线加速后时延从120ms降至28ms",
                }
            ],
        )

        # Tier 3: Experience Memory (Domain -> Strategy -> Experience Card)
        ms.ingest_experience_cards(
            tenant_id="default",
            user_id="USER_001",
            cards=[
                {
                    "experience_id": "EXP_LIVE_WEAKNET_01",
                    "domain": "video_streaming",
                    "strategy_type": "weak_net_degradation",
                    "title": "户外弱网直播卡顿降级预案",
                    "problem_pattern": "户外直播上行抖动导致推流频繁掉帧",
                    "trigger_conditions": {
                        "keywords": ["户外直播", "弱网", "掉帧", "卡顿", "上行不稳定"],
                    },
                    "recommended_actions": ["优先保帧率稳定，将分辨率从1080p平滑降级至720p，开启上行冗余编码"],
                }
            ],
        )

        # Verify 4 query categories across the 3 tiers
        queries = [
            ("fact_alias", "开播保障是什么意思？", ["expression"]),
            ("fact_pref", "抖音直播默认分辨率和时段是什么？", ["application_business", "preferred_value"]),
            ("working_memory", "三角洲行动校园网丢包工单怎么处理的？", ["working_record"]),
            ("experience_memory", "户外直播弱网掉帧卡顿有什么排障经验预案？", ["experience"]),
        ]

        for label, q, mtypes in queries:
            hits = ms.search(
                tenant_id="default",
                user_id="USER_001",
                query=q,
                memory_types=mtypes,
                top_k=3,
                levels="full",
            )
            print(f"[Verify 3-Tier] Category={label} Query={q!r} -> {len(hits)} hits")
            assert hits, f"Expected hit for category {label}"
            top = hits[0]
            print(f"   Top hit: subtype={top['subtype']} key={top['memory_key']} score={top['score']}")

        print("[Verify 3-Tier] All 4 query categories across Fact / Working / Experience tiers succeeded!")
        ms.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
