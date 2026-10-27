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
    tmp_dir = Path(tempfile.mkdtemp(prefix="memory_smoke_"))
    print(f"[Smoke] Using temporary data_root: {tmp_dir}")
    try:
        ms = MemorySystem(data_root=tmp_dir)
        print(f"[Smoke] Loaded {len(ms.registry)} subtypes; LLM={ms.llm_provider.model}, Emb={ms.embedding_provider.model}")

        # 1. Record a real session using DashScope qwen3.8-flash + text-embedding-v4
        res1 = ms.record_session(
            tenant_id="tenant_smoke",
            user_id="USER_SMOKE_01",
            agent_id="AGENT_01",
            session_id="SESS_001",
            turns=[
                {
                    "turn": 1,
                    "role": "user",
                    "content": "以后我用抖音直播都按老规矩来，画质用1080p，稳定优先，每天12时30分开始、13时30分结束。",
                },
                {
                    "turn": 2,
                    "role": "assistant",
                    "content": "好的，已为您记录抖音直播长期配置：1080p、稳定优先，时段 12:30-13:30。",
                },
            ],
        )
        print("[Smoke] Session 1 recorded:", res1)

        # 2. Record a temporary variant session (should NOT pollute the long-term 1080p node)
        res2 = ms.record_session(
            tenant_id="tenant_smoke",
            user_id="USER_SMOKE_01",
            agent_id="AGENT_01",
            session_id="SESS_002",
            turns=[
                {
                    "turn": 1,
                    "role": "user",
                    "content": "今天抖音直播临时调成720p吧，网络不太好。",
                },
                {
                    "turn": 2,
                    "role": "assistant",
                    "content": "收到，本次为您临时调整为 720p。",
                },
            ],
        )
        print("[Smoke] Session 2 (temporary variant) recorded:", res2)

        # 3. Search preference
        hits = ms.search(
            tenant_id="tenant_smoke",
            user_id="USER_SMOKE_01",
            query="抖音直播按老规矩用什么分辨率和时段？",
            top_k=5,
            levels="full",
        )
        print(f"[Smoke] Search returned {len(hits)} hits:")
        for h in hits:
            print(
                f"  - [{h['subtype']}] key={h['memory_key']} status={h['status']} "
                f"res={h['content'].get('resolution')} tw={h['content'].get('time_window')} "
                f"score={h['score']}"
            )
        assert hits, "Expected at least 1 search hit"
        top_ab = next((h for h in hits if h["subtype"] == "application_business"), hits[0])
        assert top_ab["content"].get("resolution") == "1080p", f"Expected 1080p, got {top_ab['content']}"
        assert top_ab["content"].get("time_window") == "12:30-13:30", f"Expected 12:30-13:30, got {top_ab['content']}"
        print("[Smoke] SUCCESS: Real LLM + Embedding smoke test passed without variant pollution!")
        ms.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
