from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from memory_system.api import MemorySystem
from memory_system.core.namespace import user_namespace
from memory_system.pipeline.updater import compute_ebbinghaus_retention, normalize_ts_str

CROSS_DATA_DIR = Path(r"D:\coding\cross_memory_data\data\generated")


def load_sessions_list(persona_dir: Path) -> List[Dict[str, Any]]:
    sessions_file = persona_dir / "sessions.jsonl"
    result: List[Dict[str, Any]] = []
    for line in sessions_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(json.loads(line))
    return result


def load_traces_map(persona_dir: Path) -> Dict[str, Dict[str, Any]]:
    traces_file = persona_dir / "memory_traces.json"
    if not traces_file.exists():
        return {}
    traces = json.loads(traces_file.read_text(encoding="utf-8"))
    return {t["session_id"]: t for t in traces}


def dump_user_memories(
    ms: MemorySystem, user_id: str, now_ts: str, tenant_id: str = "default"
) -> List[Dict[str, Any]]:
    ns = user_namespace(user_id=user_id, tenant_id=tenant_id)
    nodes = list(ms.filestore.iter_nodes(ns))
    out = []
    for n in sorted(nodes, key=lambda x: (x.subtype, x.memory_key)):
        spec = ms.registry.get(n.subtype)
        base_lam = spec.decay_lambda if spec is not None else 0.02
        ret, stab, eff_lam = compute_ebbinghaus_retention(
            decay_lambda=base_lam,
            last_reinforced_at=n.stats.last_reinforced_at or n.updated_at or n.created_at,
            now_ts=now_ts,
            positive_votes=n.stats.positive_votes,
            reuse_count=n.stats.reuse_count,
        )
        out.append(
            {
                "memory_id": n.memory_id,
                "memory_type": n.memory_type,
                "subtype": n.subtype,
                "memory_key": n.memory_key,
                "status": n.status,
                "votes": n.stats.votes,
                "positive_votes": n.stats.positive_votes,
                "reuse_count": n.stats.reuse_count,
                "validity": n.validity.type,
                "heat": round(n.stats.heat, 4),
                "last_reinforced_at": n.stats.last_reinforced_at,
                "retention": round(ret, 4),
                "stability_factor": round(stab, 4),
                "effective_lambda": round(eff_lam, 6),
                "content": n.content,
                "description": n.description,
            }
        )
    return out


def run_sample_evaluation() -> None:
    test_data_root = repo_root / "data_storage_cross_sample_test"
    if test_data_root.exists():
        shutil.rmtree(test_data_root)

    ms = MemorySystem(data_root=test_data_root, async_mode=False)

    sample_plan = [
        (
            "mobile_persona_001",
            "USER_001",
            [
                ("老规矩，在常规周期保障现场，按‘午间例行’的时间，用抖音做开直播，帮我把保障开上。", "2026年10月22日12时20分"),
                ("老规矩，在大巴转场途中，用腾讯会议做会议，画质按‘保个基本画质’，帮我把保障开上。", "2026年10月22日18时40分"),
            ],
        ),
        (
            "mobile_persona_002",
            "USER_002",
            [
                ("老规矩，在驻地复盘业务，用阿抖做短视频，帮我把保障开上。", "2026年11月15日20时10分"),
                ("老规矩，在外勤保障现场，用微信做视频通话，时延按‘顺畅推流’，帮我把保障开上。", "2026年11月15日14时50分"),
            ],
        ),
    ]

    report: Dict[str, Any] = {}

    for persona_name, user_id, test_queries in sample_plan:
        persona_dir = CROSS_DATA_DIR / persona_name
        sessions_list = load_sessions_list(persona_dir)
        traces_map = load_traces_map(persona_dir)

        print(f"\n================ Testing {persona_name} ({user_id}) [{len(sessions_list)} sessions] ================")
        step_snapshots = []
        last_ref_ts = ""

        for sess in sessions_list:
            sid = sess["session_id"]
            meta = sess.get("session_meta", {})
            trace = traces_map.get(sid, {})
            role = trace.get("memory_role") or meta.get("memory_role", "")
            blueprint = trace.get("blueprint_type") or meta.get("scenario", {}).get("blueprint_type", "")
            action = meta.get("scenario", {}).get("memory_action", "")
            ref_time = trace.get("reference_time") or meta.get("reference_time") or ""
            last_ref_ts = normalize_ts_str(ref_time)

            # Format turns for display
            dialogue_turns = []
            for t_obj in sess.get("event_sequence", []):
                t_num = t_obj.get("turn", 1)
                u_txt = (t_obj.get("user") or {}).get("utterance", "")
                a_txt = (t_obj.get("agent") or {}).get("utterance", "")
                dialogue_turns.append({"turn": t_num, "user": u_txt, "agent": a_txt})

            gt_events = trace.get("memory_events", [])

            # If this session reuses/reinforces/corrects memory, run search on Turn 1 user query BEFORE recording session
            session_search = None
            if role in ("reinforcement_session", "reuse_session", "correction_session") and dialogue_turns:
                u1_query = dialogue_turns[0]["user"]
                hits = ms.search(
                    tenant_id="default",
                    user_id=user_id,
                    query=u1_query,
                    top_k=3,
                    levels="L0+L1",
                    now_ts=last_ref_ts,
                )
                session_search = {
                    "query": u1_query,
                    "query_time": ref_time,
                    "hits": [
                        {
                            "memory_key": h["memory_key"],
                            "subtype": h["subtype"],
                            "status": h["status"],
                            "score": round(h["score"], 4),
                            "content": h["content"],
                            "explain": h["explain"],
                            "description": h["description"],
                        }
                        for h in hits[:3]
                    ],
                }

            print(f"\n-> Ingesting {sid} @ {ref_time} ({last_ref_ts}) [role={role}, blueprint={blueprint}] ...")
            if session_search and session_search["hits"]:
                top_h = session_search["hits"][0]
                print(
                    f"   [Turn-1 Search] Q='{session_search['query']}' -> #1 [{top_h['subtype']}] {top_h['memory_key']} "
                    f"(score={top_h['score']}) | {top_h['explain']['structured_match']}"
                )

            rec_res = ms.record_session(
                tenant_id="default",
                user_id=user_id,
                agent_id="qos_assistant",
                session_id=sid,
                turns=sess["event_sequence"],
                metadata=meta,
                reference_time=ref_time,
            )

            extracted_events = rec_res.get("extracted_events", [])
            for ev in extracted_events:
                print(
                    f"   [Extracted] {ev['event_type']} -> [{ev['subtype']}] key={ev['memory_key_fields']} "
                    f"validity={ev['validity']} content={ev['content']}"
                )
            if not extracted_events:
                print("   [Extracted] (None - OOD / No memory event)")

            eb_decay = rec_res.get("ebbinghaus_decay", {})
            if eb_decay.get("expired_candidates") or eb_decay.get("degraded_reinforced"):
                print(f"   [Ebbinghaus Lifecycle] {eb_decay}")

            current_memories = dump_user_memories(ms, user_id, now_ts=last_ref_ts)
            for m in current_memories:
                print(
                    f"   * [{m['subtype']}] {m['memory_key']} | status={m['status']} (votes={m['votes']}, "
                    f"R={m['retention']}, S={m['stability_factor']}) | "
                    f"res={m['content'].get('resolution')} lat={m['content'].get('latency')} "
                    f"tw={m['content'].get('time_window')} target={m['content'].get('target')}"
                )

            step_snapshots.append(
                {
                    "session_id": sid,
                    "reference_time": ref_time,
                    "normalized_ts": last_ref_ts,
                    "memory_role": role,
                    "blueprint_type": blueprint,
                    "memory_action": action,
                    "dialogue_turns": dialogue_turns,
                    "gt_events": gt_events,
                    "session_search": session_search,
                    "extracted_events": extracted_events,
                    "ebbinghaus_decay": eb_decay,
                    "memories": current_memories,
                }
            )

        # Final Retrieval Queries
        query_evaluations = []
        for q_text, q_time in test_queries:
            q_ts = normalize_ts_str(q_time)
            print(f"\n[Final Search Test] Query='{q_text}' @ {q_time} ({q_ts}):")
            hits = ms.search(
                tenant_id="default",
                user_id=user_id,
                query=q_text,
                top_k=4,
                levels="L0+L1",
                now_ts=q_ts,
            )
            for rank, h in enumerate(hits, 1):
                exp = h["explain"]
                print(
                    f"   #{rank} [{h['subtype']}] {h['memory_key']} (status={h['status']}) -> score={h['score']:.4f} "
                    f"| sim={exp['sim']:.2f}, R={exp['freshness']:.4f} (S={exp['stability_factor']}), heat={exp['heat']:.4f} "
                    f"| {exp['structured_match']}"
                )
            query_evaluations.append(
                {
                    "query": q_text,
                    "query_time": q_time,
                    "hits": [
                        {
                            "memory_key": h["memory_key"],
                            "subtype": h["subtype"],
                            "status": h["status"],
                            "score": round(h["score"], 4),
                            "content": h["content"],
                            "explain": h["explain"],
                            "description": h["description"],
                        }
                        for h in hits[:3]
                    ],
                }
            )

        report[persona_name] = {
            "user_id": user_id,
            "session_count": len(step_snapshots),
            "time_span": f"{step_snapshots[0]['reference_time']} ~ {step_snapshots[-1]['reference_time']}",
            "steps": step_snapshots,
            "final_memories": step_snapshots[-1]["memories"] if step_snapshots else [],
            "query_evaluations": query_evaluations,
        }

    out_path = repo_root / "test" / "cross_memory_sample_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full 2-sample chronological report to {out_path}")
    ms.close()


if __name__ == "__main__":
    run_sample_evaluation()
