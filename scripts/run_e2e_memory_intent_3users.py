from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

QWEN_INTENT_DIR = Path(r"D:\coding\qwen_intent_rag")
MEMORY_SYSTEM_DIR = Path(r"D:\coding\memory_system")
CROSS_DATA_DIR = Path(r"D:\coding\cross_memory_data\data\generated")

sys.path.insert(0, str(QWEN_INTENT_DIR))
sys.path.insert(0, str(MEMORY_SYSTEM_DIR / "src"))

from gt_build import turn_gt
from memory_bridge import MemoryBridge
from pipeline import MultiIntentPipeline, load_config
from session_eval import judge_session, judge_turn, run_session, session_gt


def load_persona_info(persona_dir: Path) -> Dict[str, Any]:
    p_file = persona_dir / "persona.json"
    if p_file.exists():
        return json.loads(p_file.read_text(encoding="utf-8"))
    return {"persona_id": persona_dir.name}


def load_sessions(persona_dir: Path) -> List[Dict[str, Any]]:
    s_json = persona_dir / "sessions.json"
    if s_json.exists():
        return json.loads(s_json.read_text(encoding="utf-8"))
    s_jsonl = persona_dir / "sessions.jsonl"
    out = []
    for line in s_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def select_random_personas(k: int = 3, seed: int = 20261026) -> List[Path]:
    all_dirs = sorted(
        [
            d
            for d in CROSS_DATA_DIR.iterdir()
            if d.is_dir()
            and d.name.startswith("mobile_persona_")
            and d.name not in ("mobile_persona_001", "mobile_persona_002")
        ]
    )
    rng = random.Random(seed)
    return sorted(rng.sample(all_dirs, k), key=lambda p: p.name)


def run_evaluation(selected_dirs: List[Path], work_data_root: Path) -> Dict[str, Any]:
    if work_data_root.exists():
        shutil.rmtree(work_data_root)
    work_data_root.mkdir(parents=True, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(str(QWEN_INTENT_DIR))
    try:
        cfg_path = QWEN_INTENT_DIR / "configs" / "config_memory_qwen38_flash.json"
        cfg = load_config(str(cfg_path))
        pipe = MultiIntentPipeline(cfg)
        bridge = MemoryBridge(data_root=work_data_root)
        pipe.memory_bridge = bridge  # type: ignore[attr-defined]
        mi_on = pipe.features.multi_instance_key

        all_user_reports: List[Dict[str, Any]] = []
        total_sessions = 0
        total_session_ok = 0
        total_turns = 0
        total_turn_ok = 0
        role_stats: Dict[str, Dict[str, int]] = {}

        for p_dir in selected_dirs:
            persona = load_persona_info(p_dir)
            sessions = load_sessions(p_dir)
            uid = sessions[0]["user_id"] if sessions else p_dir.name
            print(f"\n============================================================")
            print(f"Running E2E Memory + Intent Understanding for {p_dir.name} ({uid}) | {len(sessions)} sessions")
            print(f"============================================================")

            user_session_records: List[Dict[str, Any]] = []
            u_sess_ok = 0
            u_turn_total = 0
            u_turn_ok = 0

            for idx, sess in enumerate(sessions, 1):
                sid = sess["session_id"]
                ref_time = sess.get("reference_time", "")
                meta = sess.get("session_meta") or {}
                role = meta.get("memory_role", "unknown")
                tpl = meta.get("template_id", "")
                scenario = meta.get("scenario") or {}
                env_name = scenario.get("environment", "")
                bp_type = scenario.get("blueprint_type", "")

                t0 = time.time()
                session_out, per_turn_pred = run_session(pipe, sess, retrieve_on=True)
                s_gt = session_gt(sess, mi_on=mi_on)
                s_ok, s_diffs = judge_session(s_gt, session_out)

                t_gts = turn_gt(sess, cancel_credit=True, correction_credit=True, gt_derive=True, mi_on=mi_on)
                t_ok_cnt, t_tot_cnt, t_details = judge_turn(t_gts, per_turn_pred)

                # Session 结束后写入 MemorySystem
                mem_after = bridge.record_session(sess)
                elapsed = round(time.time() - t0, 2)

                total_sessions += 1
                if s_ok:
                    total_session_ok += 1
                    u_sess_ok += 1
                total_turns += t_tot_cnt
                total_turn_ok += t_ok_cnt
                u_turn_total += t_tot_cnt
                u_turn_ok += t_ok_cnt

                r_stat = role_stats.setdefault(role, {"sessions": 0, "session_ok": 0, "turns": 0, "turn_ok": 0})
                r_stat["sessions"] += 1
                if s_ok:
                    r_stat["session_ok"] += 1
                r_stat["turns"] += t_tot_cnt
                r_stat["turn_ok"] += t_ok_cnt

                status_str = "PASS" if s_ok else "FAIL"
                print(
                    f"  [{idx:02d}/{len(sessions):02d}] {sid} ({tpl} | {role} | {bp_type}:{env_name}) "
                    f"-> Session={status_str}, Turn={t_ok_cnt}/{t_tot_cnt} ({elapsed}s)"
                )
                if not s_ok:
                    print(f"       Session Diffs: {s_diffs}")
                    print(f"       GT  : {json.dumps(s_gt, ensure_ascii=False)}")
                    print(f"       Pred: {json.dumps({k: v.get('pred_flat') for k, v in session_out.items()}, ensure_ascii=False)}")
                for td in t_details:
                    if not td["match"]:
                        print(
                            f"       Turn {td['turn']} ({td['intent_key']}) Wrong={td['wrong']} | "
                            f"GT={json.dumps(td.get('gt'), ensure_ascii=False)} | "
                            f"Pred={json.dumps(td.get('pred'), ensure_ascii=False)}"
                        )

                # 组装逐轮详情
                turn_records = []
                ev_map = {e.get("turn", 0): e for e in (sess.get("event_sequence") or [])}
                for ptp in per_turn_pred:
                    t_no = ptp["turn"]
                    ev_obj = ev_map.get(t_no, {})
                    u_utt = (ev_obj.get("user") or {}).get("utterance", "")
                    a_utt = (ev_obj.get("agent") or {}).get("utterance", "")
                    t_det_for_turn = [d for d in t_details if d["turn"] == t_no]
                    turn_records.append(
                        {
                            "turn": t_no,
                            "user_utterance": u_utt,
                            "agent_utterance": a_utt,
                            "memory_trace": ptp.get("memory_trace"),
                            "llm_raw": ptp.get("llm_raw", ""),
                            "preds": ptp.get("preds", {}),
                            "eval_details": t_det_for_turn,
                            "turn_pass": all(d["match"] for d in t_det_for_turn) if t_det_for_turn else True,
                        }
                    )

                user_session_records.append(
                    {
                        "session_id": sid,
                        "reference_time": ref_time,
                        "template_id": tpl,
                        "memory_role": role,
                        "blueprint_type": bp_type,
                        "environment": env_name,
                        "memory_action": scenario.get("memory_action", ""),
                        "session_pass": s_ok,
                        "session_gt": s_gt,
                        "session_pred": {
                            k: v.get("pred_flat")
                            for k, v in session_out.items()
                            if (v.get("pred_flat") or {}).get("application_name")
                        },
                        "session_diffs": s_diffs,
                        "turn_ok_count": t_ok_cnt,
                        "turn_total_count": t_tot_cnt,
                        "turns": turn_records,
                        "extracted_events": (mem_after.get("write_result") or {}).get("extracted_events", []),
                        "ebbinghaus_decay": (mem_after.get("write_result") or {}).get("ebbinghaus_decay", {}),
                        "memory_snapshot": mem_after.get("memory_snapshot", []),
                        "elapsed_sec": elapsed,
                    }
                )

            all_user_reports.append(
                {
                    "persona_dir": p_dir.name,
                    "user_id": uid,
                    "persona": persona,
                    "session_total": len(sessions),
                    "session_ok": u_sess_ok,
                    "session_jga": round(u_sess_ok / max(1, len(sessions)), 4),
                    "turn_total": u_turn_total,
                    "turn_ok": u_turn_ok,
                    "turn_jga": round(u_turn_ok / max(1, u_turn_total), 4),
                    "sessions": user_session_records,
                }
            )

        summary = {
            "intent_model": cfg["llm"]["model"],
            "memory_model": bridge.ms.config.llm["model"],
            "embedding_model": cfg["embedder"]["model"],
            "random_seed": 20261026,
            "selected_personas": [d.name for d in selected_dirs],
            "total_sessions": total_sessions,
            "total_session_ok": total_session_ok,
            "session_jga": round(total_session_ok / max(1, total_sessions), 4),
            "total_turns": total_turns,
            "total_turn_ok": total_turn_ok,
            "turn_jga": round(total_turn_ok / max(1, total_turns), 4),
            "by_memory_role": role_stats,
            "users": all_user_reports,
        }
        return summary
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    selected = select_random_personas(k=3, seed=20261026)
    print("Selected 3 random personas:", [p.name for p in selected])
    work_root = MEMORY_SYSTEM_DIR / "test" / "e2e_3users_data_root"
    report = run_evaluation(selected, work_root)
    out_json = MEMORY_SYSTEM_DIR / "test" / "e2e_memory_intent_3users_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n============================================================")
    print(
        f"OVERALL SUMMARY: Session JGA = {report['total_session_ok']}/{report['total_sessions']} "
        f"({report['session_jga']*100:.1f}%), Turn JGA = {report['total_turn_ok']}/{report['total_turns']} "
        f"({report['turn_jga']*100:.1f}%)"
    )
    print(f"Saved JSON report to: {out_json}")
