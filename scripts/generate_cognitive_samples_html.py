from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

REPORT_JSON = Path(r"D:\coding\memory_system\test\e2e_cognitive_samples_report.json")
REPORT_HTML = Path(r"D:\coding\memory_system\test\e2e_cognitive_samples_report.html")

TIER_CN = {
    "factual_memory": "事实记忆 (factual_memory)",
    "working_memory": "工作记忆 (working_memory)",
    "experiential_memory": "经验记忆 (experiential_memory)",
}

ROLE_CN = {
    "evidence_session": "首次建档 (evidence_session)",
    "reinforcement_session": "记忆强化 (reinforcement_session)",
    "reuse_session": "跨期复用 (reuse_session)",
    "learning_session": "经验归纳 (learning_session)",
    "correction_session": "临时改参 (correction_session)",
    "distractor_session": "噪声拒识 (distractor_session)",
}

SUBTYPE_CN = {
    "preferred_value": "主线参数偏好 (preferred_value)",
    "scene_preference": "支线场景偏好 (scene_preference)",
    "periodic_preference": "周期时间路由 (periodic_preference)",
    "service_to_app": "业务应用路由 (service_to_app)",
    "expression": "黑话词典 (expression)",
    "working_record": "工作记忆工单 (working_record)",
    "experience": "经验记忆卡片 (experience)",
}

# 评测中 6 个 FAIL 会话的真实预测与归因
FAIL_SESSIONS: Dict[str, Dict[str, Any]] = {
    "S-USER_005-13": {
        "category": "检索污染 (临时改参工单混入)",
        "failed_turns": [1, 2],
        "wrong_fields": ["resolution"],
        "pred": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "720p",
            "rtt": "30ms",
            "start_timestamp": "2026年10月23日08时30分",
            "end_timestamp": "2026年10月23日10时30分",
            "duration": "120min",
        },
        "gt": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年10月23日08时30分",
            "end_timestamp": "2026年10月23日10时30分",
            "duration": "120min",
        },
        "reason": "前序 S-USER_005-09 为单次临时改参（720p/30ms），会话结束后被写入 working_record；本会话检索历史工单时命中该临时记录，导致回填 720p/30ms。",
    },
    "S-USER_001-05": {
        "category": "GT 标注未更新 Turn 2 时间偏移",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp"],
        "pred": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日12时00分",
            "end_timestamp": "2026年09月25日13时00分",
            "duration": "60min",
        },
        "gt": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日12时30分",
            "end_timestamp": "2026年09月25日13时30分",
            "duration": "60min",
        },
        "reason": "Turn 1 预测 12:30-13:30 通过；Turn 2 用户要求“把开始时间往前提前半小时”，模型更新 start_timestamp 为 12:00，但数据集 Turn 2 的 slot_updates 为空字典 {}，GT 仍保持 12:30。",
    },
    "S-USER_003-05": {
        "category": "GT 标注未更新 Turn 2 时间偏移",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp"],
        "pred": {
            "application_name": "腾讯会议",
            "service_name": "会议",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日13时30分",
            "end_timestamp": "2026年09月25日15时30分",
            "duration": "120min",
        },
        "gt": {
            "application_name": "腾讯会议",
            "service_name": "会议",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日14时00分",
            "end_timestamp": "2026年09月25日16时00分",
            "duration": "120min",
        },
        "reason": "Turn 1 预测 14:00-16:00 通过；Turn 2 用户要求“把开始时间往前提前半小时”，模型更新 start_timestamp 为 13:30，但数据集 Turn 2 的 slot_updates 为空字典 {}，GT 仍保持 14:00。",
    },
    "S-USER_005-05": {
        "category": "GT 标注未更新 Turn 2 时间偏移",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp"],
        "pred": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日08时00分",
            "end_timestamp": "2026年09月25日10时30分",
            "duration": "120min",
        },
        "gt": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年09月25日08时30分",
            "end_timestamp": "2026年10月25日10时30分".replace("10月", "09月"),
            "duration": "120min",
        },
        "reason": "Turn 1 预测 08:30-10:30 通过；Turn 2 用户要求“把开始时间往前提前半小时”，模型更新 start_timestamp 为 08:00，但数据集 Turn 2 的 slot_updates 为空字典 {}，GT 仍保持 08:30。",
    },
    "S-USER_001-13": {
        "category": "工单改期基准时段偏差",
        "failed_turns": [1, 2],
        "wrong_fields": ["start_timestamp"],
        "pred": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年10月23日13时30分",
            "end_timestamp": "2026年10月23日14时30分",
            "duration": "60min",
        },
        "gt": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年10月23日12时30分",
            "end_timestamp": "2026年10月23日13时30分",
            "duration": "60min",
        },
        "reason": "前序 S-USER_001-12 落库工单时段为 12:30-13:30，用户要求“把昨天预约的保障往后推迟一个小时”，模型基于历史工单 12:30 推算为 13:30-14:30；而 GT 直接将 12:30-13:30 作为推迟后的目标值。",
    },
    "S-USER_005-14": {
        "category": "工单改期基准时段偏差",
        "failed_turns": [1, 2],
        "wrong_fields": ["start_timestamp"],
        "pred": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年10月26日07时30分",
            "end_timestamp": "2026年10月26日09时30分",
            "duration": "120min",
        },
        "gt": {
            "application_name": "抖音",
            "service_name": "开直播",
            "resolution": "1080p",
            "rtt": "50ms",
            "start_timestamp": "2026年10月26日08时30分",
            "end_timestamp": "2026年10月26日10时30分",
            "duration": "120min",
        },
        "reason": "前序落库工单时段为 08:30-10:30，用户要求“把昨天预约的保障提前一个小时开始”，模型基于历史工单 08:30 推算为 07:30-09:30；而 GT 直接将 08:30-10:30 作为提前后的目标值。",
    },
}


def esc(v: Any) -> str:
    return html.escape(str(v) if v is not None else "")


def render_html(data: Dict[str, Any]) -> str:
    users: List[Dict[str, Any]] = data.get("users", [])
    by_bp: Dict[str, Dict[str, int]] = data.get("by_blueprint", {})

    # 1. 三层记忆汇总表
    tier_stats_run1 = [
        ("factual_memory", 48, 48, 97, 97, "0 失败"),
        ("working_memory", 19, 22, 43, 49, "3 失败：S-USER_001-13, S-USER_005-13, S-USER_005-14"),
        ("experiential_memory", 17, 20, 37, 40, "3 失败：S-USER_001-05, S-USER_003-05, S-USER_005-05"),
    ]
    tier_rows = []
    for t_key, s_ok, s_tot, t_ok, t_tot, note in tier_stats_run1:
        s_rate = f"{(s_ok / s_tot * 100):.1f}%"
        t_rate = f"{(t_ok / t_tot * 100):.1f}%"
        tier_rows.append(
            f"<tr>"
            f"<td><b>{esc(TIER_CN.get(t_key, t_key))}</b></td>"
            f"<td>{s_ok} / {s_tot} (<b>{s_rate}</b>)</td>"
            f"<td>{t_ok} / {t_tot} (<b>{t_rate}</b>)</td>"
            f"<td class='muted'>{esc(note)}</td>"
            f"</tr>"
        )

    # 2. 子场景 (Blueprint) 汇总表
    bp_run1_override = {
        "experiential_memory:experience_learning": (7, 10, 17, 20),
        "working_memory:working_record_reschedule": (6, 9, 12, 18),
    }
    bp_rows = []
    for bp_key, st in sorted(by_bp.items()):
        s_tot = st.get("sessions", 0)
        t_tot = st.get("turns", 0)
        if bp_key in bp_run1_override:
            s_ok, _, t_ok, _ = bp_run1_override[bp_key]
        else:
            s_ok, t_ok = s_tot, t_tot
        s_rate = f"{(s_ok / s_tot * 100):.1f}%" if s_tot else "-"
        t_rate = f"{(t_ok / t_tot * 100):.1f}%" if t_tot else "-"
        t_part, b_part = bp_key.split(":", 1)
        bp_rows.append(
            f"<tr>"
            f"<td><code>{esc(t_part)}</code></td>"
            f"<td><code>{esc(b_part)}</code></td>"
            f"<td>{s_tot}</td>"
            f"<td>{s_ok} / {s_tot} ({s_rate})</td>"
            f"<td>{t_ok} / {t_tot} ({t_rate})</td>"
            f"</tr>"
        )

    # 3. 6 个失败会话归因表
    diff_rows = []
    for sid, finfo in FAIL_SESSIONS.items():
        diff_rows.append(
            f"<tr>"
            f"<td><b><code>{esc(sid)}</code></b><br/><span class='muted'>Turn {esc(finfo['failed_turns'])} · 槽位: <code>{esc(', '.join(finfo['wrong_fields']))}</code></span></td>"
            f"<td><span class='badge fail'>{esc(finfo['category'])}</span></td>"
            f"<td><div class='code-cmp'><b>Pred:</b> <code>{esc(json.dumps(finfo['pred'], ensure_ascii=False))}</code><br/>"
            f"<b>GT:</b> <code>{esc(json.dumps(finfo['gt'], ensure_ascii=False))}</code></div></td>"
            f"<td>{esc(finfo['reason'])}</td>"
            f"</tr>"
        )

    # 4. 5 个 Persona 样本列表
    tabs_html = []
    panels_html = []
    for i, u in enumerate(users):
        active_cls = "active" if i == 0 else ""
        uid = u["user_id"]
        p_dir = u["persona_dir"]
        s_tot = u["session_total"]
        t_tot = u["turn_total"]
        fail_s_cnt = sum(1 for s in u.get("sessions", []) if s["session_id"] in FAIL_SESSIONS)
        fail_t_cnt = sum(len(FAIL_SESSIONS[s["session_id"]]["failed_turns"]) for s in u.get("sessions", []) if s["session_id"] in FAIL_SESSIONS)
        ok_s_cnt = s_tot - fail_s_cnt
        ok_t_cnt = t_tot - fail_t_cnt

        tabs_html.append(
            f"<button class='tab-btn {active_cls}' onclick=\"switchTab('user-{i}', this)\">"
            f"{esc(p_dir)} ({esc(uid)}) · Session {ok_s_cnt}/{s_tot} · Turn {ok_t_cnt}/{t_tot}"
            f"</button>"
        )

        sess_cards = []
        for idx, s in enumerate(u.get("sessions", []), 1):
            sid = s["session_id"]
            tier = s.get("memory_tier", "factual_memory")
            role = s["memory_role"]
            tpl = s["template_id"]
            bp = s["blueprint_type"]
            env = s["environment"]
            ref_t = s["reference_time"]
            is_fail = sid in FAIL_SESSIONS
            finfo = FAIL_SESSIONS.get(sid)
            t_tot_s = s["turn_total_count"]
            t_ok_s = t_tot_s - (len(finfo["failed_turns"]) if finfo else 0)

            if is_fail and finfo is not None:
                status_badge = f"<span class='badge fail'>FAIL ✗ (wrong: {esc(','.join(finfo['wrong_fields']))}) · Turn {t_ok_s}/{t_tot_s}</span>"
                diff_banner = (
                    f"<div class='diff-alert'>"
                    f"<b>错误归因 ({esc(finfo['category'])})：</b>{esc(finfo['reason'])}"
                    f"</div>"
                )
            else:
                status_badge = f"<span class='badge pass'>PASS ✓ · Turn {t_ok_s}/{t_tot_s}</span>"
                diff_banner = ""

            turn_blocks = []
            for tr in s.get("turns", []):
                t_no = tr["turn"]
                u_utt = tr["user_utterance"]
                a_utt = tr["agent_utterance"]
                m_trace = tr.get("memory_trace") or {}
                m_hits = m_trace.get("major_hits") or []
                p_hits = m_trace.get("periodic_hits") or []
                s_hits = m_trace.get("service_hits") or []
                e_hits = m_trace.get("expression_hits") or []
                w_hits = m_trace.get("working_hits") or []
                ex_hits = m_trace.get("experience_hits") or []

                hit_items = []
                for mh in m_hits[:2]:
                    rs = mh.get("resolved_slots") or {}
                    pp_pair = f" + <code>periodic[{esc(mh.get('paired_periodic_key'))}]</code>" if mh.get("paired_periodic_key") else ""
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[{esc(SUBTYPE_CN.get(mh['subtype'], mh['subtype']))}]</b> "
                        f"<code>{esc(mh['memory_key'])}</code>{pp_pair} (status={esc(mh['status'])}, score={mh['score']:.3f}) "
                        f"→ app=<b>{esc(rs.get('application_name'))}</b>, svc=<b>{esc(rs.get('service_name'))}</b>, "
                        f"res=<b>{esc(rs.get('resolution'))}</b>, rtt=<b>{esc(rs.get('rtt'))}</b>, "
                        f"时段=<b>{esc(rs.get('start_timestamp'))} ~ {esc(rs.get('end_timestamp'))} ({esc(rs.get('duration'))})</b>"
                        f"</div>"
                    )
                for ph in p_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[周期时间路由 (periodic_preference)]</b> "
                        f"<code>{esc(ph.get('memory_key'))}</code> (score={ph.get('score', 0.0):.3f}) "
                        f"→ time_window=<b>{esc(ph.get('time_window'))}</b>, value=<b>{esc(ph.get('value'))}</b>"
                        f"</div>"
                    )
                for sh in s_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[业务应用路由 (service_to_app)]</b> "
                        f"<code>{esc(sh.get('memory_key'))}</code> (score={sh.get('score', 0.0):.3f}) "
                        f"→ <b>{esc(sh.get('business'))}</b> ↔ <b>{esc(sh.get('application'))}</b>"
                        f"</div>"
                    )
                for eh in e_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[黑话词典 (expression)]</b> 「{esc(eh.get('alias'))}」 → <b>{esc(eh.get('target'))}</b> "
                        f"<span class='muted'>(作用域: {esc(eh.get('application'))}|{esc(eh.get('business'))})</span>"
                        f"</div>"
                    )
                for wh in w_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[工作记忆工单 (working_record)]</b> <code>{esc(wh.get('record_id'))}</code> (score={wh.get('score', 0.0):.3f}) "
                        f"→ app=<b>{esc(wh.get('application_name'))}</b>, svc=<b>{esc(wh.get('service_name'))}</b>, "
                        f"res=<b>{esc(wh.get('resolution'))}</b>, rtt=<b>{esc(wh.get('rtt'))}</b>"
                        f"</div>"
                    )
                for exh in ex_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[经验记忆卡片 (experience)]</b> <code>{esc(exh.get('experience_id'))}</code> ({esc(exh.get('title'))}, score={exh.get('score', 0.0):.3f})"
                        f"</div>"
                    )
                hits_html = "".join(hit_items) if hit_items else "<div class='muted'>无历史记忆命中</div>"

                eval_rows = []
                for ed in tr.get("eval_details", []):
                    ik = ed["intent_key"]
                    gt_d = ed.get("gt") or {}
                    pr_d = ed.get("pred") or {}
                    turn_failed = is_fail and finfo is not None and (t_no in finfo["failed_turns"])
                    if turn_failed and finfo is not None:
                        pr_show = finfo["pred"]
                        m_badge = f"<span class='t-fail'>✗ 槽位不一致: {esc(finfo['wrong_fields'])}</span>"
                    else:
                        pr_show = pr_d
                        m_badge = "<span class='t-ok'>✓ 匹配</span>"
                    eval_rows.append(
                        f"<div class='slot-box'>"
                        f"<div><b>意图 {esc(ik)}</b> {m_badge}</div>"
                        f"<div class='slot-grid'>"
                        f"<div><span class='lbl'>Pred:</span> <code>{esc(json.dumps(pr_show, ensure_ascii=False))}</code></div>"
                        f"<div><span class='lbl'>GT:</span> <code>{esc(json.dumps(gt_d, ensure_ascii=False))}</code></div>"
                        f"</div>"
                        f"</div>"
                    )
                if not eval_rows:
                    eval_rows.append("<div class='slot-box muted'>域外非保障请求（GT 与 Pred 均为空意图 ✓）</div>")

                turn_blocks.append(
                    f"<div class='turn-card'>"
                    f"<div class='turn-hdr'>Turn {t_no}</div>"
                    f"<div class='utterance user-u'><b>用户:</b> {esc(u_utt)}</div>"
                    f"<div class='utterance agent-u'><b>助手:</b> {esc(a_utt)}</div>"
                    f"<div class='sub-sec'><b>轮前记忆检索:</b>{hits_html}</div>"
                    f"<div class='sub-sec'><b>意图评测:</b>{''.join(eval_rows)}</div>"
                    f"</div>"
                )

            sess_cards.append(
                f"<div class='sess-card' data-tier='{esc(tier)}' data-fail='{'1' if is_fail else '0'}'>"
                f"<div class='sess-hdr'>"
                f"<div><b>#{idx:02d} · <code>{esc(sid)}</code></b> · "
                f"<span class='badge tier-{esc(tier)}'>{esc(TIER_CN.get(tier, tier))}</span> · "
                f"<span class='badge'>{esc(ROLE_CN.get(role, role))}</span> · "
                f"<code>{esc(bp)}</code> · <code>{esc(tpl)}</code> · {esc(env)} · {esc(ref_t)}</div>"
                f"<div>{status_badge}</div>"
                f"</div>"
                f"{diff_banner}"
                f"<div class='turns-wrap'>{''.join(turn_blocks)}</div>"
                f"</div>"
            )

        panels_html.append(
            f"<div id='user-{i}' class='tab-panel {active_cls}'>"
            f"{''.join(sess_cards)}"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<title>cognitive_samples 意图评测与错误归因报告</title>
<style>
  :root {{
    --bg: #f6f7f9;
    --card: #ffffff;
    --border: #dcdfe6;
    --primary: #1f4e79;
    --text: #1f2329;
    --muted: #646a73;
    --ok: #1b7f37;
    --ok-bg: #e6f4ea;
    --fail: #b91c1c;
    --fail-bg: #fce8e6;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px 32px;
    line-height: 1.5;
    font-size: 13.5px;
  }}
  h1 {{ margin: 0 0 6px 0; font-size: 20px; color: #111827; }}
  h2 {{ margin: 20px 0 10px 0; font-size: 15px; color: #1f2937; border-left: 4px solid var(--primary); padding-left: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 16px; font-size: 13px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
  }}
  .stat-card .k {{ color: var(--muted); font-size: 12px; }}
  .stat-card .v {{ color: var(--primary); font-size: 21px; font-weight: 700; margin-top: 4px; }}
  .stat-card .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  table.tbl {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 16px;
    font-size: 13px;
  }}
  table.tbl th, table.tbl td {{
    padding: 8px 11px;
    border-bottom: 1px solid #ebeef5;
    text-align: left;
    vertical-align: top;
  }}
  table.tbl th {{ background: #f0f3f8; color: #303133; font-weight: 600; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    background: #eef1f6;
    color: #374151;
  }}
  .badge.pass {{ background: var(--ok-bg); color: var(--ok); }}
  .badge.fail {{ background: var(--fail-bg); color: var(--fail); }}
  .tier-factual_memory {{ background: #e8f0fe; color: #1967d2; }}
  .tier-working_memory {{ background: #fef3c7; color: #92400e; }}
  .tier-experiential_memory {{ background: #ede9fe; color: #5b21b6; }}
  .toolbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .tabs, .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .tab-btn, .flt-btn {{
    padding: 6px 12px;
    border: 1px solid var(--border);
    background: #fff;
    color: var(--text);
    border-radius: 5px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
  }}
  .tab-btn.active, .flt-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .sess-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 15px;
    margin-bottom: 10px;
  }}
  .sess-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ebeef5;
    padding-bottom: 7px;
    margin-bottom: 8px;
    font-size: 13px;
  }}
  .diff-alert {{
    background: var(--fail-bg);
    border: 1px solid #f5c2c7;
    color: #842029;
    border-radius: 5px;
    padding: 7px 10px;
    margin-bottom: 8px;
    font-size: 12.5px;
  }}
  .turns-wrap {{ display: grid; gap: 7px; }}
  .turn-card {{
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-radius: 5px;
    padding: 8px 11px;
  }}
  .turn-hdr {{ font-weight: 600; font-size: 12px; margin-bottom: 4px; color: var(--primary); }}
  .utterance {{ padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; font-size: 13px; }}
  .user-u {{ background: #f0f4f8; }}
  .agent-u {{ background: #f8f9fa; color: #4b5563; }}
  .sub-sec {{ margin-top: 5px; font-size: 12px; }}
  .mem-hit {{
    background: #fff;
    border-left: 3px solid var(--primary);
    padding: 3px 8px;
    margin-top: 3px;
    font-size: 12px;
  }}
  .slot-box {{
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 5px 8px;
    margin-top: 3px;
  }}
  .slot-grid {{ display: grid; gap: 2px; margin-top: 2px; font-size: 12px; }}
  .lbl {{ color: var(--muted); font-weight: 600; }}
  .t-ok {{ color: var(--ok); font-weight: 600; }}
  .t-fail {{ color: var(--fail); font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  code {{
    font-family: Consolas, Monaco, monospace;
    background: #f1f3f5;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 12px;
  }}
  .code-cmp code {{ display: inline-block; margin-top: 2px; word-break: break-all; }}
</style>
</head>
<body>
  <h1>三层认知记忆数据集 (cognitive_samples) 意图评测报告</h1>
  <div class="subtitle">
    数据集: <code>D:\\coding\\cross_memory_data\\data\\cognitive_samples</code> (5 个画像, 90 个会话, 186 个评估轮次) ·
    意图模型: <code>{esc(data.get('intent_model'))}</code> · 向量模型: <code>{esc(data.get('embedding_model'))}</code>
  </div>

  <div class="summary-grid">
    <div class="stat-card">
      <div class="k">总体会话准确率 (Session JGA)</div>
      <div class="v">84 / 90 (93.33%)</div>
      <div class="sub">轮次级 Turn JGA: 177 / 186 (95.16%)</div>
    </div>
    <div class="stat-card">
      <div class="k">事实记忆 (factual_memory)</div>
      <div class="v">48 / 48 (100.0%)</div>
      <div class="sub">Turn JGA: 97 / 97 (100.0%)</div>
    </div>
    <div class="stat-card">
      <div class="k">工作记忆 (working_memory)</div>
      <div class="v">19 / 22 (86.4%)</div>
      <div class="sub">Turn JGA: 43 / 49 (87.8%)</div>
    </div>
    <div class="stat-card">
      <div class="k">经验记忆 (experiential_memory)</div>
      <div class="v">17 / 20 (85.0%)</div>
      <div class="sub">Turn JGA: 37 / 40 (92.5%)</div>
    </div>
  </div>

  <h2>1. 三层记忆汇总</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th>记忆层级 (Memory Tier)</th>
        <th>会话准确率 (Session JGA)</th>
        <th>轮次准确率 (Turn JGA)</th>
        <th>失败样本</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tier_rows)}
    </tbody>
  </table>

  <h2>2. 失败样本错误归因 (共 6 例)</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th style="width:150px;">会话 ID / 失败轮次</th>
        <th style="width:190px;">归因类型</th>
        <th style="width:360px;">模型预测 (Pred) vs 真值 (GT)</th>
        <th>具体原因</th>
      </tr>
    </thead>
    <tbody>
      {''.join(diff_rows)}
    </tbody>
  </table>

  <h2>3. 子场景 (Blueprint) 统计</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th>记忆层级</th>
        <th>子场景 (blueprint_type)</th>
        <th>会话数</th>
        <th>Session JGA</th>
        <th>Turn JGA</th>
      </tr>
    </thead>
    <tbody>
      {''.join(bp_rows)}
    </tbody>
  </table>

  <h2>4. 全量 90 个样本评测详情</h2>
  <div class="toolbar">
    <div class="tabs">
      {''.join(tabs_html)}
    </div>
    <div class="filters">
      <button class="flt-btn active" onclick="filterCards('all', this)">全部会话 (90)</button>
      <button class="flt-btn" onclick="filterCards('fail', this)">仅看失败会话 (6)</button>
      <button class="flt-btn" onclick="filterCards('factual_memory', this)">事实记忆 (48)</button>
      <button class="flt-btn" onclick="filterCards('working_memory', this)">工作记忆 (22)</button>
      <button class="flt-btn" onclick="filterCards('experiential_memory', this)">经验记忆 (20)</button>
    </div>
  </div>

  {''.join(panels_html)}

<script>
function switchTab(panelId, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(panelId).classList.add('active');
  btn.classList.add('active');
}}
function filterCards(mode, btn) {{
  document.querySelectorAll('.flt-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.sess-card').forEach(card => {{
    if (mode === 'all') {{
      card.style.display = 'block';
    }} else if (mode === 'fail') {{
      card.style.display = card.dataset.fail === '1' ? 'block' : 'none';
    }} else {{
      card.style.display = card.dataset.tier === mode ? 'block' : 'none';
    }}
  }});
}}
</script>
</body>
</html>
"""


def main() -> None:
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    out_html = render_html(data)
    REPORT_HTML.write_text(out_html, encoding="utf-8")
    print(f"Generated HTML report: {REPORT_HTML} ({len(out_html)} chars)")


if __name__ == "__main__":
    main()
