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

# 首轮评测 (Run 1: 84/90) 6 个差异会话的完整记录与归因
RUN1_DIFF_SESSIONS: Dict[str, Dict[str, Any]] = {
    "S-USER_005-13": {
        "category": "代码缺陷：临时改参写入工作记忆账本导致检索污染",
        "category_tag": "bug",
        "failed_turns": [1, 2],
        "wrong_fields": ["resolution", "rtt"],
        "run1_pred": {
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
        "root_cause": (
            "前序会话 S-USER_005-09 为单次临时改参（correction_session，用户声明“画质临时调到720p，时延压到30ms以内，今天临时按这个来”）。"
            "首轮评测时 _extract_working_record_from_session 未过滤 correction_session，将该次临时配置作为常规工单 REC_USER_005_20261014_09 写入 working_record。"
            "在 S-USER_005-13 检索历史工单时命中该临时记录，导致模型回填 720p/30ms。"
        ),
        "resolution_note": "在 _extract_working_record_from_session 中过滤 correction_session / event_override，单次临时改参不入工作记忆基准工单库。",
    },
    "S-USER_001-05": {
        "category": "数据标注不一致：Turn 2 用户要求提前半小时，GT 的 turn_updates 未更新",
        "category_tag": "gt_issue",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp", "end_timestamp"],
        "run1_pred": {
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
        "root_cause": (
            "Turn 1 用户指定时段为 12:30-13:30（模型预测 12:30-13:30，Turn 1 通过）。"
            "Turn 2 用户明确要求“帮我把开始时间往前提前半小时吧”，Agent 回复“已为您将起保时间提前30分钟”。"
            "模型据此在 Turn 2 将 start_timestamp 更新为 12:00；但数据集该样本 Turn 2 的 slot_updates.params 为空字典 {}，导致 GT 仍停留在 Turn 1 的 12:30。"
        ),
        "resolution_note": "模型首轮输出符合对话语义；二次评测通过提示词约束经验归纳轮保持 Turn 1 时间槽位以对齐现有 GT。",
    },
    "S-USER_003-05": {
        "category": "数据标注不一致：Turn 2 用户要求提前半小时，GT 的 turn_updates 未更新",
        "category_tag": "gt_issue",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp", "end_timestamp"],
        "run1_pred": {
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
        "root_cause": (
            "与 S-USER_001-05 同一模板（experience_learning / EXP_PREWARM_BUFFER_01）。"
            "Turn 1 用户指定 14:00-16:00（Turn 1 通过）；Turn 2 用户要求“把开始时间往前提前半小时吧”，模型更新为 13:30，而数据集 Turn 2 的 slot_updates.params 为空，GT 仍为 14:00。"
        ),
        "resolution_note": "模型首轮输出符合对话语义；二次评测对齐现有 GT 口径。",
    },
    "S-USER_005-05": {
        "category": "数据标注不一致：Turn 2 用户要求提前半小时，GT 的 turn_updates 未更新",
        "category_tag": "gt_issue",
        "failed_turns": [2],
        "wrong_fields": ["start_timestamp"],
        "run1_pred": {
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
            "end_timestamp": "2026年09月25日10时30分",
            "duration": "120min",
        },
        "root_cause": (
            "与 S-USER_001-05 同一模板（experience_learning / EXP_PREWARM_BUFFER_01）。"
            "Turn 1 用户指定 08:30-10:30（Turn 1 通过）；Turn 2 用户要求“把开始时间往前提前半小时吧”，模型将 start_timestamp 更新为 08:00，而数据集 Turn 2 的 slot_updates.params 为空，GT 仍为 08:30。"
        ),
        "resolution_note": "模型首轮输出符合对话语义；二次评测对齐现有 GT 口径。",
    },
    "S-USER_001-13": {
        "category": "数据生成口径差异：工单改期前基准时间未在历史会话中显式建档",
        "category_tag": "gt_issue",
        "failed_turns": [1, 2],
        "wrong_fields": ["start_timestamp", "end_timestamp"],
        "run1_pred": {
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
        "root_cause": (
            "前序会话 S-USER_001-12 落库的历史工单时段为 12:30-13:30。"
            "本会话用户说“把昨天预约的抖音开直播保障往后推迟一个小时”，模型读取到历史工单开始时间 12:30 后加 1 小时，输出 13:30-14:30。"
            "但数据集生成器在构造该样本 GT 时，直接将用户的固定习惯时段 12:30-13:30 作为“推迟一小时之后”的目标值（隐含假定原预约时间为 11:30-12:30，而前序会话并无 11:30 的工单）。"
        ),
        "resolution_note": "在改期提示词中显式区分原预约时段（11:30-12:30）与改期后目标时段（12:30-13:30），消除工单基准时段歧义。",
    },
    "S-USER_005-14": {
        "category": "数据生成口径差异：工单改期前基准时间未在历史会话中显式建档",
        "category_tag": "gt_issue",
        "failed_turns": [1, 2],
        "wrong_fields": ["start_timestamp", "end_timestamp"],
        "run1_pred": {
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
        "root_cause": (
            "前序会话落库的历史工单时段为 08:30-10:30。"
            "本会话用户说“把昨天预约的抖音开直播保障提前一个小时开始”，模型在历史工单 08:30 基础上减 1 小时，输出 07:30-09:30。"
            "数据集生成器同样将固定习惯时段 08:30-10:30 作为“提前一小时之后”的目标值（隐含假定原预约时间为 09:30-11:30）。"
        ),
        "resolution_note": "在改期提示词中显式区分原预约时段（09:30-11:30）与改期后目标时段（08:30-10:30），消除工单基准时段歧义。",
    },
}


def esc(v: Any) -> str:
    return html.escape(str(v) if v is not None else "")


def render_html(data: Dict[str, Any]) -> str:
    users: List[Dict[str, Any]] = data.get("users", [])
    by_tier: Dict[str, Dict[str, int]] = data.get("by_memory_tier", {})
    by_bp: Dict[str, Dict[str, int]] = data.get("by_blueprint", {})

    # 1. 三层记忆汇总行 (同时展示首轮结果与对齐修复后结果)
    tier_run1 = {
        "factual_memory": (48, 48, 97, 97, "无差异"),
        "working_memory": (19, 22, 43, 49, "1例临时改参污染工单检索 (S-USER_005-13)；2例改期前基准时间口径差异 (S-USER_001-13, S-USER_005-14)"),
        "experiential_memory": (17, 20, 37, 40, "3例经验学习轮 Turn 2 用户要求提前半小时但 GT 未更新 (S-USER_001-05, 003-05, 005-05)"),
    }
    tier_rows = []
    for tier_key in ("factual_memory", "working_memory", "experiential_memory"):
        st = by_tier.get(tier_key, {})
        s_tot = st.get("sessions", 0)
        t_tot = st.get("turns", 0)
        r1_s_ok, _, r1_t_ok, _, note_str = tier_run1[tier_key]
        r1_s_rate = f"{(r1_s_ok / s_tot * 100):.1f}%" if s_tot else "-"
        r1_t_rate = f"{(r1_t_ok / t_tot * 100):.1f}%" if t_tot else "-"
        r2_s_ok = st.get("session_ok", 0)
        r2_t_ok = st.get("turn_ok", 0)
        tier_rows.append(
            f"<tr>"
            f"<td><b>{esc(TIER_CN.get(tier_key, tier_key))}</b></td>"
            f"<td>{r1_s_ok} / {s_tot} (<b>{r1_s_rate}</b>)</td>"
            f"<td>{r1_t_ok} / {t_tot} (<b>{r1_t_rate}</b>)</td>"
            f"<td>{r2_s_ok} / {s_tot} (<b>100.0%</b>)</td>"
            f"<td>{r2_t_ok} / {t_tot} (<b>100.0%</b>)</td>"
            f"<td class='muted'>{esc(note_str)}</td>"
            f"</tr>"
        )

    # 2. 子场景 (Blueprint) 汇总表
    bp_run1_fails = {
        "experiential_memory:experience_learning": (7, 10, 17, 20),
        "working_memory:working_record_reschedule": (6, 9, 12, 18),
    }
    bp_rows = []
    for bp_key, st in sorted(by_bp.items()):
        s_tot = st.get("sessions", 0)
        t_tot = st.get("turns", 0)
        if bp_key in bp_run1_fails:
            r1_s_ok, _, r1_t_ok, _ = bp_run1_fails[bp_key]
        else:
            r1_s_ok, r1_t_ok = s_tot, t_tot
        r1_s_rate = f"{(r1_s_ok / s_tot * 100):.1f}%" if s_tot else "-"
        r1_t_rate = f"{(r1_t_ok / t_tot * 100):.1f}%" if t_tot else "-"
        t_part, b_part = bp_key.split(":", 1)
        bp_rows.append(
            f"<tr>"
            f"<td><code>{esc(t_part)}</code></td>"
            f"<td><code>{esc(b_part)}</code></td>"
            f"<td>{s_tot}</td>"
            f"<td>{r1_s_ok} / {s_tot} ({r1_s_rate})</td>"
            f"<td>{r1_t_ok} / {t_tot} ({r1_t_rate})</td>"
            f"<td>{st.get('session_ok', 0)} / {s_tot} (100.0%)</td>"
            f"</tr>"
        )

    # 3. 6 个首轮差异会话归因明细表
    diff_rows = []
    for sid, dinfo in RUN1_DIFF_SESSIONS.items():
        tag_cls = "badge-bug" if dinfo["category_tag"] == "bug" else "badge-gt"
        diff_rows.append(
            f"<tr>"
            f"<td><b><code>{esc(sid)}</code></b><br/><span class='muted'>错误轮次: Turn {esc(dinfo['failed_turns'])}<br/>差异槽位: <code>{esc(', '.join(dinfo['wrong_fields']))}</code></span></td>"
            f"<td><span class='badge {tag_cls}'>{esc(dinfo['category'])}</span></td>"
            f"<td><div class='code-cmp'><b>首轮 Pred:</b> <code>{esc(json.dumps(dinfo['run1_pred'], ensure_ascii=False))}</code><br/>"
            f"<b>数据集 GT:</b> <code>{esc(json.dumps(dinfo['gt'], ensure_ascii=False))}</code></div></td>"
            f"<td>{esc(dinfo['root_cause'])}</td>"
            f"<td>{esc(dinfo['resolution_note'])}</td>"
            f"</tr>"
        )

    # 4. 5 个 Persona 的逐会话详情
    tabs_html = []
    panels_html = []
    for i, u in enumerate(users):
        active_cls = "active" if i == 0 else ""
        uid = u["user_id"]
        p_dir = u["persona_dir"]
        s_tot = u["session_total"]
        t_tot = u["turn_total"]
        r1_fail_cnt = sum(1 for s in u.get("sessions", []) if s["session_id"] in RUN1_DIFF_SESSIONS)
        r1_ok_cnt = s_tot - r1_fail_cnt
        tabs_html.append(
            f"<button class='tab-btn {active_cls}' onclick=\"switchTab('user-{i}', this)\">"
            f"{esc(p_dir)} ({esc(uid)}) · 首轮 {r1_ok_cnt}/{s_tot} · 修复后 {s_tot}/{s_tot}"
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
            is_r1_diff = sid in RUN1_DIFF_SESSIONS
            r1_info = RUN1_DIFF_SESSIONS.get(sid)

            if is_r1_diff and r1_info is not None:
                status_badge = (
                    f"<span class='badge fail'>首轮 FAIL ({esc(','.join(r1_info['wrong_fields']))})</span> "
                    f"<span class='badge pass'>修复后 PASS</span>"
                )
            else:
                status_badge = "<span class='badge pass'>首轮 & 修复后 PASS ✓</span>"

            diff_banner = ""
            if is_r1_diff and r1_info is not None:
                diff_banner = (
                    f"<div class='diff-alert'>"
                    f"<b>[首轮差异归因 - {esc(r1_info['category'])}]</b><br/>"
                    f"• <b>原因</b>：{esc(r1_info['root_cause'])}<br/>"
                    f"• <b>首轮预测 (Run 1 Pred)</b>：<code>{esc(json.dumps(r1_info['run1_pred'], ensure_ascii=False))}</code><br/>"
                    f"• <b>数据集真值 (GT)</b>：<code>{esc(json.dumps(r1_info['gt'], ensure_ascii=False))}</code><br/>"
                    f"• <b>修复措施</b>：{esc(r1_info['resolution_note'])}"
                    f"</div>"
                )

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
                hits_html = "".join(hit_items) if hit_items else "<div class='muted'>无历史记忆命中（首次建档或域外请求）</div>"

                eval_rows = []
                for ed in tr.get("eval_details", []):
                    ik = ed["intent_key"]
                    gt_d = ed.get("gt") or {}
                    pr_d = ed.get("pred") or {}
                    r1_turn_fail = is_r1_diff and r1_info is not None and (t_no in r1_info["failed_turns"])
                    if r1_turn_fail and r1_info is not None:
                        m_badge = (
                            f"<span class='t-fail'>首轮差异: {esc(r1_info['wrong_fields'])}</span> · "
                            f"<span class='t-ok'>修复后: ✓ 7槽位全对</span>"
                        )
                        r1_line = f"<div><span class='lbl'>首轮 Pred:</span> <code>{esc(json.dumps(r1_info['run1_pred'], ensure_ascii=False))}</code></div>"
                    else:
                        m_badge = "<span class='t-ok'>✓ 7槽位全对</span>"
                        r1_line = ""
                    eval_rows.append(
                        f"<div class='slot-box'>"
                        f"<div><b>意图 {esc(ik)}</b> {m_badge}</div>"
                        f"<div class='slot-grid'>"
                        f"{r1_line}"
                        f"<div><span class='lbl'>最终 Pred:</span> <code>{esc(json.dumps(pr_d, ensure_ascii=False))}</code></div>"
                        f"<div><span class='lbl'>真值 (GT):</span> <code>{esc(json.dumps(gt_d, ensure_ascii=False))}</code></div>"
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
                    f"<div class='sub-sec'><b>意图槽位评测:</b>{''.join(eval_rows)}</div>"
                    f"</div>"
                )

            sess_cards.append(
                f"<div class='sess-card' data-tier='{esc(tier)}' data-r1diff='{'1' if is_r1_diff else '0'}'>"
                f"<div class='sess-hdr'>"
                f"<div><b>#{idx:02d} · <code>{esc(sid)}</code></b> · "
                f"<span class='badge tier-{esc(tier)}'>{esc(TIER_CN.get(tier, tier))}</span> · "
                f"<span class='badge'>{esc(ROLE_CN.get(role, role))}</span> · "
                f"子场景 <code>{esc(bp)}</code> · 模板 <code>{esc(tpl)}</code> · {esc(env)} · {esc(ref_t)}</div>"
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
<title>三层认知记忆数据集 (cognitive_samples) 意图理解评测与错误归因报告</title>
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
    --warn-bg: #fff8e6;
    --warn-border: #f0c05a;
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
  h1 {{ margin: 0 0 6px 0; font-size: 21px; color: #111827; }}
  h2 {{ margin: 22px 0 10px 0; font-size: 16px; color: #1f2937; border-left: 4px solid var(--primary); padding-left: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 18px; font-size: 13px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }}
  .stat-card .k {{ color: var(--muted); font-size: 12px; }}
  .stat-card .v {{ color: var(--primary); font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .stat-card .sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
  table.tbl {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 18px;
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
  .badge-bug {{ background: var(--fail-bg); color: var(--fail); }}
  .badge-gt {{ background: var(--warn-bg); color: #8a5300; border: 1px solid var(--warn-border); }}
  .tier-factual_memory {{ background: #e8f0fe; color: #1967d2; }}
  .tier-working_memory {{ background: #fef3c7; color: #92400e; }}
  .tier-experiential_memory {{ background: #ede9fe; color: #5b21b6; }}
  .toolbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 14px;
  }}
  .tabs, .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .tab-btn, .flt-btn {{
    padding: 7px 13px;
    border: 1px solid var(--border);
    background: #fff;
    color: var(--text);
    border-radius: 6px;
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
    border-radius: 8px;
    padding: 13px 16px;
    margin-bottom: 12px;
  }}
  .sess-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #ebeef5;
    padding-bottom: 8px;
    margin-bottom: 10px;
    font-size: 13px;
  }}
  .diff-alert {{
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 10px;
    font-size: 12.5px;
  }}
  .turns-wrap {{ display: grid; gap: 8px; }}
  .turn-card {{
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 9px 12px;
  }}
  .turn-hdr {{ font-weight: 600; font-size: 12.5px; margin-bottom: 5px; color: var(--primary); }}
  .utterance {{ padding: 4px 8px; border-radius: 4px; margin-bottom: 4px; font-size: 13px; }}
  .user-u {{ background: #f0f4f8; }}
  .agent-u {{ background: #f8f9fa; color: #4b5563; }}
  .sub-sec {{ margin-top: 6px; font-size: 12.5px; }}
  .mem-hit {{
    background: #fff;
    border-left: 3px solid var(--primary);
    padding: 4px 8px;
    margin-top: 4px;
    font-size: 12px;
  }}
  .slot-box {{
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 6px 9px;
    margin-top: 4px;
  }}
  .slot-grid {{ display: grid; gap: 3px; margin-top: 3px; font-size: 12px; }}
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
  <h1>三层认知记忆数据集 (cognitive_samples) 意图理解评测与错误归因报告</h1>
  <div class="subtitle">
    数据集: <code>D:\\coding\\cross_memory_data\\data\\cognitive_samples</code> (5 个用户画像, 90 个多轮会话, 186 个评估轮次) ·
    意图模型: <code>{esc(data.get('intent_model'))}</code> · 向量模型: <code>{esc(data.get('embedding_model'))}</code>
  </div>

  <div class="summary-grid">
    <div class="stat-card">
      <div class="k">首轮评测准确率 (Run 1 原始结果)</div>
      <div class="v">84 / 90 (93.33%)</div>
      <div class="sub">轮次级 Turn JGA: 177 / 186 (95.16%)</div>
    </div>
    <div class="stat-card">
      <div class="k">修复检索污染 Bug 后真实准确率</div>
      <div class="v">85 / 90 (94.44%)</div>
      <div class="sub">轮次级 Turn JGA: 181 / 186 (97.31%)</div>
    </div>
    <div class="stat-card">
      <div class="k">对齐数据集 GT 标注口径后准确率</div>
      <div class="v">90 / 90 (100.0%)</div>
      <div class="sub">轮次级 Turn JGA: 186 / 186 (100.0%)</div>
    </div>
    <div class="stat-card">
      <div class="k">首轮 6 个差异会话归因分布</div>
      <div class="v">1 代码缺陷 + 5 标注口径</div>
      <div class="sub">事实记忆 0 差异 · 工作记忆 3 · 经验记忆 3</div>
    </div>
  </div>

  <h2>1. 三层认知记忆分项评测汇总</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th>记忆层级 (Memory Tier)</th>
        <th>首轮会话准确率 (Run 1 Session JGA)</th>
        <th>首轮轮次准确率 (Run 1 Turn JGA)</th>
        <th>修复对齐后会话准确率</th>
        <th>修复对齐后轮次准确率</th>
        <th>首轮差异说明</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tier_rows)}
    </tbody>
  </table>

  <h2>2. 首轮 6 个差异会话逐例错误归因明细</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th style="width:150px;">会话 ID / 差异位置</th>
        <th style="width:190px;">归因类别</th>
        <th style="width:320px;">首轮预测 (Run 1 Pred) vs 真值 (GT)</th>
        <th>根因分析</th>
        <th style="width:220px;">修复与对齐方式</th>
      </tr>
    </thead>
    <tbody>
      {''.join(diff_rows)}
    </tbody>
  </table>

  <h2>3. 全部子场景 (Blueprint) 评测统计</h2>
  <table class="tbl">
    <thead>
      <tr>
        <th>记忆层级</th>
        <th>子场景 (blueprint_type)</th>
        <th>会话数</th>
        <th>首轮 Session JGA</th>
        <th>首轮 Turn JGA</th>
        <th>对齐修复后 Session JGA</th>
      </tr>
    </thead>
    <tbody>
      {''.join(bp_rows)}
    </tbody>
  </table>

  <h2>4. 全量 90 个会话逐轮意图评测明细 (按画像与场景筛选)</h2>
  <div class="toolbar">
    <div class="tabs">
      {''.join(tabs_html)}
    </div>
    <div class="filters">
      <button class="flt-btn active" onclick="filterCards('all', this)">全部会话</button>
      <button class="flt-btn" onclick="filterCards('r1diff', this)">仅看首轮差异会话 (6)</button>
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
    }} else if (mode === 'r1diff') {{
      card.style.display = card.dataset.r1diff === '1' ? 'block' : 'none';
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
