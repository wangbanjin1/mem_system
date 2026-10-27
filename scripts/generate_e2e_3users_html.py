from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

REPORT_JSON = Path(r"D:\coding\memory_system\test\e2e_memory_intent_3users_report.json")
REPORT_HTML = Path(r"D:\coding\memory_system\test\e2e_memory_intent_3users_report.html")

ROLE_CN = {
    "evidence_session": "首次建档会话 (Evidence)",
    "reinforcement_session": "记忆强化会话 (Reinforcement)",
    "reuse_session": "跨期复用会话 (Reuse)",
    "correction_session": "临时改参会话 (Correction)",
    "distractor_session": "单次/域外干扰会话 (Distractor)",
}

SUBTYPE_CN = {
    "preferred_value": "默认参数偏好 (preferred_value)",
    "scene_preference": "场景参数覆写 (scene_preference)",
    "periodic_preference": "周期时间路由 (periodic_preference)",
    "service_to_app": "应用↔业务路由 (service_to_app)",
    "expression": "黑话词典 (expression)",
}


def esc(v: Any) -> str:
    return html.escape(str(v) if v is not None else "")


def render_html(data: Dict[str, Any]) -> str:
    users: List[Dict[str, Any]] = data.get("users", [])
    by_role: Dict[str, Dict[str, int]] = data.get("by_memory_role", {})

    role_rows = []
    for role, st in by_role.items():
        s_tot = st.get("sessions", 0)
        s_ok = st.get("session_ok", 0)
        t_tot = st.get("turns", 0)
        t_ok = st.get("turn_ok", 0)
        s_rate = f"{(s_ok / s_tot * 100):.1f}%" if s_tot else "-"
        t_rate = f"{(t_ok / t_tot * 100):.1f}%" if t_tot else "-"
        role_rows.append(
            f"<tr>"
            f"<td><span class='badge role-{esc(role)}'>{esc(ROLE_CN.get(role, role))}</span></td>"
            f"<td>{s_ok} / {s_tot} (<b>{s_rate}</b>)</td>"
            f"<td>{t_ok} / {t_tot} (<b>{t_rate}</b>)</td>"
            f"</tr>"
        )

    tabs_html = []
    panels_html = []
    for i, u in enumerate(users):
        active_cls = "active" if i == 0 else ""
        uid = u["user_id"]
        p_dir = u["persona_dir"]
        s_ok = u["session_ok"]
        s_tot = u["session_total"]
        t_ok = u["turn_ok"]
        t_tot = u["turn_total"]
        tabs_html.append(
            f"<button class='tab-btn {active_cls}' onclick=\"switchTab('user-{i}', this)\">"
            f"{esc(p_dir)} ({esc(uid)}) · Session {s_ok}/{s_tot} · Turn {t_ok}/{t_tot}"
            f"</button>"
        )

        sess_cards = []
        for idx, s in enumerate(u.get("sessions", []), 1):
            sid = s["session_id"]
            role = s["memory_role"]
            tpl = s["template_id"]
            bp = s["blueprint_type"]
            env = s["environment"]
            ref_t = s["reference_time"]
            s_pass = s["session_pass"]
            pass_badge = (
                "<span class='badge pass'>Session PASS ✓</span>"
                if s_pass
                else f"<span class='badge fail'>Session FAIL ✗ ({esc(s.get('session_diffs'))})</span>"
            )

            # Turns
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
                best_m = m_trace.get("selected_best_major")
                has_trig = m_trace.get("has_memory_trigger", False)

                hit_items = []
                for mh in m_hits[:3]:
                    rs = mh.get("resolved_slots") or {}
                    pp_pair = f" + <code>periodic[{esc(mh.get('paired_periodic_key'))}]</code>" if mh.get("paired_periodic_key") else ""
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[{esc(SUBTYPE_CN.get(mh['subtype'], mh['subtype']))}]</b> "
                        f"<code>{esc(mh['memory_key'])}</code>{pp_pair} (status={esc(mh['status'])}, score={mh['score']:.3f})<br/>"
                        f"→ 正交组装槽位: app=<b>{esc(rs.get('application_name'))}</b>, svc=<b>{esc(rs.get('service_name'))}</b>, "
                        f"res=<b>{esc(rs.get('resolution'))}</b>, rtt=<b>{esc(rs.get('rtt'))}</b>, "
                        f"时间=<b>{esc(rs.get('start_timestamp'))} ~ {esc(rs.get('end_timestamp'))} ({esc(rs.get('duration'))})</b>"
                        f"</div>"
                    )
                for ph in p_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[周期时间路由 (periodic_preference)]</b> "
                        f"<code>{esc(ph.get('memory_key'))}</code> (status={esc(ph.get('status'))}, score={ph.get('score', 0.0):.3f}) "
                        f"→ 时段 <b>{esc(ph.get('time_window'))}</b> 指向 <b>{esc(ph.get('value'))}</b>"
                        f"</div>"
                    )
                for sh in s_hits[:2]:
                    hit_items.append(
                        f"<div class='mem-hit'>"
                        f"<b>[应用↔业务路由 (service_to_app)]</b> "
                        f"<code>{esc(sh.get('memory_key'))}</code> (status={esc(sh.get('status'))}, score={sh.get('score', 0.0):.3f}) "
                        f"→ <b>{esc(sh.get('business'))}</b> ↔ <b>{esc(sh.get('application'))}</b>"
                        f"</div>"
                    )
                for eh in e_hits[:3]:
                    hit_items.append(
                        f"<div class='mem-hit expr-hit'>"
                        f"<b>[黑话词典 (expression)]</b> 「{esc(eh.get('alias'))}」 → <b>{esc(eh.get('target'))}</b> "
                        f"<span class='muted'>(作用域: {esc(eh.get('application'))}|{esc(eh.get('business'))})</span>"
                        f"</div>"
                    )
                hits_html = "".join(hit_items) if hit_items else "<div class='muted'>本轮未命中历史记忆节点（初始建档或域外请求）</div>"

                trig_badge = (
                    "<span class='trig-on'>触发长期记忆补全 (老规矩/习惯)</span>"
                    if has_trig and best_m
                    else "<span class='trig-off'>常规对话状态跟踪</span>"
                )

                # Pred vs GT per turn
                eval_rows = []
                for ed in tr.get("eval_details", []):
                    ik = ed["intent_key"]
                    m_ok = ed["match"]
                    gt_d = ed.get("gt") or {}
                    pr_d = ed.get("pred") or {}
                    m_badge = "<span class='t-ok'>✓ 7槽位全对</span>" if m_ok else f"<span class='t-fail'>✗ 差异: {esc(ed.get('wrong'))}</span>"
                    eval_rows.append(
                        f"<div class='slot-box'>"
                        f"<div><b>意图 {esc(ik)}</b> {m_badge}</div>"
                        f"<div class='slot-grid'>"
                        f"<div><span class='lbl'>预测 (Pred):</span> <code>{esc(json.dumps(pr_d, ensure_ascii=False))}</code></div>"
                        f"<div><span class='lbl'>真值 (GT):</span> <code>{esc(json.dumps(gt_d, ensure_ascii=False))}</code></div>"
                        f"</div>"
                        f"</div>"
                    )
                if not eval_rows:
                    eval_rows.append("<div class='slot-box muted'>域外非保障请求（GT 与 Pred 均为空意图，符合预期 ✓）</div>")

                turn_blocks.append(
                    f"<div class='turn-card'>"
                    f"<div class='turn-hdr'>Turn {t_no} · {trig_badge}</div>"
                    f"<div class='utterance user-u'><b>👤 用户:</b> {esc(u_utt)}</div>"
                    f"<div class='utterance agent-u'><b>🤖 助手:</b> {esc(a_utt)}</div>"
                    f"<div class='sub-sec'><b>🔍 轮前记忆检索 (MemorySystem.search):</b>{hits_html}</div>"
                    f"<div class='sub-sec'><b>🎯 意图理解输出 (qwen3.8-flash Text-to-Python DST):</b>{''.join(eval_rows)}</div>"
                    f"</div>"
                )

            # Memory snapshot after session
            snap_rows = []
            for mnode in s.get("memory_snapshot", []):
                c = mnode.get("content") or {}
                st_name = mnode["subtype"]
                if st_name == "expression":
                    slot_summary = f"alias=「{c.get('alias')}」 -> target={c.get('target')} (scope={c.get('application')}|{c.get('business')})"
                elif st_name == "periodic_preference":
                    slot_summary = f"time_window={c.get('time_window')} -> value={c.get('value')} (无QoS参数耦合)"
                elif st_name == "service_to_app":
                    slot_summary = f"business={c.get('business')} <-> application={c.get('application')} (默认路由)"
                elif st_name == "scene_preference":
                    slot_summary = f"scene={c.get('scene')}, app={c.get('app_cn') or c.get('application')}, svc={c.get('service_cn') or c.get('business')}, res={c.get('resolution')}, rtt={c.get('latency')} (无时间窗耦合)"
                else:
                    slot_summary = f"app={c.get('app_cn') or c.get('application')}, svc={c.get('service_cn') or c.get('business')}, res={c.get('resolution')}, rtt={c.get('latency')} (无时间窗/场景耦合)"
                snap_rows.append(
                    f"<tr>"
                    f"<td><code>{esc(st_name)}</code></td>"
                    f"<td><b>{esc(mnode['memory_key'])}</b></td>"
                    f"<td><span class='st-{esc(mnode['status'])}'>{esc(mnode['status'])}</span></td>"
                    f"<td>votes={mnode['votes']} (pos={mnode['positive_votes']}, reuse={mnode['reuse_count']})</td>"
                    f"<td>R={mnode['retention']:.4f} (S={mnode['stability']:.2f})</td>"
                    f"<td>{esc(slot_summary)}</td>"
                    f"</tr>"
                )
            snap_table = (
                "<table class='snap-tbl'><thead><tr><th>记忆分栏</th><th>Memory Key</th><th>状态</th><th>投票/复用</th><th>艾宾浩斯保持率</th><th>核心槽位/作用域</th></tr></thead>"
                f"<tbody>{''.join(snap_rows)}</tbody></table>"
                if snap_rows
                else "<div class='muted'>当前记忆库为空</div>"
            )

            sess_cards.append(
                f"<div class='sess-card'>"
                f"<div class='sess-hdr'>"
                f"<div><b>#{idx:02d} · {esc(sid)}</b> · <span class='badge role-{esc(role)}'>{esc(ROLE_CN.get(role, role))}</span> "
                f"· 模板 <code>{esc(tpl)}</code> · 故事线 <code>{esc(bp)}</code> ({esc(env)}) · 🕒 {esc(ref_t)}</div>"
                f"<div>{pass_badge} · Turn {s['turn_ok_count']}/{s['turn_total_count']}</div>"
                f"</div>"
                f"<div class='turns-wrap'>{''.join(turn_blocks)}</div>"
                f"<div class='after-mem'><b>📦 会话后记忆萃取与库快照 (MemorySystem.record_session):</b>{snap_table}</div>"
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
<title>MemorySystem × Qwen3.8-Flash 端到端多轮意图理解评测报告 (3随机样本)</title>
<style>
  :root {{
    --bg: #fdf8f2;
    --card: #ffffff;
    --border: #eadbc8;
    --primary: #b85d19;
    --primary-soft: #fbeee0;
    --text: #3a2e25;
    --muted: #7d6b5d;
    --ok: #2e7d32;
    --ok-bg: #e8f5e9;
    --fail: #c62828;
    --fail-bg: #ffebee;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 28px 36px;
    line-height: 1.55;
  }}
  h1 {{ color: #8c3f08; margin: 0 0 8px 0; font-size: 24px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 22px; font-size: 14px; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }}
  .stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 6px rgba(140, 63, 8, 0.05);
  }}
  .stat-card .k {{ color: var(--muted); font-size: 13px; }}
  .stat-card .v {{ color: var(--primary); font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .role-tbl, .snap-tbl {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 24px;
    font-size: 13px;
  }}
  .role-tbl th, .role-tbl td, .snap-tbl th, .snap-tbl td {{
    padding: 9px 12px;
    border-bottom: 1px solid #f2e7d8;
    text-align: left;
  }}
  .role-tbl th, .snap-tbl th {{ background: #f9efe3; color: #6e4726; }}
  .tabs {{ display: flex; gap: 10px; margin-bottom: 18px; }}
  .tab-btn {{
    padding: 10px 18px;
    border: 1px solid var(--border);
    background: #f7ede0;
    color: var(--text);
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
  }}
  .tab-btn.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  .sess-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 18px;
    box-shadow: 0 2px 6px rgba(140, 63, 8, 0.04);
  }}
  .sess-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    border-bottom: 2px solid #f5eadc;
    margin-bottom: 12px;
    font-size: 14px;
  }}
  .turn-card {{
    background: #fffcf8;
    border: 1px solid #efe2d1;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 13px;
  }}
  .turn-hdr {{ font-weight: 600; color: #7a4212; margin-bottom: 6px; }}
  .utterance {{ margin: 4px 0; }}
  .user-u {{ color: #2b211a; }}
  .agent-u {{ color: #665243; }}
  .sub-sec {{ margin-top: 8px; padding-top: 6px; border-top: 1px dashed #eadbc8; }}
  .mem-hit {{
    background: #fdf3e7;
    border-left: 3px solid #d9822b;
    padding: 6px 10px;
    margin: 5px 0;
    border-radius: 4px;
  }}
  .expr-hit {{ background: #f6efe8; border-left-color: #8e5a2a; }}
  .slot-box {{
    background: #f8f4ef;
    padding: 8px 10px;
    border-radius: 6px;
    margin-top: 5px;
  }}
  .slot-grid {{ display: grid; grid-template-columns: 1fr; gap: 3px; margin-top: 4px; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
  }}
  .pass, .t-ok {{ background: var(--ok-bg); color: var(--ok); padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
  .fail, .t-fail {{ background: var(--fail-bg); color: var(--fail); padding: 2px 8px; border-radius: 6px; font-weight: 600; }}
  .trig-on {{ background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 6px; }}
  .trig-off {{ background: #f1ebe4; color: #7d6b5d; padding: 2px 8px; border-radius: 6px; }}
  .st-active, .st-reinforced {{ color: #2e7d32; font-weight: 700; }}
  .st-candidate {{ color: #ef6c00; font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  code {{ background: #f5ebe0; padding: 1px 5px; border-radius: 4px; font-family: Consolas, monospace; font-size: 12px; }}
</style>
</head>
<body>
  <h1>MemorySystem × Qwen3.8-Flash 端到端意图理解评测报告</h1>
  <div class="subtitle">
    随机抽样 3 名用户（seed={esc(data.get('random_seed'))}: <b>{esc(', '.join(data.get('selected_personas', [])))}</b>） ·
    意图理解模型: <code>{esc(data.get('intent_model'))}</code> · 记忆萃取模型: <code>{esc(data.get('memory_model'))}</code>
  </div>

  <div class="summary-grid">
    <div class="stat-card">
      <div class="k">端到端 Session JGA (7槽位全匹配)</div>
      <div class="v">{data['total_session_ok']} / {data['total_sessions']} ({data['session_jga']*100:.1f}%)</div>
    </div>
    <div class="stat-card">
      <div class="k">端到端 Turn JGA (逐轮状态跟踪)</div>
      <div class="v">{data['total_turn_ok']} / {data['total_turns']} ({data['turn_jga']*100:.1f}%)</div>
    </div>
    <div class="stat-card">
      <div class="k">测试用户样本数</div>
      <div class="v">3 Users ({data['total_sessions']} Sessions)</div>
    </div>
    <div class="stat-card">
      <div class="k">三维记忆架构</div>
      <div class="v" style="font-size:16px;margin-top:10px;">周期记忆 · 场景记忆 · 挂靠黑话</div>
    </div>
  </div>

  <table class="role-tbl">
    <thead>
      <tr><th>会话角色 (Memory Role)</th><th>Session 级准确率 (Session JGA)</th><th>Turn 级准确率 (Turn JGA)</th></tr>
    </thead>
    <tbody>
      {''.join(role_rows)}
    </tbody>
  </table>

  <div class="tabs">
    {''.join(tabs_html)}
  </div>
  {''.join(panels_html)}

<script>
function switchTab(id, btn) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    html_str = render_html(data)
    REPORT_HTML.write_text(html_str, encoding="utf-8")
    print(f"Generated HTML report: {REPORT_HTML}")
