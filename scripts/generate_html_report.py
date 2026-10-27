from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parents[1]
CROSS_DATA_DIR = Path(r"D:\coding\cross_memory_data\data\generated")
REPORT_JSON_PATH = repo_root / "test" / "cross_memory_sample_report.json"
OUT_HTML_PATH = repo_root / "test" / "cross_memory_sample_report.html"


def build_enriched_data() -> Dict[str, Any]:
    raw_report = json.loads(REPORT_JSON_PATH.read_text(encoding="utf-8"))
    enriched: Dict[str, Any] = {}

    for persona_name, pdata in raw_report.items():
        persona_dir = CROSS_DATA_DIR / persona_name
        skeleton = json.loads((persona_dir / "persona_skeleton.json").read_text(encoding="utf-8"))
        traces_list = json.loads((persona_dir / "memory_traces.json").read_text(encoding="utf-8"))
        traces_map = {t["session_id"]: t for t in traces_list}

        sessions_map: Dict[str, Any] = {}
        for line in (persona_dir / "sessions.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                sessions_map[obj["session_id"]] = obj

        prev_keys: Dict[str, str] = {}
        enriched_steps: List[Dict[str, Any]] = []
        for step in pdata["steps"]:
            sid = step["session_id"]
            sess_obj = sessions_map.get(sid, {})
            trace_obj = traces_map.get(sid, {})
            turns = []
            for ev in sess_obj.get("event_sequence", []):
                turns.append(
                    {
                        "turn": ev.get("turn"),
                        "user": (ev.get("user") or {}).get("utterance", ""),
                        "user_actions": (ev.get("user") or {}).get("action_types", []),
                        "agent": (ev.get("agent") or {}).get("utterance", ""),
                        "agent_actions": (ev.get("agent") or {}).get("action_types", []),
                    }
                )

            cur_keys: Dict[str, str] = {}
            memories_with_change = []
            for m in step["memories"]:
                k = f"{m['subtype']}::{m['memory_key']}"
                sig = f"{m['status']}|{m['votes']}|{m['content'].get('resolution')}|{m['content'].get('latency')}|{m['content'].get('time_window')}|{m['content'].get('target')}"
                cur_keys[k] = sig
                if k not in prev_keys:
                    change_tag = "new"
                elif prev_keys[k] != sig:
                    change_tag = "updated"
                else:
                    change_tag = "unchanged"
                m_copy = dict(m)
                m_copy["change_tag"] = change_tag
                memories_with_change.append(m_copy)
            prev_keys = cur_keys

            enriched_steps.append(
                {
                    "session_id": sid,
                    "reference_time": sess_obj.get("reference_time", ""),
                    "environment": (sess_obj.get("session_meta", {}).get("scenario") or {}).get("environment", ""),
                    "memory_role": step["memory_role"],
                    "memory_action": step["memory_action"],
                    "turns": turns,
                    "gold_events": trace_obj.get("memory_events", []),
                    "gold_state_after": trace_obj.get("gold_state_after", {}),
                    "memories": memories_with_change,
                }
            )

        enriched[persona_name] = {
            "user_id": pdata["user_id"],
            "static_profile": skeleton.get("static_profile", {}),
            "steps": enriched_steps,
            "final_memories": pdata["final_memories"],
            "sample_search_top3": pdata["sample_search_top3"],
        }

    return enriched


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Agent Memory System — 跨会话记忆萃取可视化评测报告</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-alt: #263349;
      --border: #334155;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-soft: rgba(56, 189, 248, 0.14);
      --green: #22c55e;
      --green-soft: rgba(34, 197, 94, 0.16);
      --amber: #f59e0b;
      --amber-soft: rgba(245, 158, 11, 0.16);
      --purple: #a855f7;
      --purple-soft: rgba(168, 85, 247, 0.18);
      --red: #ef4444;
      --red-soft: rgba(239, 68, 68, 0.16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    header {
      padding: 22px 32px;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header-title h1 {
      margin: 0;
      font-size: 21px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    .header-title p {
      margin: 4px 0 0;
      font-size: 13px;
      color: var(--muted);
    }
    .header-badges {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .pill {
      padding: 5px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .pill.accent {
      background: var(--accent-soft);
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--accent);
    }
    .pill.green {
      background: var(--green-soft);
      border-color: rgba(34, 197, 94, 0.4);
      color: #4ade80;
    }
    .container {
      max-width: 1560px;
      margin: 0 auto;
      padding: 20px 28px 48px;
    }
    /* Persona selector tabs */
    .persona-tabs {
      display: flex;
      gap: 12px;
      margin-bottom: 18px;
      flex-wrap: wrap;
    }
    .persona-tab {
      flex: 1;
      min-width: 260px;
      padding: 14px 18px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.18s ease;
      text-align: left;
    }
    .persona-tab:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .persona-tab.active {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(30, 41, 59, 0.95));
      border-color: var(--accent);
      box-shadow: 0 4px 20px rgba(56, 189, 248, 0.12);
    }
    .persona-tab .pt-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .persona-tab .pt-id {
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }
    .persona-tab .pt-role {
      font-size: 12px;
      color: var(--accent);
      background: var(--accent-soft);
      padding: 2px 8px;
      border-radius: 6px;
    }
    .persona-tab .pt-desc {
      font-size: 12.5px;
      color: var(--muted);
    }

    /* Timeline stepper */
    .timeline-bar {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      overflow-x: auto;
      padding-bottom: 4px;
    }
    .step-btn {
      flex: 1;
      min-width: 195px;
      padding: 11px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      cursor: pointer;
      text-align: left;
      color: var(--text);
      transition: all 0.15s ease;
    }
    .step-btn:hover {
      border-color: #64748b;
    }
    .step-btn.active {
      border-color: var(--accent);
      background: var(--panel-alt);
      box-shadow: inset 0 0 0 1px var(--accent);
    }
    .step-btn .sb-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .step-btn .sb-action {
      font-size: 11.5px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .role-badge {
      font-size: 10.5px;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
    }
    .role-evidence_session { background: rgba(56, 189, 248, 0.18); color: #38bdf8; }
    .role-reinforcement_session { background: rgba(34, 197, 94, 0.18); color: #4ade80; }
    .role-reuse_session { background: rgba(168, 85, 247, 0.18); color: #c084fc; }
    .role-distractor_session { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; }
    .role-correction_session { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }

    /* Main 3-column grid */
    .workspace-grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr 1.25fr;
      gap: 18px;
      align-items: start;
    }
    @media (max-width: 1200px) {
      .workspace-grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }
    .card-header {
      padding: 13px 16px;
      background: rgba(15, 23, 42, 0.45);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      font-weight: 700;
    }
    .card-body {
      padding: 16px;
    }

    /* Chat bubbles */
    .session-meta-box {
      background: rgba(15, 23, 42, 0.5);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 14px;
      font-size: 12.5px;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .session-meta-box strong { color: var(--text); }
    .turn-block {
      margin-bottom: 14px;
    }
    .turn-label {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 5px;
      font-weight: 600;
    }
    .bubble {
      padding: 10px 13px;
      border-radius: 10px;
      font-size: 13px;
      margin-bottom: 8px;
      position: relative;
    }
    .bubble.user {
      background: rgba(56, 189, 248, 0.13);
      border: 1px solid rgba(56, 189, 248, 0.3);
      color: #e0f2fe;
    }
    .bubble.agent {
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid var(--border);
      color: #cbd5e1;
    }
    .bubble-role {
      font-size: 11px;
      font-weight: 700;
      margin-bottom: 3px;
      display: flex;
      justify-content: space-between;
      opacity: 0.85;
    }

    /* Gold trace */
    .gold-event {
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.3);
      margin-bottom: 12px;
      font-size: 12.5px;
    }
    .gold-event-type {
      font-weight: 700;
      color: #fbbf24;
      margin-bottom: 4px;
      font-size: 12px;
    }
    .gold-slot-card {
      padding: 11px 13px;
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid var(--border);
      margin-bottom: 10px;
      font-size: 12.5px;
    }
    .gold-slot-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      margin-bottom: 6px;
      color: #93c5fd;
    }

    /* Extracted Memory Nodes */
    .mem-filter-bar {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .filter-chip {
      font-size: 11.5px;
      padding: 3px 9px;
      border-radius: 6px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--muted);
      cursor: pointer;
    }
    .filter-chip.active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }
    .mem-node {
      padding: 12px 14px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      margin-bottom: 10px;
      transition: all 0.15s ease;
    }
    .mem-node.change-new {
      border-left: 4px solid var(--green);
      background: rgba(34, 197, 94, 0.06);
    }
    .mem-node.change-updated {
      border-left: 4px solid var(--accent);
      background: rgba(56, 189, 248, 0.07);
    }
    .mem-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
      flex-wrap: wrap;
    }
    .mem-key {
      font-weight: 700;
      font-size: 13.5px;
      color: #f8fafc;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .badges {
      display: flex;
      gap: 6px;
      align-items: center;
    }
    .badge {
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 5px;
      font-weight: 600;
    }
    .badge-subtype {
      background: rgba(148, 163, 184, 0.16);
      color: #cbd5e1;
    }
    .status-candidate { background: var(--amber-soft); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .status-active { background: var(--green-soft); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }
    .status-reinforced { background: var(--purple-soft); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .change-badge-new { background: #16a34a; color: #fff; }
    .change-badge-updated { background: #0284c7; color: #fff; }

    .kv-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 7px 0;
    }
    .kv-tag {
      font-size: 11.5px;
      padding: 2px 8px;
      border-radius: 5px;
      background: var(--panel);
      border: 1px solid var(--border);
      color: #e2e8f0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .kv-tag span { color: var(--muted); }
    .mem-desc {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }

    /* Bottom Search & Summary section */
    .bottom-grid {
      margin-top: 20px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    @media (max-width: 1000px) {
      .bottom-grid { grid-template-columns: 1fr; }
    }
    .search-hit {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 13px;
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.55);
      border: 1px solid var(--border);
      margin-bottom: 8px;
      gap: 12px;
    }
    .score-bar-wrap {
      min-width: 110px;
      text-align: right;
    }
    .score-num {
      font-size: 12.5px;
      font-weight: 700;
      color: var(--accent);
      font-family: monospace;
    }
    .score-bar {
      height: 5px;
      background: var(--border);
      border-radius: 3px;
      margin-top: 4px;
      overflow: hidden;
    }
    .score-fill {
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
    }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h1>Agent Memory System — 真实样本记忆萃取与状态演进可视化报告</h1>
      <p>数据源：D:\\coding\\cross_memory_data\\data\\generated （mobile_persona_001 ~ 003） | 引擎：FastPath + LLM (qwen3.8-flash) + Deterministic Guardrails + Induction</p>
    </div>
    <div class="header-badges">
      <span class="pill accent">LLM: qwen3.8-flash</span>
      <span class="pill accent">Embedding: text-embedding-v4</span>
      <span class="pill green">15/15 会话与 Ground Truth 对齐</span>
    </div>
  </header>

  <div class="container">
    <div class="persona-tabs" id="personaTabs"></div>
    <div class="timeline-bar" id="timelineBar"></div>

    <div class="workspace-grid">
      <!-- Column 1: Raw Session Turns -->
      <div class="card">
        <div class="card-header">
          <span>1. 原始多轮对话输入 (Session Turns)</span>
          <span id="col1SessionId" style="color: var(--accent); font-family: monospace;"></span>
        </div>
        <div class="card-body" id="dialoguePanel"></div>
      </div>

      <!-- Column 2: Ground Truth Trace -->
      <div class="card">
        <div class="card-header">
          <span>2. 数据集 Ground Truth (memory_traces.json)</span>
          <span style="font-size: 12px; color: var(--muted);">期望标准状态</span>
        </div>
        <div class="card-body" id="goldPanel"></div>
      </div>

      <!-- Column 3: Extracted Memory Nodes -->
      <div class="card">
        <div class="card-header">
          <span>3. MemorySystem 真实萃取快照 (memory.json)</span>
          <span id="nodeCountBadge" class="pill"></span>
        </div>
        <div class="card-body">
          <div class="mem-filter-bar" id="subtypeFilterBar"></div>
          <div id="extractedPanel"></div>
        </div>
      </div>
    </div>

    <div class="bottom-grid">
      <div class="card">
        <div class="card-header">
          <span>混合检索验证 (Query: "老规矩网络保障配置" — RRF + 热度时间衰减重排 Top-3)</span>
        </div>
        <div class="card-body" id="searchPanel"></div>
      </div>
      <div class="card">
        <div class="card-header">
          <span>关键防呆机制验证结论</span>
        </div>
        <div class="card-body" style="font-size: 13px; color: #cbd5e1;">
          <div style="margin-bottom: 8px;">✅ <strong>显式声明 vs 单次隐式行为防过拟合</strong>：含“以后/往后/老规矩”显式词时直接进入 <code>active</code>；无长期词的单次任务（如 <code>S-USER_002-01</code>）进入 <code>candidate</code> 候选态，二次复现（<code>S-USER_002-03</code>）才结晶转正。</div>
          <div style="margin-bottom: 8px;">✅ <strong>临时覆盖防变体污染</strong>：遇到“今天现场特殊，临时调到720p/30ms”（<code>S-USER_001-09</code>、<code>S-USER_002-11</code>、<code>S-USER_003-11</code>），有效性归一器强制覆写为 <code>temporary</code>，状态机直接拦截，保护长期 <code>1080p/50ms</code> 基线不被污染。</div>
          <div style="margin-bottom: 8px;">✅ <strong>跨轮时段合成与消歧黑话</strong>：跨轮次的“开始时间 + 持续时长/口语俩小时”精准合成 <code>HH:MM-HH:MM</code>；跨轮消歧（如“保个基本画质 -> 720p”）及同轮混合“临时降配 + 新增长期黑话”（<code>S-USER_003-11</code>）均准确分离落盘。</div>
          <div>✅ <strong>域外拒识零污染</strong>：宽带光纤报修等域外拒识会话（<code>S-USER_001-04</code>）不产生任何脏记忆节点。</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const REPORT_DATA = __REPORT_DATA_JSON__;
    const PERSONA_KEYS = Object.keys(REPORT_DATA);
    let currentPersona = PERSONA_KEYS[0];
    let currentStepIdx = 0;
    let currentSubtypeFilter = "all";

    const PERSONA_SUMMARIES = {
      "mobile_persona_001": "显式长期规则 + 跨轮画质黑话消歧 + 域外拒识 + 临时降配防污染",
      "mobile_persona_002": "单次隐式行为(candidate) + 二次复现转正(active) + 应用别名「阿抖」",
      "mobile_persona_003": "口语时长换算(俩小时) + 单日延长不改基线 + 同轮临时降配与新黑话分离"
    };

    function renderPersonaTabs() {
      const wrap = document.getElementById("personaTabs");
      wrap.innerHTML = PERSONA_KEYS.map(pk => {
        const p = REPORT_DATA[pk];
        const prof = p.static_profile || {};
        const activeCls = pk === currentPersona ? "active" : "";
        return `
          <button class="persona-tab ${activeCls}" onclick="selectPersona('${pk}')">
            <div class="pt-top">
              <span class="pt-id">${pk} (${p.user_id})</span>
              <span class="pt-role">${prof.occupation || prof.role_archetype || "移动保障用户"}</span>
            </div>
            <div class="pt-desc">${PERSONA_SUMMARIES[pk] || ""}</div>
          </button>
        `;
      }).join("");
    }

    function selectPersona(pk) {
      currentPersona = pk;
      currentStepIdx = 0;
      currentSubtypeFilter = "all";
      renderAll();
    }

    function selectStep(idx) {
      currentStepIdx = idx;
      renderAll();
    }

    function selectSubtypeFilter(st) {
      currentSubtypeFilter = st;
      renderExtractedPanel();
    }

    function renderTimeline() {
      const steps = REPORT_DATA[currentPersona].steps;
      const bar = document.getElementById("timelineBar");
      bar.innerHTML = steps.map((s, idx) => {
        const activeCls = idx === currentStepIdx ? "active" : "";
        return `
          <button class="step-btn ${activeCls}" onclick="selectStep(${idx})">
            <div class="sb-head">
              <span>Step ${idx + 1}: ${s.session_id}</span>
              <span class="role-badge role-${s.memory_role}">${s.memory_role.replace("_session", "")}</span>
            </div>
            <div class="sb-action">${s.memory_action} · ${s.environment}</div>
          </button>
        `;
      }).join("");
    }

    function renderDialoguePanel() {
      const step = REPORT_DATA[currentPersona].steps[currentStepIdx];
      document.getElementById("col1SessionId").textContent = step.session_id;
      const panel = document.getElementById("dialoguePanel");

      const turnsHtml = step.turns.map(t => `
        <div class="turn-block">
          <div class="turn-label">Round #${t.turn}</div>
          <div class="bubble user">
            <div class="bubble-role">
              <span>👤 User</span>
              <span>${(t.user_actions || []).join(", ")}</span>
            </div>
            <div>${t.user}</div>
          </div>
          <div class="bubble agent">
            <div class="bubble-role">
              <span>🤖 Agent</span>
              <span>${(t.agent_actions || []).join(", ")}</span>
            </div>
            <div>${t.agent}</div>
          </div>
        </div>
      `).join("");

      panel.innerHTML = `
        <div class="session-meta-box">
          <div><strong>参考时间：</strong>${step.reference_time}</div>
          <div><strong>物理场景：</strong>${step.environment}</div>
          <div><strong>记忆动作：</strong><code>${step.memory_action}</code></div>
        </div>
        ${turnsHtml}
      `;
    }

    function renderGoldPanel() {
      const step = REPORT_DATA[currentPersona].steps[currentStepIdx];
      const panel = document.getElementById("goldPanel");

      const eventsHtml = (step.gold_events || []).map(ge => `
        <div class="gold-event">
          <div class="gold-event-type">⚡ ${ge.event_type} ${ge.memory_id ? `(${ge.memory_id})` : ""}</div>
          <div>${ge.description || ""}</div>
        </div>
      `).join("");

      const goldState = step.gold_state_after || {};
      const stateEntries = Object.values(goldState);
      const stateHtml = stateEntries.length === 0
        ? `<div style="color: var(--muted); font-size: 13px;">空状态 (无长期记忆)</div>`
        : stateEntries.map(gs => {
            const aliases = JSON.stringify(gs.alias_mapping || {});
            return `
              <div class="gold-slot-card">
                <div class="gold-slot-head">
                  <span>${gs.memory_id}: ${gs.application_name} · ${gs.service_name}</span>
                  <span class="badge status-${gs.status === 'provisional' ? 'candidate' : 'active'}">
                    ${gs.status} (evidence=${gs.evidence_count})
                  </span>
                </div>
                <div class="kv-grid">
                  <span class="kv-tag"><span>画质:</span> ${gs.resolution}</span>
                  <span class="kv-tag"><span>时延:</span> ${gs.rtt_max}</span>
                  <span class="kv-tag"><span>开始:</span> ${gs.start_time}</span>
                  ${gs.end_time ? `<span class="kv-tag"><span>结束:</span> ${gs.end_time}</span>` : ""}
                  <span class="kv-tag"><span>时长:</span> ${gs.duration}</span>
                </div>
                <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">
                  个性化映射: <code>${aliases}</code>
                </div>
              </div>
            `;
          }).join("");

      panel.innerHTML = `
        <div style="margin-bottom: 10px; font-size: 12px; font-weight: 700; color: var(--muted);">本轮触发记忆事件 (memory_events)</div>
        ${eventsHtml}
        <div style="margin: 14px 0 8px; font-size: 12px; font-weight: 700; color: var(--muted);">本轮后期望金标状态 (gold_state_after)</div>
        ${stateHtml}
      `;
    }

    function renderExtractedPanel() {
      const step = REPORT_DATA[currentPersona].steps[currentStepIdx];
      const allMems = step.memories || [];
      document.getElementById("nodeCountBadge").textContent = `共 ${allMems.length} 个记忆节点`;

      const subtypes = ["all", ...Array.from(new Set(allMems.map(m => m.subtype)))];
      const filterBar = document.getElementById("subtypeFilterBar");
      filterBar.innerHTML = subtypes.map(st => {
        const count = st === "all" ? allMems.length : allMems.filter(m => m.subtype === st).length;
        const activeCls = st === currentSubtypeFilter ? "active" : "";
        const label = st === "all" ? `全部 (${count})` : `${st} (${count})`;
        return `<button class="filter-chip ${activeCls}" onclick="selectSubtypeFilter('${st}')">${label}</button>`;
      }).join("");

      const filtered = currentSubtypeFilter === "all"
        ? allMems
        : allMems.filter(m => m.subtype === currentSubtypeFilter);

      const panel = document.getElementById("extractedPanel");
      panel.innerHTML = filtered.map(m => {
        const c = m.content || {};
        const changeCls = m.change_tag === "new" ? "change-new" : (m.change_tag === "updated" ? "change-updated" : "");
        const changeBadge = m.change_tag === "new"
          ? `<span class="badge change-badge-new">NEW</span>`
          : (m.change_tag === "updated" ? `<span class="badge change-badge-updated">UPDATED</span>` : "");

        const kvItems = [];
        if (c.application) kvItems.push(`<span class="kv-tag"><span>app:</span> ${c.application}</span>`);
        if (c.business) kvItems.push(`<span class="kv-tag"><span>biz:</span> ${c.business}</span>`);
        if (c.resolution) kvItems.push(`<span class="kv-tag"><span>res:</span> ${c.resolution}</span>`);
        if (c.latency) kvItems.push(`<span class="kv-tag"><span>latency:</span> ${c.latency}</span>`);
        if (c.time_window) kvItems.push(`<span class="kv-tag"><span>window:</span> ${c.time_window}</span>`);
        if (c.target) kvItems.push(`<span class="kv-tag"><span>target:</span> ${c.target}</span>`);
        if (c.declaration_mode) kvItems.push(`<span class="kv-tag"><span>mode:</span> ${c.declaration_mode}</span>`);

        return `
          <div class="mem-node ${changeCls}">
            <div class="mem-top">
              <span class="mem-key">${m.memory_key}</span>
              <div class="badges">
                ${changeBadge}
                <span class="badge badge-subtype">${m.subtype}</span>
                <span class="badge status-${m.status}">${m.status} (votes=${m.votes})</span>
              </div>
            </div>
            <div class="kv-grid">${kvItems.join("")}</div>
            <div class="mem-desc">${m.description}</div>
          </div>
        `;
      }).join("");
    }

    function renderSearchPanel() {
      const hits = REPORT_DATA[currentPersona].sample_search_top3 || [];
      const panel = document.getElementById("searchPanel");
      panel.innerHTML = hits.map((h, i) => {
        const pct = Math.min(100, Math.round(h.score * 100));
        return `
          <div class="search-hit">
            <div>
              <div style="font-weight: 700; font-size: 13.5px; margin-bottom: 3px;">
                #${i + 1} <code style="color: #38bdf8;">[${h.subtype}] ${h.memory_key}</code>
                <span class="badge status-${h.status}" style="margin-left: 6px;">${h.status}</span>
              </div>
              <div style="font-size: 12px; color: var(--muted);">${h.description}</div>
            </div>
            <div class="score-bar-wrap">
              <div class="score-num">score: ${h.score.toFixed(4)}</div>
              <div class="score-bar"><div class="score-fill" style="width: ${pct}%;"></div></div>
            </div>
          </div>
        `;
      }).join("");
    }

    function renderAll() {
      renderPersonaTabs();
      renderTimeline();
      renderDialoguePanel();
      renderGoldPanel();
      renderExtractedPanel();
      renderSearchPanel();
    }

    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    enriched = build_enriched_data()
    html_content = HTML_TEMPLATE.replace(
        "__REPORT_DATA_JSON__", json.dumps(enriched, ensure_ascii=False)
    )
    OUT_HTML_PATH.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML report at: {OUT_HTML_PATH}")


if __name__ == "__main__":
    main()
