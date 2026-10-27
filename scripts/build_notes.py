from __future__ import annotations

import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
report_json_path = repo_root / "test" / "cross_memory_sample_report.json"
out_html_path = repo_root / "test" / "cross_memory_sample_report.html"


def build_html() -> None:
    report_data = json.loads(report_json_path.read_text(encoding="utf-8"))
    embedded_json = json.dumps(report_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>两个长跨度样本（USER_001 / USER_002）记忆萃取与检索全过程实测</title>
  <style>
    :root {{
      --bg-page: #F6EFE2;
      --bg-card: #FFFDF9;
      --bg-panel: #FBF5EA;
      --bg-subtle: #F3E9D8;
      --ink-main: #2C2018;
      --ink-sec: #5A4636;
      --ink-muted: #846C58;
      --accent-main: #B45309;
      --accent-dark: #92400E;
      --accent-soft: #FEF3C7;
      --border-warm: #E5D5BE;
      --border-strong: #CFB491;

      --role-evidence-bg: #FEF3C7;
      --role-evidence-fg: #92400E;
      --role-reinforce-bg: #DCFCE7;
      --role-reinforce-fg: #166534;
      --role-reuse-bg: #E0F2FE;
      --role-reuse-fg: #075985;
      --role-distractor-bg: #F3E8FF;
      --role-distractor-fg: #6B21A8;
      --role-correction-bg: #FEE2E2;
      --role-correction-fg: #991B1B;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 28px 18px 60px;
      background-color: var(--bg-page);
      color: var(--ink-main);
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans SC", sans-serif;
      line-height: 1.65;
    }}

    .container {{
      max-width: 1180px;
      margin: 0 auto;
    }}

    .header-box {{
      background: linear-gradient(135deg, #FFFDF9 0%, #F9EFE0 100%);
      border: 1px solid var(--border-warm);
      border-radius: 14px;
      padding: 24px 28px;
      margin-bottom: 22px;
      box-shadow: 0 4px 16px rgba(92, 64, 40, 0.05);
    }}

    .header-box h1 {{
      margin: 0 0 8px 0;
      font-size: 24px;
      color: var(--accent-dark);
    }}

    .header-box p {{
      margin: 0;
      color: var(--ink-sec);
      font-size: 14.5px;
    }}

    /* Top Control Bar */
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      background: var(--bg-card);
      border: 1px solid var(--border-warm);
      border-radius: 12px;
      padding: 14px 20px;
      margin-bottom: 16px;
      position: sticky;
      top: 10px;
      z-index: 100;
      box-shadow: 0 4px 14px rgba(92, 64, 40, 0.07);
    }}

    .persona-tabs {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .btn-tab {{
      cursor: pointer;
      border: 1px solid var(--border-strong);
      background: var(--bg-panel);
      color: var(--ink-sec);
      font-weight: 600;
      font-size: 14px;
      padding: 9px 16px;
      border-radius: 8px;
      transition: all 0.15s ease;
    }}

    .btn-tab:hover {{
      background: var(--accent-soft);
      color: var(--accent-dark);
    }}

    .btn-tab.active {{
      background: var(--accent-main);
      color: #FFF;
      border-color: var(--accent-dark);
      box-shadow: 0 2px 6px rgba(180, 83, 9, 0.25);
    }}

    .view-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}

    /* Timeline pills */
    .timeline-bar {{
      background: var(--bg-card);
      border: 1px solid var(--border-warm);
      border-radius: 12px;
      padding: 14px 18px;
      margin-bottom: 20px;
    }}

    .timeline-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--ink-muted);
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .timeline-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .s-pill {{
      cursor: pointer;
      border: 1px solid var(--border-warm);
      background: var(--bg-panel);
      color: var(--ink-main);
      padding: 6px 11px;
      border-radius: 7px;
      font-size: 12.5px;
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 118px;
      transition: all 0.15s;
    }}

    .s-pill:hover {{
      border-color: var(--accent-main);
      background: #FFF9EB;
    }}

    .s-pill.active {{
      border: 2px solid var(--accent-main);
      background: var(--accent-soft);
      font-weight: 600;
    }}

    .s-pill-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 12.5px;
    }}

    .s-pill-time {{
      font-size: 11px;
      color: var(--ink-muted);
    }}

    /* Role badges */
    .role-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11.5px;
      font-weight: 700;
    }}
    .role-evidence_session {{ background: var(--role-evidence-bg); color: var(--role-evidence-fg); }}
    .role-reinforcement_session {{ background: var(--role-reinforce-bg); color: var(--role-reinforce-fg); }}
    .role-reuse_session {{ background: var(--role-reuse-bg); color: var(--role-reuse-fg); }}
    .role-distractor_session {{ background: var(--role-distractor-bg); color: var(--role-distractor-fg); }}
    .role-correction_session {{ background: var(--role-correction-bg); color: var(--role-correction-fg); }}

    /* Session Card */
    .session-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-warm);
      border-radius: 14px;
      margin-bottom: 24px;
      overflow: hidden;
      box-shadow: 0 3px 12px rgba(92, 64, 40, 0.04);
    }}

    .session-header {{
      background: var(--bg-panel);
      border-bottom: 1px solid var(--border-warm);
      padding: 14px 22px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }}

    .session-header-left {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .session-id {{
      font-size: 17px;
      font-weight: 800;
      color: var(--accent-dark);
      font-family: "JetBrains Mono", Consolas, monospace;
    }}

    .session-time {{
      font-size: 13.5px;
      color: var(--ink-sec);
      background: var(--bg-subtle);
      padding: 3px 10px;
      border-radius: 6px;
      font-weight: 600;
    }}

    .session-body {{
      padding: 20px 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    .sub-block {{
      border: 1px solid var(--border-warm);
      border-radius: 10px;
      padding: 14px 18px;
      background: #FFFEFA;
    }}

    .sub-block-title {{
      font-size: 13.5px;
      font-weight: 700;
      color: var(--accent-dark);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px dashed var(--border-warm);
      padding-bottom: 6px;
    }}

    /* Dialogue turns */
    .turn-row {{
      margin-bottom: 8px;
      font-size: 13.8px;
    }}
    .turn-user {{
      background: #F7F9F0;
      border-left: 3px solid #65A30D;
      padding: 6px 12px;
      border-radius: 4px;
      margin-bottom: 4px;
    }}
    .turn-agent {{
      background: #FBF6EC;
      border-left: 3px solid #D97706;
      padding: 6px 12px;
      border-radius: 4px;
      color: var(--ink-sec);
    }}
    .gt-note {{
      margin-top: 8px;
      padding: 7px 12px;
      background: #FEF9E7;
      border: 1px solid #FDE68A;
      border-radius: 6px;
      font-size: 12.8px;
      color: #78350F;
    }}

    /* Search block */
    .search-box {{
      background: #FFF9EB;
      border: 1px solid #FCD34D;
    }}
    .sql-trace {{
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 12.5px;
      background: #2C2018;
      color: #FDE68A;
      padding: 8px 12px;
      border-radius: 6px;
      margin: 8px 0;
      overflow-x: auto;
    }}

    /* Extracted Events */
    .ev-item {{
      padding: 8px 12px;
      border-radius: 6px;
      background: var(--bg-panel);
      border-left: 4px solid var(--accent-main);
      margin-bottom: 8px;
      font-size: 13px;
    }}
    .ev-item.temp-veto {{
      background: #FEF2F2;
      border-left-color: #DC2626;
    }}
    .decay-banner {{
      padding: 8px 12px;
      border-radius: 6px;
      background: #FEF2F2;
      border: 1px solid #FCA5A5;
      color: #991B1B;
      font-size: 13px;
      font-weight: 600;
      margin-top: 6px;
    }}

    /* Table */
    table.mem-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.8px;
    }}
    table.mem-table th {{
      background: var(--bg-panel);
      color: var(--ink-sec);
      text-align: left;
      padding: 8px 10px;
      border-bottom: 2px solid var(--border-warm);
      font-weight: 700;
    }}
    table.mem-table td {{
      padding: 7px 10px;
      border-bottom: 1px solid var(--border-warm);
      vertical-align: top;
    }}
    table.mem-table tr:hover {{
      background: #FDF8EE;
    }}

    /* Status tags */
    .st-tag {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 700;
      font-family: "JetBrains Mono", Consolas, monospace;
    }}
    .st-reinforced {{ background: #DCFCE7; color: #166534; }}
    .st-active {{ background: #FEF3C7; color: #92400E; }}
    .st-candidate {{ background: #E0E7FF; color: #3730A3; }}
    .st-expired {{ background: #FEE2E2; color: #991B1B; text-decoration: line-through; }}

    .subtype-tag {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 4px;
      background: var(--bg-subtle);
      color: var(--ink-sec);
      font-family: "JetBrains Mono", Consolas, monospace;
      font-size: 11.5px;
    }}

    .final-search-section {{
      background: var(--bg-card);
      border: 2px solid var(--border-strong);
      border-radius: 14px;
      padding: 20px 24px;
      margin-top: 28px;
    }}

    .final-search-section h2 {{
      margin: 0 0 14px 0;
      font-size: 18px;
      color: var(--accent-dark);
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header-box">
      <h1>两个跨会话样本（USER_001 / USER_002）记忆萃取与检索全过程</h1>
      <p>
        完整展示每个样本从第 1 个 Session 到最后 1 个 Session 的长跨度演化过程：包含<b>原始对话</b>、<b>会话内 Turn-1 记忆检索（黑话展开 + 指针跳转 + 结构化 SQL SELECT）</b>、<b>记忆事件萃取与护栏拦截（临时纠偏不污染长期基线、单次偶发进观察期、噪声过滤）</b>以及<b>艾宾浩斯遗忘曲线演化快照</b>。
      </p>
    </div>

    <div class="toolbar">
      <div class="persona-tabs">
        <button class="btn-tab active" id="tab-p1" onclick="switchPersona('mobile_persona_001')">
          样本 1：mobile_persona_001 (USER_001 · 13 个 Session · 12天跨度)
        </button>
        <button class="btn-tab" id="tab-p2" onclick="switchPersona('mobile_persona_002')">
          样本 2：mobile_persona_002 (USER_002 · 14 个 Session · 39天跨度)
        </button>
      </div>
      <div class="view-controls">
        <button class="btn-tab active" id="btn-show-all" onclick="selectSession('ALL')">
          展开全部 Session 完整长流程
        </button>
      </div>
    </div>

    <div class="timeline-bar">
      <div class="timeline-title">
        <span id="timeline-summary">时间轴导航（点击任意 Session 可单独聚焦，或点击右上角「展开全部 Session」）</span>
        <span id="timeline-span"></span>
      </div>
      <div class="timeline-pills" id="timeline-pills"></div>
    </div>

    <div id="sessions-container"></div>

    <div class="final-search-section" id="final-search-container"></div>
  </div>

  <script>
    const REPORT = {embedded_json};
    let currentPersona = 'mobile_persona_001';
    let currentSessionFilter = 'ALL';

    const ROLE_NAMES = {{
      'evidence_session': '首次建档 / 新证据 (evidence)',
      'reinforcement_session': '记忆强化 (reinforcement)',
      'reuse_session': '跨期检索复用 (reuse)',
      'distractor_session': '噪声/单次干扰 (distractor)',
      'correction_session': '临时突发纠偏 (correction)'
    }};

    function switchPersona(pKey) {{
      currentPersona = pKey;
      currentSessionFilter = 'ALL';
      document.getElementById('tab-p1').classList.toggle('active', pKey === 'mobile_persona_001');
      document.getElementById('tab-p2').classList.toggle('active', pKey === 'mobile_persona_002');
      document.getElementById('btn-show-all').classList.add('active');
      renderAll();
    }}

    function selectSession(sid) {{
      currentSessionFilter = sid;
      document.getElementById('btn-show-all').classList.toggle('active', sid === 'ALL');
      renderTimeline();
      renderSessions();
    }}

    function renderTimeline() {{
      const pData = REPORT[currentPersona];
      document.getElementById('timeline-span').textContent = `时间跨度：${{pData.time_span}} （共 ${{pData.session_count}} 个会话）`;
      const container = document.getElementById('timeline-pills');
      container.innerHTML = '';

      pData.steps.forEach((st, idx) => {{
        const pill = document.createElement('div');
        pill.className = 's-pill' + (currentSessionFilter === st.session_id ? ' active' : '');
        pill.onclick = () => selectSession(st.session_id);

        const shortRole = st.memory_role.replace('_session', '');
        const shortTime = st.reference_time.replace('2026年', '');
        pill.innerHTML = `
          <div class="s-pill-top">
            <span>#${{idx + 1}} ${{st.session_id.split('-').slice(-1)[0]}}</span>
            <span class="role-badge role-${{st.memory_role}}" style="font-size:10px;padding:1px 5px;">${{shortRole}}</span>
          </div>
          <div class="s-pill-time">${{shortTime}}</div>
        `;
        container.appendChild(pill);
      }});
    }}

    function formatParams(content, subtype) {{
      if (subtype === 'expression' || (content.alias && content.target)) {{
        let parentTag = '';
        if (content.parent_subtype && content.parent_key) {{
          parentTag = ` &nbsp;<span style="background:#FEF3C7;border:1px solid #F59E0B;color:#92400E;padding:1px 7px;border-radius:4px;font-size:11.5px;">🔗 挂靠母记忆：<b>${{content.parent_subtype}}</b> [${{content.parent_key}}]</span>`;
        }}
        return `黑话映射：<b>${{content.alias}} → ${{content.target}}</b>${{parentTag}}`;
      }}
      if (subtype === 'service_to_app') {{
        return `业务默认应用：<b>${{content.business}} → ${{content.application}}</b>`;
      }}
      const parts = [];
      if (subtype === 'periodic_preference') {{
        if (content.time_window || content.condition) parts.push(`周期时段=<b>${{content.time_window || content.condition}}</b>`);
        if (content.application) parts.push(`app=<b>${{content.application}}</b>`);
        if (content.business) parts.push(`biz=<b>${{content.business}}</b>`);
        if (content.resolution) parts.push(`画质=<b>${{content.resolution}}</b>`);
        if (content.latency) parts.push(`时延=<b>${{content.latency}}</b>`);
        if (content.priority) parts.push(`优先级=<b>${{content.priority}}</b>`);
        return parts.join(' ， ');
      }}
      if (content.scene) parts.push(`场景=<b>${{content.scene}}</b>`);
      if (content.application) parts.push(`app=<b>${{content.application}}</b>`);
      if (content.business) parts.push(`biz=<b>${{content.business}}</b>`);
      if (content.resolution) parts.push(`画质=<b>${{content.resolution}}</b>`);
      if (content.latency) parts.push(`时延=<b>${{content.latency}}</b>`);
      if (content.priority) parts.push(`优先级=<b>${{content.priority}}</b>`);
      if (content.time_window) parts.push(`时段=<b>${{content.time_window}}</b>`);
      return parts.join(' ， ');
    }}

    function renderSessions() {{
      const pData = REPORT[currentPersona];
      const container = document.getElementById('sessions-container');
      container.innerHTML = '';

      const stepsToShow = currentSessionFilter === 'ALL'
        ? pData.steps
        : pData.steps.filter(s => s.session_id === currentSessionFilter);

      stepsToShow.forEach((st) => {{
        const card = document.createElement('div');
        card.className = 'session-card';

        // 1. Dialogue HTML
        const turnsHtml = st.dialogue_turns.map(t => `
          <div class="turn-row">
            <div class="turn-user"><b>User (T${{t.turn}})：</b>${{t.user}}</div>
            <div class="turn-agent"><b>Agent (T${{t.turn}})：</b>${{t.agent}}</div>
          </div>
        `).join('');

        const gtHtml = (st.gt_events || []).map(e => `
          <div class="gt-note">
            <b>样本标准设计意图 (${{e.event_type}})：</b>${{e.description}}
          </div>
        `).join('');

        // 2. Turn-1 Retrieval HTML (if applicable)
        let searchHtml = '';
        if (st.session_search && st.session_search.hits && st.session_search.hits.length > 0) {{
          const topHit = st.session_search.hits[0];
          const sqlMatch = (topHit.explain && topHit.explain.structured_match) || '向量+关键词混合检索';
          const hitRows = st.session_search.hits.map((h, i) => `
            <tr>
              <td><b>#${{i + 1}}</b></td>
              <td><span class="subtype-tag">${{h.subtype}}</span></td>
              <td><b>${{h.memory_key}}</b></td>
              <td><span class="st-tag st-${{h.status}}">${{h.status}}</span></td>
              <td><b>${{h.score.toFixed(4)}}</b> <span style="color:#846C58;font-size:11.5px;">(sim=${{h.explain.sim}}, R=${{h.explain.freshness}}, S=${{h.explain.stability_factor}})</span></td>
              <td>${{formatParams(h.content, h.subtype)}}</td>
            </tr>
          `).join('');

          searchHtml = `
            <div class="sub-block search-box">
              <div class="sub-block-title">
                <span>② 会话首轮 (Turn-1) 记忆检索过程（User 省略参数或使用黑话触发召回）</span>
                <span>查询时间：${{st.session_search.query_time}}</span>
              </div>
              <div style="font-size:13px;margin-bottom:6px;">
                <b>Turn-1 检索 Query：</b>「${{st.session_search.query}}」
              </div>
              <div class="sql-trace">两跳解析与结构化查询：${{sqlMatch}}</div>
              <table class="mem-table">
                <thead>
                  <tr>
                    <th style="width:48px;">排名</th>
                    <th style="width:155px;">记忆分栏 (Subtype)</th>
                    <th style="width:190px;">记忆主键 (Key)</th>
                    <th style="width:90px;">状态</th>
                    <th style="width:210px;">综合得分 (sim / R / S)</th>
                    <th>召回参数内容</th>
                  </tr>
                </thead>
                <tbody>${{hitRows}}</tbody>
              </table>
            </div>
          `;
        }}

        // 3. Extraction & Guardrails HTML
        let extractItemsHtml = '';
        if (!st.extracted_events || st.extracted_events.length === 0) {{
          extractItemsHtml = `
            <div class="ev-item">
              <b>未抽取出长期记忆事件（0 Events）：</b>该会话属于域外业务诉求（如宽带光纤报修），被 FastPath / Extractor 直接过滤，不写入任何记忆节点。
            </div>
          `;
        }} else {{
          extractItemsHtml = st.extracted_events.map(ev => {{
            const isTemp = ev.validity === 'temporary';
            const declMode = (ev.content && ev.content.declaration_mode) || '';
            let guardNote = '';
            if (isTemp) {{
              guardNote = `<div style="color:#991B1B;font-weight:600;margin-top:4px;">🛡️ 临时纠偏护栏生效：检测到「临时/今天现场特殊」一票否决词，validity 强制标为 temporary；Updater 拦截该事件，绝不覆盖长期基线记忆（1080p / 50ms 保持完好）！</div>`;
            }} else if (declMode === 'implicit' && st.memory_role === 'distractor_session') {{
              guardNote = `<div style="color:#6B21A8;font-weight:600;margin-top:4px;">🛡️ 防单次行为过拟合护栏：检测到「周末午休/今天上午」单次任务词且无长期声明，设为隐式观察态 (status=candidate, 需满2票才转正)，若后续14天未复现将自动按艾宾浩斯曲线过期淘汰。</div>`;
            }} else if (declMode === 'implicit') {{
              guardNote = `<div style="color:#92400E;font-weight:600;margin-top:4px;">⏳ 隐式首次行为记录：未显式说“老规矩/以后”，先进入 candidate 候选观察期（votes=1），等待后续会话二次确认后再转正。</div>`;
            }}
            return `
              <div class="ev-item ${{isTemp ? 'temp-veto' : ''}}">
                <div>
                  <b>事件类型：</b><code>${{ev.event_type}}</code> &nbsp;|&nbsp;
                  <b>目标分栏：</b><span class="subtype-tag">${{ev.subtype}}</span> &nbsp;|&nbsp;
                  <b>时效性：</b><code>${{ev.validity}}</code> ${{declMode ? `(mode=${{declMode}})` : ''}}
                </div>
                <div style="margin-top:4px;"><b>抽取参数：</b>${{formatParams(ev.content, ev.subtype)}}</div>
                ${{guardNote}}
              </div>
            `;
          }}).join('');
        }}

        let decayBannerHtml = '';
        if (st.ebbinghaus_decay && st.ebbinghaus_decay.expired_candidates && st.ebbinghaus_decay.expired_candidates.length > 0) {{
          decayBannerHtml = `
            <div class="decay-banner">
              📉 艾宾浩斯遗忘曲线自动淘汰触发：候选节点 ${{st.ebbinghaus_decay.expired_candidates.join(', ')}} 因超过 14 天观察窗口未获二次强化，保留率衰减至阈值以下，状态自动流转为 expired（已过期淘汰）！
            </div>
          `;
        }}

        // 4. Memory Snapshot Table HTML
        const memRowsHtml = st.memories.map(m => `
          <tr>
            <td><span class="subtype-tag">${{m.subtype}}</span></td>
            <td><b>${{m.memory_key}}</b></td>
            <td><span class="st-tag st-${{m.status}}">${{m.status}}</span></td>
            <td><b>${{m.votes}}</b> <span style="color:#846C58;font-size:11.5px;">(pos=${{m.positive_votes}}, reuse=${{m.reuse_count}})</span></td>
            <td><b>R=${{m.retention.toFixed(4)}}</b> <span style="color:#846C58;font-size:11.5px;">(S=${{m.stability_factor.toFixed(2)}})</span></td>
            <td>${{formatParams(m.content, m.subtype)}}</td>
          </tr>
        `).join('');

        card.innerHTML = `
          <div class="session-header">
            <div class="session-header-left">
              <span class="session-id">${{st.session_id}}</span>
              <span class="session-time">🕒 ${{st.reference_time}}</span>
              <span class="role-badge role-${{st.memory_role}}">${{ROLE_NAMES[st.memory_role] || st.memory_role}}</span>
            </div>
            <div style="font-size:12.5px;color:var(--ink-muted);">
              故事线：<b>${{st.blueprint_type}}</b> · 动作：<code>${{st.memory_action}}</code>
            </div>
          </div>
          <div class="session-body">
            <div class="sub-block">
              <div class="sub-block-title">
                <span>① 原始对话内容与样本测试目标</span>
                <span>共 ${{st.dialogue_turns.length}} 轮对话</span>
              </div>
              ${{turnsHtml}}
              ${{gtHtml}}
            </div>

            ${{searchHtml}}

            <div class="sub-block">
              <div class="sub-block-title">
                <span>${{searchHtml ? '③' : '②'}} 本轮记忆萃取结果与护栏判定 (FastPath + LLM Extractor + Updater)</span>
                <span>抽取事件数：${{(st.extracted_events || []).length}}</span>
              </div>
              ${{extractItemsHtml}}
              ${{decayBannerHtml}}
            </div>

            <div class="sub-block">
              <div class="sub-block-title">
                <span>${{searchHtml ? '④' : '③'}} 本轮结束后的用户记忆库全量快照（正交 4 分栏 + 场景分栏）</span>
                <span>当前落盘节点总数：${{st.memories.length}}</span>
              </div>
              <table class="mem-table">
                <thead>
                  <tr>
                    <th style="width:155px;">分栏 (Subtype)</th>
                    <th style="width:200px;">记忆主键 (Key)</th>
                    <th style="width:95px;">生命周期状态</th>
                    <th style="width:140px;">累计证据票数</th>
                    <th style="width:165px;">艾宾浩斯保留率 R (S)</th>
                    <th>存储内容（单一事实源，无冗余克隆）</th>
                  </tr>
                </thead>
                <tbody>${{memRowsHtml}}</tbody>
              </table>
            </div>
          </div>
        `;
        container.appendChild(card);
      }});
    }}

    function renderFinalSearch() {{
      const pData = REPORT[currentPersona];
      const container = document.getElementById('final-search-container');
      const queriesHtml = (pData.query_evaluations || []).map((q, qIdx) => {{
        const topMatch = (q.hits[0] && q.hits[0].explain && q.hits[0].explain.structured_match) || '';
        const rows = q.hits.map((h, i) => `
          <tr>
            <td><b>#${{i + 1}}</b></td>
            <td><span class="subtype-tag">${{h.subtype}}</span></td>
            <td><b>${{h.memory_key}}</b></td>
            <td><span class="st-tag st-${{h.status}}">${{h.status}}</span></td>
            <td><b>${{h.score.toFixed(4)}}</b> <span style="color:#846C58;font-size:11.5px;">(sim=${{h.explain.sim}}, R=${{h.explain.freshness}}, S=${{h.explain.stability_factor}})</span></td>
            <td>${{formatParams(h.content)}}</td>
          </tr>
        `).join('');

        return `
          <div class="sub-block search-box" style="margin-bottom:14px;">
            <div class="sub-block-title">
              <span>综合检索测试 #${{qIdx + 1}}：「${{q.query}}」</span>
              <span>检索时间：${{q.query_time}}</span>
            </div>
            <div class="sql-trace">两跳解析与 SQL SELECT：${{topMatch}}</div>
            <table class="mem-table">
              <thead>
                <tr>
                  <th style="width:48px;">排名</th>
                  <th style="width:155px;">分栏 (Subtype)</th>
                  <th style="width:190px;">记忆主键 (Key)</th>
                  <th style="width:90px;">状态</th>
                  <th style="width:210px;">综合得分 (sim / R / S)</th>
                  <th>召回内容</th>
                </tr>
              </thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        `;
      }}).join('');

      container.innerHTML = `
        <h2>🔍 样本全部会话结束后的跨期检索专项验证（${{pData.user_id}}）</h2>
        ${{queriesHtml}}
      `;
    }}

    function renderAll() {{
      renderTimeline();
      renderSessions();
      renderFinalSearch();
    }}

    renderAll();
  </script>
</body>
</html>
"""
    out_html_path.write_text(html, encoding="utf-8")
    print(f"[OK] Wrote clean warm-toned HTML report to {out_html_path}")


if __name__ == "__main__":
    build_html()
