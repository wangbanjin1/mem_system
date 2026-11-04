---
name: feedback-html-artifact-and-terminology-style
description: Visual, structural, modular sub-HTML, formula rendering, academic tone, and Lavish workflow standards for HTML technical reports
type: feedback
---

技术方案、设计维度溯源与评测报告优先交付**主从式多页双浅色主题 HTML 页面**（同步注册为会话 HTML Artifact 并写入 `.lavish/` 支持 `lavish-axi` 实时批注），严格执行客观学术/工程语态、CSS Grid 自适应架构图、克制公式与原生 Unicode 符号渲染，以及纯白底色 PDF 打印规范。

**Why:** 
1. **单页过载问题**：将方法论主干、单篇论文深度剖析与数十轮样本推演全部塞入单个 HTML 会导致主页面冗长失焦，难以一屏看清核心设计维度与论文来源；
2. **公式喧宾夺主与渲染乱码**：过度堆砌形式化公式会掩盖工程机制本质，且行内简单比较符（如 `$\ge 3$`、`$\mathcal{C}$`）若依赖外部 CDN 极易因网络或转义问题暴露出未渲染的 `$` 源码；
3. **语态与自证倾向**：口语化词汇（如“堵死作弊捷径”、“瞎编”）或刻意强调“★★★ 强实证直接支撑”会削弱技术文档的客观性；直接陈述“`<功能/因子>` —— 来源于 `<论文/标准>`”更具学术与工程说服力；
4. **定宽 SVG 截断与打印底色污染**：定宽 SVG `<rect>` 内嵌中文极易截断溢出；米纸主题（`#FBF7F0`）若未配置 `@media print` 白底，导出 PDF 时会在页面周围留下黄框。

**How to apply:** 
1. **主从式多页 HTML 架构与上下文原位跳转（Sub-HTML & In-Context Linking）**：
   - **主 HTML**：保持简洁、扼要、一针见血，聚焦核心架构与设计维度对照表；
   - **子 HTML（Sub-HTML）**：单篇核心论文深度拆解（如 `personamem_v3_deep_dive.html`）、长篇逐轮样本全景推演（如 `storyline_sample_walkthrough.html`）独立成页；在主页面**直接将上下文中的论文名称（如 `<a href="personamem_v3_deep_dive.html"><code>PersonaMem-v3</code></a>`）或样本描述文字设为内联超链接点击跳转**，不在顶栏或页面首尾额外堆砌独立跳转按钮。
2. **平实直白总结语态与三列维度溯源表（拒绝过度包装抽象词）**：
   - 全文保持普通总结思考文档的自然、直白语感，禁用口语化黑话、生硬的 `vs.` / `+` 拼接标题，**更严禁将简单机制包装成生涩抽象的学术造词**（如用“多次重复才记为长期习惯”代替“隐式证据累积门限”，用“单据标准名称与用户口语不一致”代替“线索-触发跨模态语义脱钩”，用“无关问题不强行带入个人记忆”代替“反过度个性化克制门控”）；
   - 阐述机制与论文关系时直接写“`<功能机制>` —— 参考 `<论文名称>`”，去除刻意强调的“支撑”或“来源于”，不空泛套用“标准”二字；
   - 维度表标题简洁直接（如 `七个设计维度及其影响因子`），统一采用三列表头：**`设计维度 | 影响因子与业务定义 | 参考论文`**，其中 `参考论文` 列直接写**“论文名字：该论文对应的大概工作”**，不堆砌繁琐的章节号/图表号。
3. **克制公式与零未渲染 `$` 源码铁律**：
   - 非核心不写复杂数学公式；行内符号、不等式与上下标（如 `≥ 3`、`±10min`、`Δk`、`→`、`R_NE`）统一使用原生 Unicode 与 HTML `<code>/<sub>/<sup>` 标签，确保离线/无 CDN 环境下 100% 零 `$` 乱码；
   - 确需保留的核心公式下方必须紧跟符号释义列表。
4. **视觉、防溢出卡片与 PDF 打印白底**：
   - 架构图统一使用 CSS Grid / Flexbox 自适应卡片 + 轻量 SVG 连接线；代码块统一使用浅色底（`#F2ECE1` / `#F6F8FA`）；
   - `@media print` 强制设置 `html, body { background: #FFFFFF !important; }` 与 `@page { margin: 12mm; }`。
5. **HTML Artifact 与 Lavish 协同工作流**：
   - 使用 `write_file`（含 `ArtifactMetadata`）在会话 Artifact 目录生成 `.html` 供 IDE 预览，同时同步至项目 `.lavish/`；
   - 用户需要交互式修改时，通过 `lavish-axi.cmd <path.html>` 启动本地服务并用 `lavish-axi.cmd poll <path.html>` 后台持续监听、自动应用批注并回传 `--agent-reply`。
