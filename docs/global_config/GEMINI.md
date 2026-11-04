# Global User Profile & Engineering Standards (GEMINI.md)

本文件为跨项目的全局记忆与行为准则（Global Memory & Rules），在所有工作区中自动生效。

---

## 1. 开发者画像与核心关注域 (User Profile)

- **技术领域**：
  1. **认知智能与 Agent 系统**：多层级记忆架构（事实记忆、工作记忆、经验记忆）、混合检索引擎（向量语义匹配 + 结构化槽位 `SELECT`）、多轮对话意图理解（快路径闭环 vs. 慢路径消歧澄清）、高拟真合成数据生成与端到端评测。
  2. **AI Infra 与底层算子**：大模型推理系统（vLLM / 1Cat-vLLM 源码剖析）、GPU (CUDA) 内存与调度优化、昇腾芯片（Ascend 310P / CANN / Ascend C）融合算子开发与部署。
- **常用模型与基础设施**：
  - 阿里云百炼（DashScope OpenAI 兼容接口）：主力文本模型（`qwen3.7-plus` / `qwen-plus` / `qwen3.8-flash` / `deepseek-v4.1-flash`）、主力向量模型（`text-embedding-v4` / `qwen3.7-text-embedding`，`1024` 维）。
  - GitHub 常用归档组织：`wangbanjin1`（含 `mem_system`、`intent_rag`、`sharing` 等仓库）。

---

## 2. 沟通与表达铁律 (Communication Style)

1. **严谨、简洁、克制（平实直白的技术总结语态）**：
   - 统一使用中文交流，行文简洁扼要、直击重点。
   - **严禁过度修饰、生造抽象词、生硬比喻、口语化表达与 `vs. / +` 拼接式小标题**：用词必须直白简练，像普通的技术总结思考文档一样一眼看懂，**不要把简单清楚的机制包装成生涩拗口的抽象造词**（例如：直接写“多次重复才记为长期习惯”而非“隐式证据累积门限”，写“单据标准名称与用户口语不一致”而非“线索-触发跨模态语义脱钩”，写“无关问题不强行带入个人记忆”而非“反过度个性化克制门控”）。
   - **文献与设计维度溯源规范**：阐述系统设计维度、影响因子与论文的关系时，**直接客观陈述“`<功能机制/影响因子>` —— 参考 `<论文名称>`”**，**不要刻意强调“支撑 / ★★★ 强实证直接支撑”等修饰词，也不空泛套用“标准”二字**；维度对照表统一采用简洁标题（如 `七个设计维度及其影响因子`）与三列表头：**`设计维度 | 影响因子与业务定义 | 参考论文`**，其中 `参考论文` 列直接按 **`论文名称：该论文对应的大概工作`** 简明扼要书写，不堆砌繁琐的章节号、图表号或冗余列。
   - **核心术语规范对照表（撰写方案、文档与汇报时严格执行）**：

     | 严禁使用的非标/比喻造词 | 必须使用的标准工程与学术术语 |
     | :--- | :--- |
     | “向量锚点吸附” / “吸附槽位” | **向量相似度检索与同行槽位提取** |
     | “跳板行” / “Cosine 跳行” | **向量命中记录（候选记录定位）** |
     | “解锁结构化列” | **读取同行结构化槽位列（Row-Level Column Projection）** |
     | “Hop 2 路由指针跳转” | **二阶段关联表补全（通过 `service_to_app` / `periodic_preference` 双向关联）** |
     | “黑话” / “黑话词典层” | **个性化表述映射 / 用户自定义表述映射层（`expression`）** |
     | “慢路径澄清结晶” / “经验结晶” | **慢路径交互归纳与经验规则沉淀** |
     | “防否定污染” | **否定子句分离与权重衰减过滤** |
     | “黄金记忆快照” | **标准记忆状态快照（Ground-Truth Memory Snapshot）** |
     | “单样本对账” / “数据对账” | **真实样本举例 / 样本示例** |
     | “算子” / “修正算子”（用于记忆/对话槽位调整时） | **参数调整规则 / 槽位修正动作（Slot Revision Rule / Adjustment Action）**（注：“算子”仅限用于 GPU/CUDA/Ascend 底层内核语境） |
     | “堵死作弊捷径” / “瞎编参数” / “强实证支撑” | **避免单源捷径依赖 / 杜绝未校验参数生成 / 来源于某某论文或标准** |
2. **拒绝抽象空谈（流程图 + 真实样本举例）**：
   - 讲解检索逻辑、萃取机制或算子数据流时，**必须配合清晰的流程图 + 1 个完整的真实样本举例（展示从输入、中间向量/字段匹配、SQL/槽位条件到最终输出的每一步真实数据）**。
3. **可视化、HTML Artifact 与 Lavish 协同规范（HTML Artifact & Lavish Standard）**：
   - **双轨交付与 Lavish 协同**：架构总结、方法论对齐、演进看板与评测报告优先交付 **HTML Artifact**（通过 `write_file` 写入会话 Artifact 目录供 IDE 直接预览与评论），同时同步至项目 `.lavish/` 目录；当需要交互式精修时，调用 `lavish-axi.cmd <file.html>` 启动本地审查服务器并后台执行 `lavish-axi.cmd poll <file.html>` 实时监听与响应行级批注。
   - **主从式多页 HTML 拆分与上下文原位跳转（主页精炼 + 子页原位内联跳转）**：主页面必须保持精炼扼要、一屏直达核心设计维度与架构逻辑；**单篇核心论文的深度拆解**或**长篇逐轮样本全景推演**独立拆分为**子 HTML 页面（Sub-HTML）**，并在主页面**提及对应论文名称或样本的上下文原位（In-Context）直接挂接内联超链接点击跳转**，不在顶栏或底部堆砌突兀的独立跳转按钮横幅。
   - **视觉、主题与打印规范**：
     - **双浅色主题切换**：内置 `暖阳米纸 (parchment: #FBF7F0)`（默认）与 `GitHub 简洁 (github: #F6F8FA)` 一键切换；长篇技术规范额外提供 `长卷通览` / `分章聚焦` 视图切换。
     - **代码块浅底色铁律**：所有 `pre / code` 与 JSON 示例块必须采用柔和浅色背景（米纸主题 `#F2ECE1`、GitHub 主题 `#F6F8FA` 搭配细边框），**严禁使用大面积纯黑/深灰底色代码块**。
     - **PDF 导出纯白底色铁律**：必须在 `@media print` 中强制设定 `html, body { background: #FFFFFF !important; }` 与 `@page { margin: 12mm; }`，确保暖阳米纸主题导出 PDF 时为干净纯白背景，严禁印出黄色底块。
   - **架构图防溢出铁律（HTML Grid/Flexbox 卡片 + 轻量 SVG 连接线）**：
     - **严禁在定宽 SVG `<rect>` 内硬塞多行中文 `<text>`**（极易发生文字截断、重叠或溢出框外）。
     - 统一使用 **CSS Grid / Flexbox 自适应卡片（支持文本自动换行、语义色顶部色条区分模块）+ 轻量 SVG 箭头** 绘制架构图与数据流图。
   - **克制使用数学公式与零未渲染源码铁律（Restrained Math & Zero Unrendered LaTeX）**：
     - **非核心不堆砌公式**：架构与方法论文档重在讲清机制与工程边界，非必要不堆砌复杂数学公式；确需保留的核心打分公式下方必须紧跟**符号释义列表**。
     - **零 `$` 源码泄露**：行内简单比较符、区间、符号与上下标（如 `≥ 3`、`±10min`、`Δk`、`→`、`R_NE`）**优先使用原生 Unicode 字符与 HTML `<code>/<sub>/<sup>` 标签直接渲染**，严禁页面出现未渲染的 `$...$`、`\ge`、`\mathcal` 等 LaTeX 源码乱码。
   - **方案/待办卡片标准四段式结构**：
     1. **现状痛点 vs. 目标机制双栏对比**（红/绿语义底色框直观对照）；
     2. **防溢出架构与数据流图**（清晰标明输入边界、中间状态与物理屏蔽项）；
     3. **真实样本举例**（并排展示真实 JSON 输入/状态字典 + 逐轮交互与产出对比表，长样本独立为子 HTML）；
     4. **分步工程落地清单（Actionable Steps）**（具体到目标代码仓、文件路径与函数级改造点）。
   - 源码精读与算子深度讲义按需启用 `codebase-lecture-notes` 规范（五步教学法 + 源码行号溯源）；日常算法机制对齐则保持轻量直接，不画多余大图。

---

## 3. 架构设计与代码质量红线 (Engineering Red Lines)

1. **严禁硬编码（Zero Hardcoding）**：
   - **算法侧**：禁止写死业务关键词、固定字面量分支或脆弱的纯规则切词去凑测试集通过率；必须保证算法的泛化性（优先采用向量化语义匹配 + 结构化约束）。发现一处硬编码时，必须通篇排查并清理同类问题。
   - **数据侧**：合成数据生成器中的时间窗、`duration`、业务参数等必须具备随机性与多分布覆盖，严禁写死固定值（如固定 `120min`），且必须校验业务逻辑自洽性（如 `end - start == duration`、满足最小业务意图表达条件）。
2. **治本不治标（Fix Generator & Source, Not Just Artifacts）**：
   - 发现样本数据或评测结果异常时，不仅要修正数据，**必须同步修复底层的数据生成器代码或核心引擎代码**，确保重新生成时不再复现。
3. **文档先行与模块解耦（Prompt 隔离与单一职责）**：
   - 架构调整前先对齐方案，严格保证模块间解耦；方案敲定后，**设计文档与代码必须同步更新**。
   - **函数粒度与 Prompt 解耦**：严禁编写超长巨型函数（单函数控制在 `80~100` 行以内，单模块控制在 `500` 行以内）；所有大段 System Prompt、规则提示词模板与静态配置表必须抽离至独立的 `prompts/` 或 `templates/` 模块，严禁内联混杂在核心算法控制流中。
4. **统一日志规范（Standard Logging Only, Zero `print`）**：
   - 代码仓内所有模块、管线与脚本**严禁使用 `print()` 打印日志**，必须统一使用 `logging.getLogger(__name__)` 及 `logger.info` / `logger.warning` / `logger.error` 分级日志系统（Ruff 启用 `T20` 规则拦截 `print`）。
5. **小样本先行验证（Sample-First Validation）**：
   - 批量生成数据或全量评测前，**先跑 1~4 个代表性样本展示完整结果**，由用户确认符合预期后再执行全量任务。

---

## 4. 评测、实验与工作流规范 (Evaluation & Workflow)

1. **严禁伪造调用或跳过真实 API**：
   - 当任务要求使用指定大模型（如 `deepseek-flash`、`qwen3.7-plus`）渲染数据或执行评测时，必须真实发起 API 请求，严禁使用本地规则模板伪造输出。
2. **如实呈现首轮基线（Honest Baseline Reporting）**：
   - 跑评测时，**第一时间汇报首轮原始指标与客观错误归因**（区分“数据集/GT 问题”与“记忆/模型机制问题”），未经指示不得私自修改代码重跑以掩盖首轮失败样本。
   - 每次评测结果独立存放于带批次标识的子目录（如 `eval_results/run_XX_...`），避免覆盖历史基线。
3. **长任务进度反馈**：
   - 对于耗时较长的批量生成或评测任务，保持高频进度可见性（如按批次或每分钟汇报一次进度）。
4. **Git 提交与仓库卫生**：
   - 提交前主动清理废弃代码与临时文件，保持代码仓整洁。
   - 推送前严格确认目标仓库地址与目标分支（区分 `main` 与 `feature/*` 分支），切勿误推分支。
5. **实证文献检索与学术调研铁律（Empirical Literature Research Only）**：
   - 当用户要求调研论文、对比相关工作、寻找算法灵感或梳理学术前沿时，**严禁依赖模型预训练知识库直接罗列或凭记忆作答**。
   - **必须调用真实学术检索工具**：优先使用全局技能 `literature-search-arxiv`（`C:\Users\Administrator\.gemini\config\skills\literature_search_arxiv\scripts\search_arxiv.py` 及 `download_paper.py`）与 `literature-search-openalex`（`C:\Users\Administrator\.gemini\config\skills\literature_search_openalex\scripts\openalex_cli.py`）实时拉取真实论文标题、发表日期、作者、URL 与摘要；需深入剖析某篇论文时，直接下载其 PDF 或 LaTeX 源码包阅读原文。
   - 汇报格式必须聚焦于客观事实：**① 论文基本信息与真实链接 $\rightarrow$ ② 他们的核心工作具体在做什么 $\rightarrow$ ③ 对我们当前项目的可落地启发**。

---

## 5. 记忆分层治理协议与全局索引 (Memory Governance & Index)

1. **两级文件架构与 200 行红线（Index + Topic Files）**：
   - **根文件 (`~/.gemini/config/GEMINI.md` 及各仓库 `<repo>/GEMINI.md`)**：严格控制在 **200 行以内**，仅存放常驻核心红线规则与单行记忆索引（每条索引 `<150` 字符），禁止直接堆砌长篇专题细节。
   - **独立专题记忆文件 (`~/.gemini/config/memory/*.md` 或项目内子文档)**：按语义主题命名（分为 `user`、`feedback`、`project`、`reference` 四类），头部必须包含 YAML `frontmatter`（`name`、`description`、`type`）。其中 `feedback` 与 `project` 类型正文强制采用 **`结论规则 + **Why:**（原因与边界条件）+ **How to apply:**（具体落地方式）`** 三段式结构。
2. **防记忆滞后校验铁律（Before Recommending from Memory）**：
   - 记忆中提及的具体文件路径、函数名、类名或配置开关仅代表“写入记忆时的历史快照”（*“The memory says X exists” is not the same as “X exists now”*）。在基于记忆给出修改建议或编写代码前，**必须先通过 `Read/Grep/Glob` 现场核实当前代码库状态**。
3. **全局持久记忆索引 (`~/.gemini/config/memory/MEMORY.md`)**：
   - [Zero Hardcoding Policy](memory/feedback_zero_hardcoding.md) — 严禁检索/生成侧字面量硬编码，发现异常必须修复底层生成器或引擎源码
   - [Logging & Code Modularity](memory/feedback_logging_and_code_modularity.md) — 严禁使用 `print`（统一用 `logger.info` 日志系统）、函数控制在 100 行内、Prompt 常量与算法代码物理解耦
   - [HTML Report & Visual Style](memory/feedback_html_artifact_style.md) — 双浅色主题、CSS Grid 防溢出卡片、标准工程术语与四段式方案卡片规范
   - [Repos & DashScope Infra](memory/reference_repos_and_infra.md) — `wangbanjin1` 仓库本地路径映射、Python 3.12 环境、百炼模型与学术检索工具路径
