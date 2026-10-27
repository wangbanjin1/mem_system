# Memory System 复刻指南（面向其他模型 / 新实现团队）

> 目标读者：想**用另一套模型 / 另一种语言 / 从零重建**这个 Agent 记忆系统的人。
> 本文不重复 `design.md` / `implementation_plan.md` 的完整设计推导，而是聚焦：
> **核心机制长什么样 → 按什么顺序做 → 最容易踩的坑在哪 → 怎么验证自己没做歪**。
> 配套源码即本仓库 `src/memory_system/`，文中标注的 `path:line` 指向关键实现位置。

---

## 0. 一句话概括

这是一个**「文件即真相 + SQLite 二级索引」**的 Agent 记忆系统：
每次会话归档 → 确定性规则（Fast Path，免 LLM）+ LLM 结构化抽取 → 一个**纯 Python 状态机**（Updater）把事件合并成记忆节点 → 检索时**混合打分**（向量 + 关键词 + 热度 + 新鲜度）→ 定期维护（淘汰 / 压缩 / 归档）。

最近一次大改（commit `b3d494b`）给它加了 **三层认知记忆**（事实 / 工作 / 经验）+ **热度激活（heat）** + **按类型的艾宾浩斯 λ 衰减**。

复刻它的**最小骨架**（不做 LLM 也能跑通）只有 5 件事：

1. `memory_key` 归一（去重 / 强化 / 争议 / 取代的锚点）
2. 确定性状态机 `MemoryUpdater`（vote + 状态跃迁）
3. 文件落盘 + 可重建的向量/全文二级索引
4. 混合检索打分（α·sim + β·freshness + γ·heat + δ·conf）
5. LLM 抽取 + **确定性校验**（schema / 值域 / support matrix / evidence）

其余（Fast Path、路由、图、队列、维护、三层记忆）都是在此之上的增强。

---

## 1. 架构总览

### 1.1 分层

```
┌──────────────────────────────────────────────────────────────┐
│ 门面 MemorySystem (src/memory_system/api.py)                │
│   唯一公开入口；同步/异步两种模式；组装下面所有组件           │
├──────────────────────────────────────────────────────────────┤
│ 写入管线 (pipeline/)                                         │
│   archiver → fastpath(免LLM) → extractor(LLM+校验)           │
│            → updater(状态机) → induction(跨会话归纳)          │
├──────────────────────────────────────────────────────────────┤
│ 检索 (retrieval/)                                            │
│   router(LLM路由) → search(过滤→混合召回→重排→分层加载)       │
├──────────────────────────────────────────────────────────────┤
│ 存储 (storage/)                                              │
│   filestore(权威文件) + index(SQLite 向量+FTS5) + graph(jsonl)│
├──────────────────────────────────────────────────────────────┤
│ 外部依赖 (providers/)  → OpenAI 兼容 HTTP 的 embedding / llm │
│ 配置 (core/config.py + config/)  → JSON + YAML 注册表         │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 一次会话写入的完整时序（最重要的一张图）

```
record_session(tenant, user, agent, session)
  │
  ├─[同步] 1. archive_session        → sessions/{sid}/history.jsonl + meta.json
  ├─[同步] 2. fastpath.process        → 规则产出事件（别名/槽值/纠正/噪声）
  │          ├─ confirmed（无需LLM）  → 立即 _apply_candidate 落库
  │          └─ needs_llm / 含参数信号 / 含让步 → 3
  ├─[同步或异步] 3. LLM 抽取
  │          extractor.extract        → LLM 输出 events[]
  │          _backfill_time_window    → 确定性正则回填时段（LLM 常漏）
  │          _normalize_validity      → 确定性改写临时/长期标记（防污染关键！）
  │          extractor.validate       → 纯 Python 校验，不合格直接丢弃+记日志
  │          → _apply_llm_event       → updater.apply_event 落库
  ├─[同步] 4. induction（最后做！）     → 基于最新节点做跨会话归纳
  └─ 返回 task_id
```

**关键顺序**：induction 必须放最后（`api.py:222-226`），因为 LLM 刚可能把某个节点提升到 active，归纳要看到最新状态。

### 1.3 命名空间与隔离（复刻时最容易做错的地方之一）

- **隔离键只有 `user_id`**，`tenant_id` 只是元数据（`core/namespace.py:27-31`、`storage/filestore.py:107-114`）。
- 路径：`users/{user_id}/memory/...`、`users/{user_id}/agent_experience/{agent_id}/...`。
- Agent 经验**按用户隔离**，不跨用户共享（`api.py:296-325`）。
- 所有 SQL 查询必须带 `owner_type + user_id (+ agent_id)` 作用域过滤（`storage/index.py:157-171`），**绝不允许全局相似度扫描**。

> ⚠️ 如果复刻时把 `tenant_id` 也塞进路径/索引键，你会得到"看似更安全但和原系统对不上"的隔离。原设计是 `user_id` 全局唯一，`tenant_id` 只用于审计。

---

## 2. 核心机制详解

### 2.1 memory_key：一切归一的锚点

- 由 `MemoryTypeSpec.memory_key_fields` 的**固定有序字段列表**拼接，分隔符 `" | "`，缺省值渲染为 `"_"`（`core/memory_key.py`）。
- 例：`application_business` 的 key = `application | business` → `douyin | live`；`periodic_preference` 的 key = `condition | value`。
- **同一 (namespace, memory_key) 就是一个记忆节点**。Updater 按它定位：没有就创建，有就 merge/强化/争议/取代。
- 节点目录名 = `sha256(memory_key)[:12]`（`storage/filestore.py:58-59`），不是明文 key。

> 坑：memory_key 字段的**顺序**就是 key 的格式。复刻时改了字段顺序 = 改了所有历史 key，索引重建后节点会"漂移"成新的。

### 2.2 状态机 MemoryUpdater（纯 Python，无 LLM）

节点状态机（`core/models.py:20-31`）：`candidate → active → reinforced / disputed / superseded / expired / archived`。

每次事件触发（`pipeline/updater.py`）：

1. **找节点**：按 `(namespace, subtype, memory_key)` 遍历文件（`_find_by_key`）。
2. **变体污染防线**（先于一切 merge）：临时事件不覆盖长期节点（见 §4 坑 2）。
3. **merge content**：非空即覆盖（`_update`），但保留旧值若新值为空。
4. **累计 vote**：`_VOTE` 表给出每个 `event_type` 的 `(vote_delta, action, heat_delta)`（`updater.py:38-61`）。
5. **状态跃迁**：`_transition`（`updater.py:139-167`）—— 投票达 `vote_threshold` → active；correct → disputed（concession 除外）；clear → archived；archived/expired 可被再断言复活。
6. **热度叠加**：先按半衰期衰减旧热度，再叠加本次信号（`_apply_heat`）。
7. **落盘**：memory.json + content.md + history.jsonl（append-only），更新索引，写图边（SUPPORTS）。

**事件类型 → 动作/热度的权重表**（复刻时建议原样照抄）：

| event_type                              | vote | action    | heat |
| --------------------------------------- | ---- | --------- | ---- |
| PREFERENCE_SET / EXPRESSION_ALIAS / …   | +1   | set       | +0.2 |
| PREFERENCE_CONFIRM                      | +1   | reinforce | +0.3 |
| PREFERENCE_CORRECT                      | +1   | correct   | −0.3 |
| PREFERENCE_CLEAR                        | −1   | clear     | −0.3 |
| WORKING_RECORD                          | +1   | set       | +0.1 |
| WORKING_RECORD_REUSE / EXPERIENCE_APPLY | +1   | reinforce | +0.3 |
| EXPERIENCE_CRYSTALLIZE                  | +1   | set       | +0.2 |

> 注意：`_VOTE` 里 `PREFERENCE_SET` 等热度和设计文档 §6 的"检索命中 +0.02 / 加入上下文 +0.05"是**两套机制**——事件写入叠加信号权重，检索命中单独 `bump_heat`（+0.05）。

### 2.3 热度激活（heat）——本次新增的核心

- 每个节点有 `stats.heat` + `stats.last_heat_at`。
- **写入时**：事件信号权重叠加（上表）。
- **读取时**：`effective_heat = heat × 0.5^(Δt_hours / half_life)`（`retrieval/search.py:167-182`），半衰期默认 168h。
- **检索命中**：`bump_heat(node, +0.05)` 回写（`api.py:158` 把 `updater.bump_heat` 注册为 `heat_callback`；`search.py:137-143` 在返回前调用）——**只改热度，不改 vote/status**。

### 2.4 艾宾浩斯动态稳定性曲线与双向遗忘闭环

- **间隔复习动态稳定性公式**：
  $$\text{freshness} = R(t, n) = \exp\!\left(-\frac{\lambda_0 \cdot \Delta t_{\text{days}}}{1 + \kappa \cdot \ln(1 + n)}\right), \qquad n = \max(0, \text{positive\_votes} - 1) + \text{reuse\_count}$$
  其中 $\Delta t_{\text{days}}$ 距 `last_reinforced_at`（天），$\kappa=0.8$ 为复习间隔稳定系数。强化次数 $n$ 越多，有效衰减率 $\lambda_{\text{eff}} = \lambda_0 / (1 + \kappa \ln(1+n))$ 越低，遗忘曲线越平缓。
- **$\lambda_0$ 从 registry 的 `decay_lambda` 读取**（`search.py`）：
  - `application_business` = 0.02、`periodic_preference` = 0.02、`expression` = 0.0。
  - `working_record` = 0.0（工作单据**不衰减**）、`experience` = 0.0、`environment`/`time_pattern` = 0.05。
- **双向状态退化（真遗忘闭环）**：
  - `candidate` 节点若超过观察期（默认 14 天或 $R(t,0) < 0.70$）仍未复现第 2 票，自动退化为 `expired`（淘汰单次偶然行为噪音）；
  - `reinforced` 节点若超长周期（默认 90 天且 $R(t,n) < 0.25$）未复用，自动退化回 `active`。

### 2.5 两跳黑话展开 + 结构化槽位 `SELECT` + 检索混合打分

```
score = α·sim + β·freshness + γ·heat + δ·confidence
```

- 默认 `α=0.5 β=0.2 γ=0.1 δ=0.2`（`config/config.json`）。
- **三步协同召回**：
  1. **Query 黑话两跳展开与槽位挖掘**：先扫描用户已激活的 `expression` 节点，将 Query 中出现的黑话（如 `午间例行`、`阿抖`、`顺畅推流`）展开为标准槽位值（`12:30-13:30`、`抖音`、`50ms`），再结合 `app_map` 与时间/场景正则挖掘出 `(applications, businesses, scenes, time_windows)`；
  2. **指针顺藤摸瓜与 SQLite 结构化 `SELECT`**：若只提时段未提应用，先查 `periodic_preference` 指针解析出 `(application, business)`；若只提业务未提应用，先查 `service_to_app` 指针解析出 `application`；随后直接在 SQLite 结构化列（`app_col, biz_col, scene_col, time_window_col`）上 `SELECT` 唯一权威的参数节点，赋予确定性高相关性分；
  3. **Dense 向量 + FTS5 语义兜底融合**：与余弦向量检索及 FTS5 全文检索结果合并去重取 `sim` 最大值。
- **重排**：按公式排序，临时节点（validity=temporary）整体 ×0.5 降权（`search.py`）。
- **分层加载**：`levels="L0"`（description）/ `"L0+L1"`（+概览行）/ `"full"`（完整 content.md）。
- 结果带 `explain`（含 `structured_match` 槽位解析轨迹、`sim`/`freshness`/`stability_factor`/`heat`/`final_score`/`memory_key`），满足完全可解释性。

### 2.6 LLM 抽取 + 确定性校验（复刻的重头戏）

一次会话**一次 LLM 调用**，输出固定 schema 的 `events[]`（`pipeline/extractor.py`）。

**必须做的三道确定性后处理**（LLM 输出不可信，全靠这些兜底）：

1. **`_backfill_time_window`**（`extractor.py:226-259`）：LLM 常漏萃时段，用正则从对话回填 `time_window`（"12时30分开始、13时30分结束"→"12:30-13:30"）。**time_window 是"老规矩"轮推绝对时间的唯一依据**，漏了就全错。
2. **`_normalize_validity`**（`extractor.py:263-312`）：用确定性词表重写事件 validity。`临时` 一票否决 → temporary；强长期词（以后/默认/一直…）→ long_term；弱长期词（老规矩/照旧…）→ long_term；都没命中保留 LLM 原值。
3. **`validate`**（`extractor.py:314-384`）：纯 Python 校验——event_type 闭集、subtype 存在、content schema、memory_key 字段齐全、resolution 值域、support matrix、evidence_turns 真实存在。不合格 → 丢弃 + 记日志（不重试）。

**LLM 抽取提示词要点**（`prompts/extract_generic.md`）：

- `memory_key_fields` 必须全填（null 会被丢）；content 里要重复出现 key 字段。
- 应用名归一到 `app_map` 的 key（"三角洲行动"→sanjiaozhou），业务用归一中文名。
- 时段必须萃。
- 一次性临时要求不要抽成长记忆。

### 2.7 三层认知记忆（事实 / 工作 / 经验）与用户层 5 类正交记忆

| 层       | subtype                                                      | memory_key_fields         | vote_threshold | 特点（遵循正交无冗余方案 A）                       |
| -------- | ------------------------------------------------------------ | ------------------------- | -------------- | -------------------------------------------------- |
| 事实记忆 | `preferred_value`（默认参数偏好）                            | `[application, business]` | 1              | 只管 `(app, biz)` 下的通用 QoS 参数（`resolution/latency`），不含时间段 |
| 事实记忆 | `scene_preference`（场景参数覆写）                           | `[scene, application, business]` | 1       | 只管特定场景下的 QoS 参数覆写，不含周期时段        |
| 事实记忆 | `periodic_preference`（周期时间路由指针）                    | `[condition, value]`      | 2              | 只管 `time_window -> app:biz` 路由，绝不存画质/时延参数 |
| 事实记忆 | `service_to_app`（应用↔业务默认路由指针）                   | `[business]`              | 1              | 只管 `business <-> application` 默认绑定关系       |
| 事实记忆 | `expression`（个性化黑话词典）                               | `[alias]`                 | 1              | 只管黑话到标准槽位值的映射（`alias -> target`），λ=0 |
| 工作记忆 | `working_record`                                             | `[record_id]`             | 1              | 历史工单流水，`record_id` 保证幂等；λ=0 不衰减     |
| 经验记忆 | `experience`                                                 | `[experience_id, domain]` | 1              | 树形分支：根=domain、支=strategy、叶=experience_id |

**批量入库入口**（不走 LLM，直接构造事件走完整状态机）：

- `ingest_working_records(tenant, user, records)`（`api.py:329-366`）——每条 `WORKING_RECORD` 事件，`memory_key_fields={record_id}`，重复 ingest 同一单据只 reinforce 不重复创建（幂等）。
- `ingest_experience_cards(tenant, user, cards)`（`api.py:368-416`）——`EXPERIENCE_CRYSTALLIZE` 事件，`memory_key_fields={experience_id, domain}`。无 domain 时按 `strategy_type` 映射到预设 domain 兜底。

> 经验记忆的落盘样例见 `data_roots_8902_v7/users/USER_001/memory/experience_memory/experience/<hash>/memory.json`——`trigger_conditions.keywords` 是检索命中经验的关键字段。

### 2.8 写入侧正交拆解与跨会话指针归纳（确定性，无 LLM，严格遵循单一真相原则）

1. **写入侧正交拆解（`_decompose_to_orthogonal_events`）**：
   - 单次会话抽出的业务偏好事件在入库前被正交拆解为：
     - **参数节点**：若有特殊物理/任务场景则写入 `scene_preference[scene | app | biz]`，否则写入 `preferred_value[app | biz]`（仅保存 `resolution`/`latency` 等 QoS 参数，绝不耦合 `time_window`，且不再冗余生成空壳 `application_business`）；
     - **时间路由节点**：若包含时间段则独立写入 `periodic_preference[time_window | app:biz]`（仅保存时间路由指针，绝不冗余拷贝 `resolution`/`latency`）。
2. **跨会话应用↔业务路由归纳（`induce_service_to_app`）**：
   - `pipeline/induction.py` 扫描已转正的参数节点，当某业务只配一个应用且 $\ge 2$ 次观测时，幂等生成 `service_to_app` 指针（`business <-> application`）。

---

## 3. 复刻的最小可行路径（建议顺序）

> 目标：**每一步都可独立运行、可验证**，不引入 LLM 也能看到系统工作。

### Step 1：模型 + 配置骨架（半天）

- 定义 `MemoryNode` / `Event` / `Stats` / `ValidityInfo` 数据类（照抄 `core/models.py`）。
- 定义 `MemoryTypeSpec` 注册表加载（YAML），`memory_key` 构建/解析。
- 定义 `Config`（JSON + 默认值深合并，`core/config.py`）。
- **验证**：`load_registry` 能读到 15 个 subtype；`build/parse` 往返一致。

### Step 2：文件存储 + 状态机（1-2 天）

- `FileStore`（原子写、路径白名单、per-key 锁）+ `MemoryUpdater`（create/update + 状态跃迁 + 热度）。
- 先不接 LLM，用手写事件直接调 `updater.apply_event`。
- **验证**：跑 `tests/test_updater.py`、`tests/test_activation.py` 的逻辑（candidate→active、correct→disputed、clear→archived、heat 半衰期）。

### Step 3：索引 + 检索（1 天）

- SQLite 向量表 + FTS5（`storage/index.py`），numpy 余弦（按 namespace 过滤后算）。
- `SearchEngine` 混合打分 + 分层加载。
- **验证**：`tests/test_search.py`、`tests/test_index.py`；跨用户隔离用例（`test_api_e2e.py::test_cross_user_isolation_full_flow`）。

### Step 4：LLM 抽取 + 校验（1-2 天，最考验提示词）

- `Extractor` + `LLMProvider`（OpenAI 兼容）+ 三件套后处理（回填时段 / 归一 validity / validate）。
- **验证**：用 mock LLM 跑 `tests/test_extractor.py`、`tests/test_temporary_pollution.py`。

### Step 5：Fast Path + 路由 + 队列 + 维护（各半天）

- Fast Path 规则（别名/槽值/纠正/噪声），`record_session` 组装完整时序。
- Router（LLM 5 选 1 + 别名短路）。
- 文件任务队列 + 后台 worker（`pipeline/queue.py`）。
- 维护调度（decay / candidate_timeout / eviction / timeline_compress）。
- **验证**：`tests/test_fastpath.py`、`tests/test_api_e2e.py` 全绿；`pytest tests/` 91 个用例。

### Step 6：三层认知记忆 + FastAPI 服务（各半天）

- `working_record` / `experience` 两个注册表 + 两个 ingest 入口 + `memory_types` 检索过滤。
- FastAPI 薄封装（`server/app.py`，8 个端点 + 环境变量）。
- **验证**：`scripts/verify_three_tier.py`（对 8902 服务发 4 类检索）能命中事实/工作/经验三层。

---

## 4. 踩坑清单（按严重程度排序）

### 坑 1：LLM 抽取不可信 → 必须有确定性后处理

**症状**：同一句话反复抽、抽错时段、抽错值域、抽出不存在的轮次。

**对策**（本仓库的做法）：

- 抽取后立即 `_normalize_validity` / `_backfill_time_window`，LLM 填的 `validity` 直接覆盖（`api.py:268-280` 注释里写得很清楚："LLM 填的 validity 不可信，必须覆盖 content 里 LLM 自带的 validity，否则临时值覆盖长期节点"）。
- `validate` 逐条丢弃不合格事件，**不重试**（重试只会放大 token 成本）。
- `evidence_turns` 必须真实存在，防止 LLM 引用不存在的轮次。

### 坑 2（最贵）：变体污染——临时/一次性值覆盖长期老规矩

详见 `fix_notes/variant_pollution.md`。这是记忆服务踩过的最深坑，直接导致评测轮数预测错误。

**根因链条**：

1. LLM 把"今天临时调 720p"抽成 `validity=long_term` 的 `preference_set`。
2. Updater `_update` 曾**无条件 merge** 非空值（旧代码 `updater.py:218-222`）。
3. 变体值覆盖同 memory_key 的长期节点；下个老规矩轮检索引用到污染值。

**修复三件套（复刻时必须带上）**：

1. **萃取侧** `_normalize_validity`：
   - 含"临时"→ 一票否决 temporary（即使同句有"老规矩"/"以后"）；
   - 强长期词（以后/默认/一直/总是…）→ long_term；
   - 弱长期词（老规矩/照旧/平时/通常/一般/惯例）→ long_term；
   - 判定只看**用户原话**（同 turn 里 agent 复述"长期配置"之类系统话术会误判，`extractor.py:285-294`）。
2. **更新侧**：`event validity=temporary 且节点非 temporary` → **no-op**（不 merge、不加 vote、不改状态，记 `event_ignored_temporary`，`updater.py:280-286`）。
3. **检索侧**：temporary 节点 score ×0.5（兜底"首个 session 就是变体"，`search.py:163-164`）。

配套：`periodic_preference` 的 `vote_threshold` 从 1 提到 2（多观测确认）；prompt 加一行 validity 指引。

**回归测试**：`tests/test_temporary_pollution.py` 的三连回归（老规矩 A → 变体 → 老规矩 B，检查 B 轮检索仍是长期值）。

### 坑 3：Fast Path 短路 LLM 导致参数丢失

**症状**：对话里显式声明了参数（"画质 1080p"），却因为 fastpath 确认了别名事件而跳过 LLM，参数全丢。

**对策**（`api.py:52-63, 199-205`）：

- 维护 `_PARAM_SIGNAL_RE`（匹配 1080p/720p/ms/画质/分辨率/时延/老规矩…），命中即强制走 LLM。
- 含让步轮（"行，那就按X办吧"）也强制走 LLM，让纠正后的最终值被抽成普通偏好。

### 坑 4：description 冻结 / 与 content 不一致

**症状**：节点 content 已更新但 description（L0）还是旧的，检索结果和详情对不上。

**对策**：`_update` 里对比 **merge 前**的 content 是否带新 note（`updater.py:287-292`）。"merge 后比较"是 bug——merge 后新旧是同一个 dict，比较永远不会触发。

### 坑 5：time_window 漏萃 → 老规矩轮推不出绝对时间

**症状**："老规矩"轮要推绝对时间，但记忆里没有时段。

**对策**：`_backfill_time_window` 用 7 条正则覆盖中文时段表达（"12时30分开始、13时30分结束"、"从12点半到1点半"、"18:45至19:45"、时长推算跨零点…），只回填 LLM 留空字段；优先在 evidence_turns 对应轮次找，找不到退回全文。

### 坑 6：Windows 并发写入 / 原子性

- 原子写：`.tmp.<pid>.<tid>` + `os.replace`，失败重试 5 次（Windows 下 `PermissionError` 常见，`filestore.py:82-100`）。
- `sqlite3` 连接**不能跨线程并发**（FastAPI 线程池会并发调 upsert/search），必须 `RLock` 包住所有连接访问（`index.py:47-49`）。
- 每 (namespace, memory_key) 一把进程内锁串行化更新。
- `history.jsonl` 追加也要独立锁。

### 坑 7：SQLite FTS5 外部内容表同步

- FTS 是 `content='vector'` 外部内容表，`upsert` 要按 rowid 同步（`index.py:130-137`）。
- **删除**必须用 FTS5 的 `'delete'` 特殊命令并带上**旧值**（`index.py:140-155`），否则删除后关键词仍命中幽灵行。
- 老库缺 `last_heat_at` 列时做 `ALTER TABLE` 迁移（`index.py:79-82`）。

### 坑 8：时区 / 时间解析

- 存储用 UTC 字符串 `YYYY-MM-DDTHH:MM:SSZ`，但解析用 `time.mktime`（本地时区）——测试里特意用 `localtime` 生成 ts 才能精确往返（`test_activation.py:146-151`）。
- 所有 `strptime(...)[:19]` 截断处理，格式不合法时返回安全默认值，**不抛异常**。
- 时钟回拨（Δt≤0）不衰减。

### 坑 9：图用 JSONL 全量重写

- `graph/edges.jsonl` 每次 `add_edge` 都**全量读→追加→全量写**（`graph.py:43-54`）。量小没事，量大就是 O(n) 每次写。复刻时如果量大请换 append 或数据库。
- `neighbors()` 做 1-2 跳 BFS，边按 namespace 过滤。

### 坑 10：LLM 服务健壮性

- `httpx.Client(trust_env=False)`（不走代理，对齐参考客户端）。
- 超时 `(10s 连接, 120s 读)`；重试退避 `2^n` 秒。
- 5MB 响应上限防 OOM（`llm.py:21`）。
- embedding 数量不匹配要报错（`embedding.py:57-59`）。

### 坑 11：路径安全 / 用户身份脱敏

- 所有路径片段白名单 `^[A-Za-z0-9._-]+$`，防目录穿越（`filestore.py:43-55`）。
- Agent 经验写入前过 `CaseSanitizer`：剥离 tenant_id/user_id/session_id/raw 等身份字段（`sanitizer.py:20-23`），即使隔离出问题内容也不带身份。

### 坑 12：三层记忆的检索隔离

- 用 `memory_types` 参数过滤 subtype（`search.py:109-110`）；工作/经验/事实混在一个命名空间里，不隔离检索会互相串。
- `working_record` 的 λ=0（不随时间衰减），因为它是**流水账**不是偏好，别给它套衰减。
- 经验记忆的 `vote_threshold=1`，入库即 ACTIVE（它是预生成的卡，不是要观察确认的行为）。

---

## 5. 换模型/换服务时要调的配置

| 配置项                                | 位置                                       | 说明                                                         |
| ------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| embedding `base_url/model/dimension`  | `config/config.json`                       | 换 embedding 模型主要改这里；dimension 变了要重建索引        |
| llm / llm_router `base_url/model`     | 同上                                       | 抽取和路由可分开用不同模型                                   |
| 检索权重 α β γ δ                      | 同上                                       | 语义为主就加大 α，时效为主就加大 β                           |
| heat 半衰期                           | `maintenance.heat_half_life_hours`         | 168h=7 天                                                    |
| 各类型 λ                              | `config/registry/*.yaml` 的 `decay_lambda` | 按类型调遗忘速度                                             |
| vote_threshold                        | 各 registry yaml                           | 多少观测算确认                                               |
| app_map / app_config / support_matrix | `data/*.json`                              | 换业务域就换这 3 个文件，**Fast Path 和 LLM prompt 共用同一份**，保证值域一致 |

**换 embedding 模型的坑**：

- 维度变了必须 `rebuild_index()`（`api.py:508-509`）。
- 向量语义变了，老的 sim 排序会变——建议换模型后跑一遍评测回归。

---

## 6. 验证方法（如何确认自己没做歪）

### 6.1 离线单测（不碰网络）

```bash
cd <repo>
pip install -e ".[dev]"
pytest tests/ -q          # 91 个用例全离线，mock embedding/llm
```

关键测试文件对照：

| 文件                                | 覆盖                                                     |
| ----------------------------------- | -------------------------------------------------------- |
| `tests/test_updater.py`             | 状态机 / vote / 历史 / 索引                              |
| `tests/test_activation.py`          | heat 初始值 / 半衰期 / 回写 / λ 按类型 / 三层记忆 ingest |
| `tests/test_temporary_pollution.py` | 变体污染三连回归 / validity 归一 4 例 / 检索降权         |
| `tests/test_search.py`              | 混合打分 / 分层加载                                      |
| `tests/test_api_e2e.py`             | 全链路 / 跨用户隔离 / 重建索引 / 维护                    |

### 6.2 端到端冒烟（需内网服务）

```bash
python server/app.py                     # 或 uvicorn server.app:app --port 8900
# 另一终端
python test/real_service_smoke.py        # 真实 embedding/llm 进程内冒烟
python scripts/verify_three_tier.py      # 对 8902 服务验证三层记忆检索
```

### 6.3 数据自检清单（每跑一批数据后）

1. `users/{user_id}/memory/` 下节点是否按 `type/subtype/hash` 归位。
2. 每个节点三件套齐全：`memory.json` / `content.md` / `history.jsonl`。
3. `index/memory.db` 能否从文件全量重建（`rebuild_index()`），重建后检索结果不变。
4. 检索 `explain` 里 `sim/heat/final_score` 是否合理（temporary 节点分数应明显低）。
5. 日志（JSON lines）里有没有 `extract_discard` / `event_ignored_temporary` / `task_dead`——这些是质量告警。
6. 变体污染回归：构造"老规矩 A → 临时变体 → 老规矩 B"三连，B 轮检索仍是长期值。

### 6.4 评测对齐（如果要做数据集评测）

原系统的评测关注 3 个指标：**历史相关问题正确率**、**澄清轮数与耗时**、**检索与推理 token 消耗**（`docs/design.md` §9）。
复刻后建议在同一数据集上对比这 3 项，重点看变体污染修复前后的正确率差异（原系统预期转正 ≥10 个用例）。

---

## 7. 复刻时可以砍掉的部分（按性价比）

| 能力                   | 砍掉代价 | 备注                                                    |
| ---------------------- | -------- | ------------------------------------------------------- |
| 图（graph.jsonl）      | 低       | 检索默认 `follow_edges=0`，图目前更多是溯源/扩展        |
| LLM 路由 Router        | 低       | 可以先全部走 `memory_search`，别名短路保留              |
| Timeline 压缩          | 低       | 维护里最不关键的一项                                    |
| 信息熵剪枝 Pruner      | 低       | `pruning.enabled` 可关，默认兜底是规则剪枝              |
| Fast API 服务          | 低       | 进程内直连完全等价                                      |
| 队列后台 worker        | 中       | 同步内联模式功能等价，只是阻塞调用线程                  |
| 三层记忆的 ingest 幂等 | 中       | 靠 `record_id`/`experience_id` 做 key，丢了会重复建节点 |

**不能砍的**：memory_key 归一、Updater 状态机、变体污染防线（§4 坑 2）、LLM 抽取后的确定性校验、namespace 隔离。

---

## 8. 关键文件速查

| 文件                                                         | 作用                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| `src/memory_system/api.py`                                   | 门面，所有公开入口 + 三层记忆 ingest                   |
| `src/memory_system/core/models.py`                           | MemoryNode/Event/Stats/状态/事件类型/边类型            |
| `src/memory_system/core/registry.py` + `config/registry/*.yaml` | 15 种记忆类型定义                                      |
| `src/memory_system/pipeline/updater.py`                      | 状态机 + 变体污染防线 + 热度                           |
| `src/memory_system/pipeline/extractor.py`                    | LLM 抽取 + 时段回填 + validity 归一 + 校验             |
| `src/memory_system/pipeline/fastpath.py`                     | 免 LLM 规则                                            |
| `src/memory_system/pipeline/induction.py`                    | 跨会话归纳                                             |
| `src/memory_system/retrieval/search.py`                      | 混合打分 + 艾宾浩斯 λ + 热度 + 临时降权                |
| `src/memory_system/storage/index.py`                         | SQLite 向量 + FTS5（含外部表同步坑）                   |
| `src/memory_system/storage/filestore.py`                     | 文件真相 + 原子写                                      |
| `src/memory_system/providers/llm.py`                         | OpenAI 兼容 LLM + 健壮 JSON 解析                       |
| `prompts/extract_generic.md`                                 | LLM 抽取提示词（10 种 subtype 的抽取规则）             |
| `fix_notes/variant_pollution.md`                             | 变体污染完整复盘（强烈建议先读）                       |
| `data_roots_8902_v7/`                                        | 三层记忆真实落盘样例（experience/working_record 节点） |
| `tests/`                                                     | 91 个离线单测（复刻的验收基准）                        |

---

## 9. 一句话总结给复刻者

**先做对 memory_key 和状态机，再谈 LLM；LLM 输出的每一分可信度都要用确定性代码换回来（时段回填、validity 归一、schema 校验）；变体污染防线是评测正确率的生命线；隔离键是 user_id 不是 tenant_id；最后用 `pytest tests/` 91 例 + 变体三连回归验收。**

