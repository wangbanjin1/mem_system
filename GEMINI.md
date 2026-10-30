# Project Memory & Architecture Guide (`memory_system`)

本文件为 `D:\coding\memory_system`（远端仓库：`https://github.com/wangbanjin1/mem_system.git`，主分支：`main`）的项目级记忆与开发规范。

---

## 1. 项目定位与环境配置

- **核心职责**：面向通信网络保障场景的多层级认知记忆系统，涵盖**事实记忆（Fact Memory）**、**工作记忆（Working Memory）**与**经验记忆（Experience Memory）**的快慢双通道萃取、正交解耦落盘、向量化与结构化混合检索。
- **Python 解释器**：`C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`
- **模型配置**：
  - 记忆萃取与归纳 LLM：`qwen3.8-flash` / `qwen3.7-plus`（百炼 OpenAI 兼容接口）
  - 向量编码模型：`text-embedding-v4`（`1024` 维）
- **关联上下游仓库**：
  - 上游合成数据集：`D:\coding\cross_memory_data`（`data/cognitive_samples`）
  - 下游意图理解与端到端评测：`D:\coding\qwen_intent_rag`（通过 `memory_bridge.py` 桥接调用本仓 API）

---

## 2. 核心架构设计

### 2.1 事实记忆：方案 A 五层正交解耦架构（严禁跨层冗余耦合）
用户偏好与事实记忆通过 `_decompose_to_orthogonal_events` 拆解为正交子类型落盘至 SQLite `vector` 表，各层单一职责：
1. **`service_to_app`**（主键：`business`）：业务到默认应用的双向路由指针（如 `openlive` / `live` ↔ `douyin`），严禁存画质、时延、时间窗或场景。
2. **`periodic_preference`**（主键：`condition | value`，即 `time_window | app:biz`）：纯周期时间窗路由指针（如 `12:30-13:30` -> `douyin:openlive`），**严禁存 `resolution`、`latency`、`scene`**。
3. **`preferred_value`**（主键：`application | business`）：通用默认 QoS 参数单一事实源（`resolution` + `latency`），**严禁存 `time_window`、`scene`**。
4. **`scene_preference`**（主键：`scene | application | business`）：特定物理/移动场景（如训练场、大巴转场）下的差异化 QoS 参数覆写，**严禁存 `time_window`**。
5. **`expression`**（主键：`alias`）：个性化表述映射层（将用户自定义口语表述映射至标准槽位值 `target`，带轻量 `app/biz/scene` 作用域约束），严禁存 `time_window` 或父节点指针。
6. **`environment`**（主键：`environment | application | business`）：弱网/高铁等无线环境劣化规则。

> **业务术语规范**：开直播统一使用 `openlive`，看直播统一使用 `live`。

### 2.2 工作记忆（Working Memory）
- **职责**：处理跨会话历史保障工单指代（如“按上次保障”、“照上周在某训练场办的那笔”）。
- **检索机制**：
  - 时间指代词基于当前会话 `reference_time` 解析为相对日期窗口或序数条件；
  - 地点/场馆或业务描述采用**短窗口滑动切词 + 向量余弦相似度匹配**（匹配历史工单中的 `location_Context` 等向量列），结合结构化条件执行 `SELECT`；
  - 若同一时间窗/条件下命中多笔工单（结果不唯一），自然触发消歧分支，待下一轮用户补充特征（如画质、最早/最近）后二次筛选闭环。

### 2.3 经验记忆（Experience Memory）
- **职责**：从历史多轮交互中归纳用户的隐含习惯与动态决策规则（如主动提前预热时间 `prewarm_buffer_minutes`、工单时长动态消歧策略等）。
- **双路径机制**：
  - **快路径（Fast Path）**：当用户首轮输入信息完备且命中已归纳的高置信度经验卡片时，直接应用经验规则（如自动前置预热时间）一步闭环。
  - **慢路径（Slow Path / 消歧澄清）**：当存在多个候选经验或关键消歧条件缺失时，返回候选集引导多轮澄清，严禁盲目覆盖用户明说参数。

### 2.4 混合检索引擎与向量路由 (`retrieval/search.py` & `storage/index.py`)
- **反对死板纯规则切词**：采用滑动短窗口向量化与 SQLite Vector 表中目标字段向量（`embedding_json`）计算余弦相似度，定位向量命中记录（如 `expression`、`scene_preference`、`periodic_preference`）。
- **多阶段关联槽位 `SELECT` 拼装**：由向量命中记录读取同行结构化槽位列（`app_col`、`biz_col`、`target_col`、`scene_col`），经二阶段关联表补全后，通过 `select_by_slots` 精确查询配对的 `preferred_value` / `periodic_preference` 节点，避免召回无关噪声记忆。

---

## 3. 项目工程红线与历史踩坑记录 (Lessons Learned)

1. **严禁样本级硬编码**：
   - `experience_rules.py`、`working_rules.py`、`fastpath.py` 及 `search.py` 中严禁写死特定样本的时间窗（如 `15:00-17:00`）、固定时长（如硬编码 `120min`）或特定场馆字面量。所有参数必须从记忆卡片、工单记录或用户原话中动态提取。
2. **保持正交边界**：
   - 修改 `extractor.py` 或 `updater.py` 时，必须确保 5 层事实记忆字段严格正交，不得将 `resolution`/`latency` 混入 `periodic_preference`。

---

## 4. 常用测试与验证命令

```powershell
# 运行全量单元测试
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m pytest D:\coding\memory_system\tests -q

# 运行认知记忆专项测试
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m unittest tests/test_cognitive_memory.py
```
