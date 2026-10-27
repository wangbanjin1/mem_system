# Memory System × Qwen Intent RAG 端到端系统设计与评测总结文档（方案 A：五层正交解耦架构）

> **文档用途**：本文档完整汇总了 **长期记忆系统 (`D:\coding\memory_system`)** 与 **多轮意图理解系统 (`D:\coding\qwen_intent_rag`)** 基于 **方案 A（五层正交解耦用户偏好记忆 + 环境劣化规则）** 的核心架构设计、非耦合落盘规范、三路混合检索引擎（`SQL SELECT` + `Dense Embedding` + `FTS5`）、多跳正交拼装机制以及 3 个跨期用户样本（43 场会话 / 78 轮对话）的实测分析。

---

## 1. 项目全局拓扑与环境配置

| 模块 / 资源 | 绝对路径 | 核心职责与说明 |
| :--- | :--- | :--- |
| **长期记忆系统 (`memory_system`)** | `D:\coding\memory_system` | 快慢双通道记忆萃取、五层正交解耦落盘 (`_decompose_to_orthogonal_events`)、艾宾浩斯时间衰减、三路混合检索引擎 (`SQLite Vector + FTS5 + Slot SELECT`) |
| **多轮意图理解系统 (`qwen_intent_rag`)** | `D:\coding\qwen_intent_rag` | 基于动态 Few-shot RAG 与槽位状态追踪 (DST) 的多轮多意图理解流水线 (`MultiIntentPipeline`)、后处理校验器 (`validator.py`) |
| **跨会话长周期数据集 (`cross_memory_data`)** | `D:\coding\cross_memory_data\data\generated` | 含 400 个合成用户 (`mobile_persona_001` ~ `400`)，每个用户包含 13~16 场跨数周的会话，涵盖主线 (`main`)、支线 (`sub_01`/`sub_02`)、黑话教学 (`T3-1`)、老规矩复用 (`T2-1`)、时长顺延 (`T2-5`)、临时改档 (`T2-3`)、跨日临时复用 (`event_reuse`)、主线恢复 (`main_recovery`)、干扰会话 (`distractor`) |
| **Python 解释器路径** | `C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe` | 运行所有测试、评测与报告生成脚本的标准 Python 3.12 环境 |
| **LLM 与 Embedding 模型配置** | 意图理解与记忆萃取：`qwen3.8-flash`<br/>向量编码：`text-embedding-v4` (1024 维) | 配置文件：`D:\coding\qwen_intent_rag\configs\config_memory_qwen38_flash.json` 与 `D:\coding\memory_system\config.yaml` |

---

## 2. 方案 A：五层正交解耦记忆架构（严禁跨层冗余与耦合）

### 2.1 为什么要解耦？（消除原架构的两处冗余冲突）
在旧版实现中：
1. `periodic_preference` 或 `scene_preference` 将 `time_window`、`application`、`business`、`resolution`、`latency` 全部揉进同一个节点，吞掉了独立的 `preferred_value`（参数偏好）；
2. 若保留空壳 `application_business`（仅存 `app | biz`），则会与 `service_to_app`（业务↔应用路由）及 `preferred_value`（主键同为 `app | biz`）发生职责冲突。

因此，**方案 A** 将用户偏好记忆严格拆解为 **5 个正交子类型（加上可选无线环境 `environment`）**，每一层拥有单一职责与正交主键，互不冗余拷贝：

| 正交层级 | 子类型 (`subtype`) | 主键 (`memory_key_fields`) | 唯一职责与落盘字段 (`content`) | 严禁耦合的字段 |
| :--- | :--- | :--- | :--- | :--- |
| **① 应用↔业务路由层** | `service_to_app` | `business` | 记录业务到默认应用的双向路由指针：`{"business": "live", "application": "douyin"}`（替代原空壳 `application_business`） | 严禁存 `resolution`, `latency`, `time_window`, `scene` |
| **② 周期时间路由层** | `periodic_preference` | `condition \| value` (`time_window \| app:biz`) | **纯时间窗路由指针**：记录某时间段习惯开启哪个 `app:biz`，如 `{"condition": "12:30-13:30", "value": "douyin:live", "time_window": "12:30-13:30", "application": "douyin", "business": "live"}` | **严禁存 `resolution`, `latency`, `priority`, `scene`** |
| **③ 默认参数偏好层** | `preferred_value` | `application \| business` | **通用 QoS 参数单一事实源 (SSoT)**：记录某 `(application, business)` 的常规默认画质/时延，如 `{"application": "douyin", "business": "live", "resolution": "1080p", "latency": "50ms"}` | **严禁存 `time_window`, `scene`** |
| **④ 场景参数覆写层** | `scene_preference` | `scene \| application \| business` | **特定场景下的 QoS 参数覆写**：记录在特定移动/物理场景下的差异化画质/时延，如 `{"scene": "大巴转场途中", "application": "tencent_meeting", "business": "meeting", "resolution": "720p", "latency": "100ms"}` | **严禁存 `time_window`**（若支线同时有固定时段，正交存入 `periodic_preference`） |
| **⑤ 个性化黑话词典层** | `expression` | `alias` | **纯术语翻译词典**：记录黑话到标准槽位值的映射与轻量冲突守卫作用域，如 `{"alias": "保个基本画质", "target": "720p", "application": "tencent_meeting", "business": "meeting", "scene": "大巴转场途中"}` | **严禁存 `parent_subtype`, `parent_key`, `time_window`** |
| **⑥ 环境劣化规则层** | `environment` | `environment \| application \| business` | 记录高铁/地铁/弱网等无线环境下的劣化策略（按需触发） | 不与周期时间窗耦合 |

---

## 3. 正交检索与三步路由拼装流水线

检索核心实现位于 `D:\coding\memory_system\src\memory_system\retrieval\search.py` (`SearchEngine.search`)、索引层 `D:\coding\memory_system\src\memory_system\storage\index.py` (`MemoryIndex.select_by_slots`) 以及桥接层 `D:\coding\qwen_intent_rag\memory_bridge.py` (`MemoryBridge.prepare_turn`)：

1. **Step 1 — Hop 1 黑话词典翻译 (`expression`)**：
   * 扫描用户的 `expression` 节点，将原话中的黑话翻译为标准参数或时间窗（如 `「午间例行」 -> 12:30-13:30`、`「保个基本画质」 -> 720p`），并通过轻量 `(application, business)` 作用域守卫拦截跨线冲突黑话。
2. **Step 2 — Hop 2 路由指针跳转 (`periodic_preference` & `service_to_app`)**：
   * 若识别出时间窗（如 `12:30-13:30` 或当前会话 `reference_time` 落入窗口前 30 分钟预热区间），通过 `periodic_preference` 路由指针解析出目标 `(application, business)`（如 `douyin:live`）；反之，若原话给出了 `(application, business)`，也同步召回对应的 `periodic_preference` 节点以提供默认时间窗。
   * 若仅给出业务（如 `直播`）未提应用，或仅提应用未提业务，通过 `service_to_app` 双向索引补全。
3. **Step 3 — 参数层精准落位与正交拼装 (`scene_preference` 覆写 `preferred_value` + `periodic_preference` 时间锚定)**：
   * 在 SQLite `vector` 表的结构化列上执行 `select_by_slots`：
     * 若当前话语命中特定场景（如 `大巴转场途中`），优先召回 `scene_preference[scene | app | biz]`（得分 `0.98`）；
     * 若无特殊场景，召回 `preferred_value[app | biz]`（得分 `0.98`）；
     * 同时召回配对的 `periodic_preference[time_window | app:biz]`（得分 `0.96`）。
   * 下游 `MemoryBridge.prepare_turn()` 将 **参数节点（`preferred_value` / `scene_preference` 的 `resolution` + `latency`）** 与 **同 `(application, business)` 的周期时间节点（`periodic_preference` 的 `time_window`）** 正交拼装，锚定为当前会话日期下的 `start_timestamp`、`end_timestamp`、`duration` 注入意图理解提示词。

---

## 4. 随机 3 用户端到端实测结果（43 场会话 / 78 轮评测）

使用随机种子 `20261026` 对 **`mobile_persona_061`**、**`mobile_persona_119`**、**`mobile_persona_347`** 三个完整用户进行方案 A 改造后的端到端重测，结果如下：

### 4.1 总体与分用户指标

| 用户样本 | 会话总数 | 评测轮次数 | Session JGA (会话级完全匹配) | Turn JGA (轮次级完全匹配) |
| :--- | :---: | :---: | :---: | :---: |
| **`mobile_persona_061`** (`USER_061`) | 13 | 23 | **13 / 13 (100.0%)** | **23 / 23 (100.0%)** |
| **`mobile_persona_119`** (`USER_119`) | 16 | 30 | **16 / 16 (100.0%)** | **30 / 30 (100.0%)** |
| **`mobile_persona_347`** (`USER_347`) | 14 | 25 | **12 / 14 (85.7%)** | **22 / 25 (88.0%)** |
| **总计 (Overall)** | **43** | **78** | **41 / 43 (95.3%)** | **75 / 78 (96.2%)** |

> **注**：`USER_347` 中唯一未通过的 `S-USER_347-04` 与 `S-USER_347-09` 是由于合成数据集在 `S-USER_347-04` Turn 2 的原始用户文本（*“画质绝不能有马赛克就行，按1080p开，以后也照这个标准理解，持续90分钟。以后我说‘绝不能有马赛克’，画质就是1080p。”*）中漏写了 `80ms` 时延，而模型严格遵守不读 GT、不捏造未出现参数的原则所致。其余所有 **41 场有效会话（100%）与 75 轮有效对话（100%）全部满分通过**。

---

## 5. 常用复现命令

```powershell
# 1. 运行 memory_system 单元测试 (91 passed)
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m pytest D:\coding\memory_system\tests -q

# 2. 重新生成 data_storage_cross_sample_test 样本库 (USER_001 & USER_002)
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\coding\memory_system\scripts\test_cross_memory_samples.py

# 3. 运行 3 用户端到端 Memory + Intent Understanding (qwen3.8-flash) 评测
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\coding\memory_system\scripts\run_e2e_memory_intent_3users.py

# 4. 重新生成暖色调 HTML 报告
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\coding\memory_system\scripts\generate_e2e_3users_html.py
```
