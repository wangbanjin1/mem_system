---
name: feedback-zero-hardcoding-and-generator-first
description: Strict prohibition against keyword/slot hardcoding in retrieval or data generation; require fixing source generators rather than patching output artifacts
type: feedback
---

严禁在检索算法、状态跟踪或合成数据生成器中硬编码业务字面量、固定时间窗（如 `15:00-17:00`）或固定时长（如 `120min`）；发现评测或样本异常时，必须修复底层生成器或核心引擎源码。

**Why:** 写死特定场馆、固定时长或脆弱正则分支只能掩盖单次测试集错误，会导致跨场景泛化失效及后续重新生成数据时问题复发；同时合成数据若缺乏多分布随机性，训练或评测出的记忆系统无法适应真实多变的用户表达。

**How to apply:** 
1. 算法侧检索与消歧统一采用“滑动窗口短词向量匹配 + 结构化槽位 `SELECT` + 动态偏移计算”，发现任何针对测试样本的字面量 `if-else` 分支时立即通篇排查清理；
2. 数据侧修改 `cross_memory_data` 时，先改 `src/` 下的生成器与 `qc_validator.py`（确保 `end - start == duration` 与多分布覆盖），先跑 1~4 个代表性用户样本验证通过后再批量重跑。
