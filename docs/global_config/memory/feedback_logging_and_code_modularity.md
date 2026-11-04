---
name: Logging System, Function Granularity & Prompt Decoupling Policy
description: Enforce standard Python logging over print(), strict function/module size boundaries, and prompt template decoupling across all repositories
type: feedback
---

### 核心规则（Rule Statement）

1. **全量使用标准日志系统（Zero `print` for Logging）**：业务代码、数据生成管线、检索引擎与评测脚本中**严禁使用 `print()` 输出运行日志或调试信息**，必须统一使用 Python 标准库 `logging`（`logger = logging.getLogger(__name__)`，通过 `logger.info()` / `logger.warning()` / `logger.error()` / `logger.debug()` 分级输出）。
2. **函数与模块单一职责拆分（Function & Module Granularity）**：严禁编写超过 `80~100` 行的长函数或超过 `500` 行的巨型模块；多分支逻辑优先采用字典分发表（Dispatch Table）或独立阶段子函数拆解。
3. **Prompt 常量与静态模板物理隔离（Prompt & Template Decoupling）**：严禁在核心算法或流水线逻辑函数中硬塞大段多行 System Prompt、Few-Shot 示例或 HTML/CSS 模板字符串，必须抽离至独立的 `prompts/` 或 `templates/` 模块中集中维护。

**Why:**
- `print()` 无法控制日志级别、时间戳、模块来源及输出流重定向，在多进程批量生成或评测时会污染标准输出且难以排查问题；
- 动辄数百行的巨型函数与内联大段 Prompt 字符串严重损害代码可读性、可测试性与职责边界，修改提示词极易误触核心控制流逻辑。

**How to apply:**
- 在每个模块顶部声明 `logger = logging.getLogger(__name__)`，仅在 CLI 入口（如 `main.py` 或脚本 `__main__` 块）调用 `logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")` 配置 Handler。
- 在各仓库 `pyproject.toml` 的 `[tool.ruff.lint]` 中启用 `"T20"`（`flake8-print`）与 `"BLE"` / `"S110"`（禁止静默吞没异常）。
- 新增或修改 LLM 调用逻辑时，将 Prompt 模板统一放入 `prompts/` 子包并按语义命名导出。
