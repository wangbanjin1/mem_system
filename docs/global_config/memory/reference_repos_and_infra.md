---
name: reference-repos-and-dashscope-infra
description: Local workspace paths, GitHub repositories under wangbanjin1, Python interpreter path, and DashScope LLM/Embedding model endpoints
type: reference
---

本地核心仓库映射、运行环境与外部模型服务指针：

- **本地工作区与远端 GitHub 仓库（组织：`wangbanjin1`）**：
  - `D:\coding\memory_system` $\leftrightarrow$ `https://github.com/wangbanjin1/mem_system.git`（三层认知记忆引擎，基线 Tag：`v1.0.0-three-tier-baseline`）
  - `D:\coding\cross_memory_data` $\leftrightarrow$ `https://github.com/wangbanjin1/cross_data_gen.git`（跨会话多轮合成数据集生成器，基线 Tag：`v2.0.0-three-tier-baseline`）
  - `D:\coding\qwen_intent_rag` $\leftrightarrow$ `https://github.com/wangbanjin1/intent_rag.git`（多轮多意图 DST 与记忆桥接端到端评测）
  - `D:\coding\sharing` $\leftrightarrow$ `https://github.com/wangbanjin1/sharing.git`（技术分享 HTML Slides 与专题文档归档）
  - `D:\coding\modern_genai_bilibili`（Modern Agents / Agentic RL / LLMs 原理与源码剖析笔记库）
- **标准 Python 解释器路径**：
  - `C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`
- **阿里云百炼（DashScope OpenAI 兼容接口）主力模型**：
  - 文本推理/渲染/萃取：`qwen3.7-plus`、`qwen-plus`、`qwen3.8-flash`、`deepseek-v4.1-flash`
  - 向量编码：`text-embedding-v4` / `qwen3.7-text-embedding`（默认 `1024` 维 `float32`）
- **学术文献实时检索工具脚本**：
  - arXiv 检索与源码下载：`C:\Users\Administrator\.gemini\config\skills\literature_search_arxiv\scripts\search_arxiv.py` & `download_paper.py`
  - OpenAlex 高引文献检索：`C:\Users\Administrator\.gemini\config\skills\literature_search_openalex\scripts\openalex_cli.py`
