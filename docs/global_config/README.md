# 全局规则与持久记忆归档 (`~/.gemini/config`)

本目录存放跨项目自动生效的全局行为准则（`GEMINI.md`）与分层专题记忆（`memory/*.md`），方便在不同机器（如家里电脑与办公电脑）之间一键同步。

## 目录结构

- [`GEMINI.md`](./GEMINI.md)：全局常驻核心规则（沟通语态、术语规范、HTML 与 Lavish 协同、代码红线、文献检索与记忆索引）
- [`memory/MEMORY.md`](./memory/MEMORY.md)：全局持久记忆索引
- [`memory/feedback_html_artifact_style.md`](./memory/feedback_html_artifact_style.md)：主从式多页 HTML、平实总结语态、三列参考论文表、零未渲染公式与 Lavish 工作流规范
- [`memory/feedback_zero_hardcoding.md`](./memory/feedback_zero_hardcoding.md)：零硬编码与优先修复生成器源码规范
- [`memory/feedback_logging_and_code_modularity.md`](./memory/feedback_logging_and_code_modularity.md)：标准 `logging` 日志、函数粒度控制与 Prompt 模板解耦规范
- [`memory/reference_repos_and_infra.md`](./memory/reference_repos_and_infra.md)：本地仓库映射、Python 环境与模型基础设施索引

## 回家电脑一键恢复命令（PowerShell）

拉取仓库后，在仓库根目录执行以下命令即可同步到本机 `~/.gemini/config/`：

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.gemini\config\memory" | Out-Null
Copy-Item -Path ".\global_config\GEMINI.md" -Destination "$HOME\.gemini\config\GEMINI.md" -Force
Copy-Item -Path ".\global_config\memory\*.md" -Destination "$HOME\.gemini\config\memory\" -Force
```
