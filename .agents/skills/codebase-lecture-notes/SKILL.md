---
name: codebase-lecture-notes
description: >-
  Generate publication-grade Chinese codebase and operator deep-dive lecture notes
  following the wdkns-skills pedagogical style and tensor-formula-viz native-LaTeX
  vector diagram specification. Use when the user asks to summarize, analyze, or
  write lecture notes for a code repository, PR, commit, system architecture, or
  GPU/AI operator (e.g., vLLM, 1Cat-vLLM, CUDA/Triton kernels, PyTorch modules).
---

# Codebase Lecture Notes (`wdkns-skills` & `tensor-formula-viz` Style)

Turn a codebase module, pull request, or system/operator implementation into a structured, figure-rich, publication-grade Chinese lecture note (`.md` + `.html` / PDF) that reads like a master teacher guiding the reader from first principles to exact source lines.

---

## Core Architecture: The Five Pillars

### 1. Front-Page Titlepage & `tcolorbox` Metadata Block
Every lecture note MUST start with a YAML frontmatter block that `build_notes.py` renders into a LaTeX-style `titlepage` header and a sharp-cornered metadata `tcolorbox`:

```markdown
---
series: <系列名称，如：1Cat-vLLM 源码精读中文讲义 · 核心系统与硬件架构实战>
title: <主标题，如：Lecture 01: vLLM 架构全景与 Volta (SM70) 硬件底座>
subtitle: <副标题：一句话概括从问题痛点到核心机制的演进链路>
authors: <作者与整理者>
date: <YYYY-MM-DD>
repo: <目标仓库链接或分支/PR>
hardware: <目标硬件架构及核心物理参数，如算力峰值、显存带宽、寄存器/Smem限制>
source_files: <本讲涉及的核心源码文件路径列表>
topics: <本讲串联的 3~5 个核心问题关键词>
---
```

### 2. Five-Step Pedagogical Narrative (五步教学法)
Do NOT dump code files in directory order or write a shallow API reference. Every major section (`## 1 ...`, `## 2 ...`) MUST follow this five-step teaching progression:

1. **Motivation（问题与痛点）**: State the concrete physical bottleneck, memory wall, or failure mode first. Explain why naive or baseline designs fail.
2. **Main Idea（核心直觉）**: Give plain-language intuition and physical analogies *before* introducing formal equations or code structs.
3. **Formal Mechanism（数学与机制推导）**: Present the exact mathematical formulation, address translation, or thread/tensor layout transformation.
4. **Code & Evidence（真实源码印证与对账）**: Ground every claim in real source code snippets from the repository, with exact file paths and line numbers, plus concrete numerical calculations (e.g., byte counts, occupancy math, TFLOP/s comparisons).
5. **Takeaway（章节收束）**:
   - End **every** major section with `### X.Y 本章小结` (distilling the section into 1–2 first-principle sentences).
   - End the **entire document** with a top-level `## 总结与延伸` section containing:
     - `### X.1 核心脉络回看` (causal chain connecting all sections)
     - `### X.2 开放问题与下一步探索`
     - `### X.3 拓展阅读`

### 3. `tcolorbox` Four-Signal Callout System (动态标题高亮路标)
Use callout boxes deliberately and repeatedly after dense derivations, code blocks, or mechanism contrasts. **Every callout box MUST have a specific, dynamic pedagogical title** written on the same line after the tag (never use generic titles like "注意" or "提示"):

- 🟨 **`importantbox` (`> [!IMPORTANT] <动态归纳标题>`)**:
  For core concepts the reader must walk away with: first principles, formal invariants, key mechanism summaries, and high-level conclusions.
- 🟦 **`knowledgebox` (`> [!KNOWLEDGE] <动态归纳标题>`)**:
  For background and side knowledge that deepens intuition without derailing the main thread: hardware architecture background, framework trade-offs, algebraic cancellation insights, and analogies. (Rendered with the signature top-left floating blue title tab).
- 🟥 **`warningbox` (`> [!WARNING] <动态归纳标题>`)**:
  For common misunderstandings, performance traps, and subtle implementation bugs: e.g., Local Memory Spilling to HBM, uncoalesced memory access, shared memory bank conflicts, or broken matrix associativity.
- 🟩 **`dialoguebox` (`> [!DIALOGUE] <动态归纳标题>`)**:
  For high-signal PR review exchanges, commit message insights, or sharp Q&A contrasts.

**Strict Callout Rules**:
- Never place figures (`![...](...)`) inside callout boxes.
- Routine exposition stays in normal prose; boxes are for high-signal takeaways.

### 4. Formula & Code "Sandwich" Rules + Exact Source Provenance

#### Mathematical Formulas (公式三明治铁律)
Whenever a display formula appears:
1. **Before**: Explain in plain Chinese what the formula expresses and why it is needed here.
2. **Middle**: Show clean display math using `$$...$$` (do NOT clutter display equations with long `\underbrace` shape annotations).
3. **After**: Immediately follow with a flat numbered list starting with `符号解释如下：` that defines every symbol's physical and code meaning:
   ```markdown
   $$\text{Performance} = \min\left(P_{\max},\; I \cdot B_{\text{HBM}}\right)$$

   符号解释如下：
   1. $P_{\max}$：硬件的理论峰值浮点计算吞吐。
   2. $B_{\text{HBM}}$：硬件的全局显存可用带宽。
   3. $I$：算术强度（Arithmetic Intensity），即每搬运 1 字节数据所能支撑的浮点运算次数。
   ```

#### Code Listings (代码三明治与源码行号溯源铁律)
Whenever a code snippet appears:
1. **Before**: Explain what role this struct/kernel/function plays in the overall execution path.
2. **Middle**: Use the extended fence header with `caption=` and `source=` (exact file path and line range):
   ````markdown
   ```cuda | caption=PagedAttention 二级跳表与线程内向量化列偏移计算 | source=csrc/attention/sm70_v37/tail.cu:L339--L351
   // Code here
   ```
   ````
3. **After**: Summarize the execution path, memory access pattern, or hardware behavior in 1–2 sentences (focus on *how data moves and why it is fast*, not syntax).

### 5. `tensor-formula-viz` Three-Zone Native-LaTeX Vector SVG Diagrams
Do NOT use crude ASCII art boxes (`┌───┐`) for core tensor flows, memory layouts, or hardware pipelines. Instead, generate publication-grade vector SVG figures in `assets/` via `generate_figures.py` and reference them as:

```markdown
![图注标题|源码定位：csrc/attention/sm70_v37/tail.cu:L28--L128](assets/fig_name.svg)
```

#### Mandatory Three-Zone Layout for Every Figure
1. **Top Zone — Formula**:
   1–2 compact lines of primary mathematical definitions in `$\displaystyle ...$` rendered via `tex_box()`, separated from the middle zone by a thin horizontal rule.
2. **Middle Zone — Computation & Hardware Mapping**:
   - Use `draw_matrix_grid()` with the muted academic paper palette:
     - Teal (`#4F8FA5` / `#7FB3C5` / `#B9D6E0`)
     - Orange (`#EE995B` / `#F3B787` / `#F8D6BA`)
     - Coral (`#C95B5B` / `#DA8E8E` / `#EAC0C0`)
     - Violet (`#8A74B5` / `#AE9ECF` / `#D1C7E3`)
     - Green (`#5B9A68` / `#89C195` / `#BCE0C4`)
     - Neutral Gray (`#85898F` / `#B1B5BA` / `#DCDFE3`)
   - Encode structural support faithfully: leave masked/zero entries white (`pattern="causal"` or `"diag"`), use row/column stripes (`"stripe_row"` / `"stripe_col"`) for thread/warp partitioning or GQA packing.
   - Reserve dedicated horizontal lanes (`y_grp`, `y_sub`, `y_mat`, `y_sym`, `y_shp`) so group headers, sub-block labels, matrix brackets, bottom symbols (`$\mathbf{Q}_{\text{tile}}$`), and bottom shapes (`$(B_r \times d)$`) share aligned baselines.
   - Include concise side `note-card` callouts for key engineering pitfalls or hardware numbers.
3. **Bottom Zone — Meaning Ledger (`轴 / 对象 / 机制`)**:
   - A full-width low-contrast `.ledger-box` with three fixed rows:
     - **轴**: Dimension symbols and concrete sizes in the code.
     - **对象**: Tensor roles, dtypes, and register/shared-memory residency.
     - **机制**: Numbered `(i)`, `(ii)`, `(iii)` explanations of stride tricks, bank-conflict avoidance, algebraic cancellations, or memory aliasing.
   - A centered muted signature line at the very bottom: `<图表主题> @ <讲义系列名>`.

#### Mandatory Native-LaTeX & Zero-Collision Audit Rules
- **All formulas, tensor symbols, and dimension annotations inside SVGs MUST use `tex_box()` (`<foreignObject>` with `class="svg-latex"` and `$ ... $` LaTeX)** so that when `build_notes.py` inlines the `.svg` into `.html`, MathJax typesets every formula in authentic Computer Modern math font.
- **Avoid raw `<` or `>` inside LaTeX strings in XML**: use `\langle ... \rangle`, `\lt`, `\gt`, or `&lt;` / `&gt;` so XML parsing succeeds.
- **Run `audit_svg_text_collisions()` on every SVG before building**:
  - Verify `0` `Label-Label` overlaps, `0` `Label-Tile` overlaps, and `0` `Label-Arrow` overlaps.
  - If any collision is reported, adjust lane coordinates or box widths until `[AUDIT PASS]` is printed for every figure.

---

## Standard Execution Workflow

When asked to write or refactor a lecture note for a codebase module or PR:

1. **Inspect Source Code First**:
   Read the actual source files (`.cu`, `.cuh`, `.cpp`, `.py`) to collect exact struct names, template constants, `static_assert` budgets, line numbers, and implementation comments.
2. **Design & Generate Vector Figures (`generate_figures.py`)**:
   - Add or update `generate_figXX()` functions in `generate_figures.py` using `draw_matrix_grid()` and `tex_box()`.
   - Run `python generate_figures.py` and confirm every SVG passes `[AUDIT PASS]` with 0 collisions.
3. **Write the Lecture Note (`markdown/<name>.md`)**:
   - Include the YAML frontmatter block, 5-step pedagogical sections, Formula/Code sandwiches with exact line provenance, dynamic-titled `tcolorbox` callouts, and `![caption|provenance](assets/...)` figure links.
4. **Compile & Verify HTML Output (`build_notes.py`)**:
   - Run `python build_notes.py` to inline the LaTeX SVGs and update `html/<name>.html` and `index.html`.
5. **(Optional for Large Modules) Reviewer Pass**:
   - Cross-check the generated note against the raw source files to verify no critical optimization trick, hardware constraint, or edge condition was omitted.
