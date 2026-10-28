#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Reusable helper template for codebase-lecture-notes (tensor-formula-viz style SVG + Native LaTeX)
Provides:
- COMMON_DEFS: Standard SVG stylesheet & arrow markers
- tex_box(): Native LaTeX (<foreignObject> + MathJax) box helper
- draw_matrix_grid(): Kimi/wdkns-style matrix block tiles with 3-level lightness & brackets
- audit_svg_text_collisions(): Automated bounding-box collision auditor (Label-Label, Label-Tile, Label-Arrow)
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COMMON_DEFS = """
  <defs>
    <style>
      .fig-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; }
      .stage-title { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
      .op-node { fill: #f8fafc; stroke: #64748b; stroke-width: 1.2; }
      .op-text { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11.5px; font-weight: 700; fill: #1e293b; text-anchor: middle; }
      .note-card { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; rx: 3; }
      .note-title { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11.5px; font-weight: 700; fill: #0f172a; }
      .ledger-box { fill: #f8fafc; stroke: #94a3b8; stroke-width: 1; rx: 2; }
      .ledger-key { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 12px; font-weight: 700; fill: #0f172a; }
      .sig-line { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11px; fill: #94a3b8; text-anchor: middle; }
      .bracket { fill: none; stroke: #475569; stroke-width: 1.3; stroke-linecap: square; }
      .arrow { stroke: #64748b; stroke-width: 1.3; fill: none; marker-end: url(#arrowhead); }
      .arrow-accent { stroke: #c95b5b; stroke-width: 1.4; fill: none; marker-end: url(#arrowhead-red); }
      .arrow-teal { stroke: #4f8fa5; stroke-width: 1.4; fill: none; marker-end: url(#arrowhead-teal); }
    </style>
    <marker id="arrowhead" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#64748b" />
    </marker>
    <marker id="arrowhead-red" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#c95b5b" />
    </marker>
    <marker id="arrowhead-teal" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#4f8fa5" />
    </marker>
  </defs>
"""


def tex_box(cx, y_top, w, h, content, align="center", size=12.5, color="#0f172a", weight="normal", bg="none", wrap=False):
    x = (cx - w / 2.0) if align == "center" else cx
    justify = {"center": "center", "left": "flex-start", "right": "flex-end"}[align]
    ws = "normal" if wrap else "nowrap"
    t_align = "center" if align == "center" else "left"
    bg_css = f"background:{bg};border-radius:2px;padding:0 4px;" if bg != "none" else ""
    return (
        f'<foreignObject x="{x:.1f}" y="{y_top:.1f}" width="{w:.1f}" height="{h:.1f}">\n'
        f'  <div xmlns="http://www.w3.org/1999/xhtml" class="svg-latex" '
        f'style="width:100%;height:100%;display:flex;align-items:center;justify-content:{justify};'
        f'font-size:{size}px;color:{color};font-weight:{weight};line-height:1.32;'
        f'white-space:{ws};text-align:{t_align};{bg_css}">'
        f'<span>{content}</span></div>\n'
        f'</foreignObject>'
    )


def draw_matrix_grid(x, y, rows, cols, cell_w, cell_h, gap, base_color, pattern="dense", bracket=True):
    shades = {
        "teal":   ["#b9d6e0", "#7fb3c5", "#4f8fa5"],
        "orange": ["#f8d6ba", "#f3b787", "#ee995b"],
        "coral":  ["#eac0c0", "#da8e8e", "#c95b5b"],
        "violet": ["#d1c7e3", "#ae9ecf", "#8a74b5"],
        "green":  ["#bce0c4", "#89c195", "#5b9a68"],
        "gray":   ["#dcdfe3", "#b1b5ba", "#85898f"],
    }[base_color]
    out = []
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap

    for r in range(rows):
        for c in range(cols):
            cx = x + c * (cell_w + gap)
            cy = y + r * (cell_h + gap)
            if pattern == "causal" and c > r:
                fill = "#ffffff"
                stroke = "#cbd5e1"
            elif pattern == "diag":
                fill = shades[2] if r == c else "#ffffff"
                stroke = "none" if r == c else "#e2e8f0"
            elif pattern == "stripe_row":
                fill = shades[r % 3]
                stroke = "none"
            elif pattern == "stripe_col":
                fill = shades[c % 3]
                stroke = "none"
            else:
                idx = (r * 7 + c * 13 + r * c) % 3
                fill = shades[idx]
                stroke = "none"
            st_attr = f' stroke="{stroke}" stroke-width="0.6"' if stroke != "none" else ""
            out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w}" height="{cell_h}" rx="1" fill="{fill}"{st_attr}/>')

    if bracket:
        pad = 3.5
        bw = 4
        bx0, by0 = x - pad, y - pad
        bx1, by1 = x + total_w + pad, y + total_h + pad
        out.append(f'<path class="bracket" d="M {bx0+bw} {by0} L {bx0} {by0} L {bx0} {by1} L {bx0+bw} {by1}"/>')
        out.append(f'<path class="bracket" d="M {bx1-bw} {by0} L {bx1} {by0} L {bx1} {by1} L {bx1-bw} {by1}"/>')
    return "\n".join(out), total_w, total_h


def audit_svg_text_collisions(svg_path: Path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    label_boxes = []
    tile_boxes = []
    arrow_lines = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "text":
            x = float(elem.attrib.get("x", "0"))
            y = float(elem.attrib.get("y", "0"))
            cls = elem.attrib.get("class", "")
            anchor = elem.attrib.get("text-anchor", "")
            if not anchor and cls in ("sym-label", "shape-label", "code-pill", "op-text", "sig-line"):
                anchor = "middle"
            txt = "".join(elem.itertext()).strip()
            est_w = sum(11.2 if ord(ch) > 127 else 6.2 for ch in txt)
            est_h = 13.0
            x0 = x - est_w / 2 if anchor == "middle" else x
            x1 = x0 + est_w
            y0 = y - est_h + 2
            y1 = y + 2
            label_boxes.append((x0, y0, x1, y1, txt[:32]))
        elif tag == "foreignObject":
            x0 = float(elem.attrib.get("x", "0"))
            y0 = float(elem.attrib.get("y", "0"))
            w = float(elem.attrib.get("width", "0"))
            h = float(elem.attrib.get("height", "0"))
            txt = "".join(elem.itertext()).strip()
            label_boxes.append((x0, y0, x0 + w, y0 + h, txt[:32]))
        elif tag == "rect":
            w = float(elem.attrib.get("width", "0"))
            h = float(elem.attrib.get("height", "0"))
            if 5 <= w <= 20 and 5 <= h <= 20:
                rx = float(elem.attrib.get("x", "0"))
                ry = float(elem.attrib.get("y", "0"))
                tile_boxes.append((rx, ry, rx + w, ry + h))
        elif tag == "line":
            cls = elem.attrib.get("class", "")
            if "arrow" in cls:
                x1 = float(elem.attrib.get("x1", "0"))
                y1 = float(elem.attrib.get("y1", "0"))
                x2 = float(elem.attrib.get("x2", "0"))
                y2 = float(elem.attrib.get("y2", "0"))
                arrow_lines.append((min(x1, x2), min(y1, y2) - 2, max(x1, x2), max(y1, y2) + 2))

    collisions = []
    for i in range(len(label_boxes)):
        for j in range(i + 1, len(label_boxes)):
            a = label_boxes[i]
            b = label_boxes[j]
            if a[0] < b[2] - 2 and a[2] > b[0] + 2 and a[1] < b[3] - 2 and a[3] > b[1] + 2:
                collisions.append(f"Label-Label: '{a[4]}' <--> '{b[4]}'")

    for a in label_boxes:
        for t in tile_boxes:
            if a[0] < t[2] and a[2] > t[0] and a[1] < t[3] and a[3] > t[1]:
                collisions.append(f"Label-Tile: '{a[4]}' overlaps matrix cell at ({t[0]},{t[1]})")
                break
        for ln in arrow_lines:
            if a[0] < ln[2] and a[2] > ln[0] and a[1] < ln[3] and a[3] > ln[1]:
                collisions.append(f"Label-Arrow: '{a[4]}' overlaps arrow at ({ln[0]}..{ln[2]}, {ln[1]})")

    if collisions:
        print(f"[WARN] {svg_path.name} has {len(collisions)} collisions:")
        for c in collisions:
            print("  -", c)
    else:
        print(f"[AUDIT PASS] {svg_path.name}: 0 collisions across {len(label_boxes)} LaTeX/text boxes & {len(tile_boxes)} matrix cells.")
