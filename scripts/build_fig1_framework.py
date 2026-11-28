#!/usr/bin/env python3
"""Build Fig. 1 (DynVecAudit Architecture & 4-Gate Hygiene Framework) in pure B&W editable PPTX.

Strict house style:
  - 7.20 x 2.72 in canvas (1:1 matched to two-column print width so >=7.8 pt native stays >=7.2 pt)
  - White background, 1 pt sharp non-rounded black borders, subtle bottom-right shadow
  - Dashed outer channel frames with bold uppercase English channel name + italic subtitle
  - Times New Roman font, orthogonal black bus connectors aligned strictly to box midpoints
  - Exports PPTX -> LibreOffice vector PDF + 400 dpi PNG
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "latex" / "figures"

BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_FILL = RGBColor(0xF5, 0xF5, 0xF5)
DARK_FILL = RGBColor(0xE8, 0xE8, 0xE8)
SHADOW_GRAY = RGBColor(0xD8, 0xD8, 0xD8)


def _add_Formatted_runs(p, spec, sz: float, bold: bool, italic: bool) -> None:
    """Support plain string or list of (substr, is_sub, is_italic_override) segments."""
    if isinstance(spec, str):
        segments = [(spec, False, italic)]
    else:
        segments = spec
    for text, is_sub, it_flag in segments:
        run = p.add_run()
        run.text = text
        run.font.name = "Times New Roman"
        run.font.size = Pt(sz)
        run.font.bold = bold
        run.font.italic = it_flag
        run.font.color.rgb = BLACK
        if is_sub:
            rPr = run._r.get_or_add_rPr()
            rPr.set("baseline", "-25000")


def _set_text(
    tf,
    lines: list[tuple[object, float, bool, bool]],
    align: str = "center",
    valign=MSO_ANCHOR.MIDDLE,
    margin_lr: float = 0.04,
    margin_tb: float = 0.02,
) -> None:
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_lr)
    tf.margin_right = Inches(margin_lr)
    tf.margin_top = Inches(margin_tb)
    tf.margin_bottom = Inches(margin_tb)
    align_map = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}
    for idx, (spec, sz, bold, italic) in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.alignment = align_map[align]
        p.space_before = Pt(0.3)
        p.space_after = Pt(0.3)
        _add_Formatted_runs(p, spec, sz, bold, italic)


def add_box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[tuple[object, float, bool, bool]],
    *,
    fill: RGBColor = WHITE,
    align: str = "center",
    shadow: bool = True,
    line_pt: float = 1.0,
) -> None:
    if shadow:
        sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.022), Inches(y + 0.022), Inches(w), Inches(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = SHADOW_GRAY
        sh.line.fill.background()
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = BLACK
    box.line.width = Pt(line_pt)
    _set_text(box.text_frame, lines, align=align)


def add_channel(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
) -> None:
    frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    frame.fill.background()
    frame.line.color.rgb = BLACK
    frame.line.width = Pt(1.0)
    frame.line.dash_style = MSO_LINE_DASH_STYLE.DASH

    hdr = slide.shapes.add_textbox(Inches(x + 0.04), Inches(y + 0.03), Inches(w - 0.08), Inches(0.36))
    _set_text(
        hdr.text_frame,
        [(title, 8.8, True, False), (subtitle, 7.8, False, True)],
        align="left",
        valign=MSO_ANCHOR.TOP,
        margin_lr=0.01,
        margin_tb=0.01,
    )


def add_seg(slide, x1: float, y1: float, x2: float, y2: float, arrow: bool = False, width_pt: float = 1.05) -> None:
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = BLACK
    conn.line.width = Pt(width_pt)
    if arrow:
        ln = conn.line._get_or_add_ln()
        ln.append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="sm" len="med"/>'))


def add_bus(slide, x_left: float, x_bus: float, x_right: float, ys_in: list[float], ys_out: list[float]) -> None:
    for y in ys_in:
        add_seg(slide, x_left, y, x_bus, y, arrow=False)
    y_min = min(min(ys_in), min(ys_out))
    y_max = max(max(ys_in), max(ys_out))
    add_seg(slide, x_bus, y_min, x_bus, y_max, arrow=False)
    for y in ys_out:
        add_seg(slide, x_bus, y, x_right, y, arrow=True)


def build() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(7.20)
    prs.slide_height = Inches(2.72)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(7.20), Inches(2.72))
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()

    # Channel 1: WORKLOAD STREAMS (x=0.05..1.63)
    add_channel(slide, 0.05, 0.06, 1.58, 2.60, "WORKLOAD STREAMS", "128-d–960-d, exact Top-10 GT")
    tsw_runs = [("Abrupt (", False, False), ("t", False, True), ("sw", True, False), (") & (u, Q) grid", False, False)]
    add_box(slide, 0.11, 0.46, 1.46, 0.65, [
        ("Drift & Scale (100K–10M)", 8.2, True, False),
        ("SIFT / BigANN / Wiki-768", 7.7, False, False),
        (tsw_runs, 7.7, False, False),
    ], fill=GRAY_FILL)
    add_box(slide, 0.11, 1.19, 1.46, 0.65, [
        ("Mixed CRUD & Runbook", 8.2, True, False),
        ("20% / 30% / 50% CRUD", 7.7, False, False),
        ("0%–50% Tombstone Sweep", 7.7, False, False),
    ], fill=GRAY_FILL)
    add_box(slide, 0.11, 1.92, 1.46, 0.65, [
        ("Real & Pacing Suites", 8.2, True, False),
        ("6 drift types, 2,400 replay traces", 7.7, False, False),
        ("Wikimedia, CLINC, CLIP", 7.7, False, False),
    ], fill=GRAY_FILL)

    # Channel 2: DUAL-PLANE EXECUTION (x=1.81..3.59)
    add_channel(slide, 1.81, 0.06, 1.78, 2.60, "DUAL-PLANE EXECUTION", "Isolated core + exact replay")
    add_box(slide, 1.87, 0.46, 1.66, 1.00, [
        ("7 Codebases / 8 Backends", 8.3, True, False),
        ("FAISS-IVF/HNSW | hnswlib | Quake", 7.7, False, False),
        ("DiskANN | Annoy | USearch/R | hnsw_rs", 7.7, False, False),
        ("Attain. ≥ 0.965, p99, VmHWM", 7.7, False, True),
    ], fill=WHITE)
    add_box(slide, 1.87, 1.56, 1.66, 1.01, [
        ("Replay & Paced 16-Sh. Bridge", 8.3, True, False),
        ("Prop. 1 MILP & 5.58× Rust CLI", 7.7, False, False),
        ("Paced vs. Greedy (Exp 11-A/B)", 7.7, False, False),
        ("Causal vs. Leaky, c ∈ {1..8}", 7.7, False, False),
    ], fill=GRAY_FILL)

    # Orthogonal Bus Ch1 -> Ch2 (box midpoints -> box midpoints)
    add_bus(slide, 1.57, 1.72, 1.87, [0.785, 1.515, 2.245], [0.960, 2.065])

    # Channel 3: THREE-ACT AUDIT GATES (x=3.77..5.85)
    add_channel(slide, 3.77, 0.06, 2.08, 2.60, "THREE-ACT AUDIT CHECKS", "Five calibrated rules (Q,P,C,S,H)")
    g2_desc = [("S: Rand. hub+", False, False), ("t", False, True), ("sw", True, False), (", |ρ|<0.70; H: end+refresh", False, False)]
    gates = [
        ("Act 1 — Quality (Q)", "≥95% windows at ≥0.965 recall"),
        ("Act 2 — Placement (P)", "Idle sibling, w/C ≤ 1.18, probe ≤ 1.25×"),
        ("Act 3 — Split & Horizon (S,H)", g2_desc),
        ("Act 3 — Cadence & Pacing (C)", "Bracket c,c+1; cap ⌊Bj/(J-1)⌋"),
    ]
    gate_mids = []
    gy = 0.46
    for title_g, desc_g in gates:
        add_box(slide, 3.83, gy, 1.96, 0.47, [
            (title_g, 8.1, True, False),
            (desc_g, 7.6, False, False),
        ], fill=WHITE)
        gate_mids.append(gy + 0.235)
        gy += 0.545

    # Orthogonal Bus Ch2 -> Ch3
    add_bus(slide, 3.53, 3.68, 3.83, [0.960, 2.065], gate_mids)

    # Channel 4: AUDITED OUTPUTS (x=6.03..7.15)
    add_channel(slide, 6.03, 0.06, 1.12, 2.60, "OUTCOMES", "5,462 runs")
    add_box(slide, 6.09, 0.46, 1.00, 0.65, [
        ("Act 1: Quality", 8.2, True, False),
        ("12/17 Cfg Flips", 7.6, False, False),
        ("IVF CPU, graph p99", 7.6, False, False),
    ], fill=DARK_FILL)
    add_box(slide, 6.09, 1.19, 1.00, 0.65, [
        ("Act 2: Placement", 8.2, True, False),
        ("4/6 Policy Flips", 7.6, False, False),
        ("Probe AUC 1.00", 7.6, False, False),
    ], fill=DARK_FILL)
    add_box(slide, 6.09, 1.92, 1.00, 0.65, [
        ("Act 3: Pacing", 8.2, True, False),
        ("+24.4 pp Replay", 7.6, False, False),
        ("+26.3 pp Live", 7.6, False, False),
    ], fill=DARK_FILL)

    # Orthogonal Bus Ch3 -> Ch4
    add_bus(slide, 5.79, 5.94, 6.09, gate_mids, [0.785, 1.515, 2.245])

    pptx_path = FIG_DIR / "fig1_framework.pptx"
    prs.save(str(pptx_path))
    print(f"Saved {pptx_path}")

    subprocess.run(
        ["/opt/homebrew/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(FIG_DIR), str(pptx_path)],
        check=True,
    )
    pdf_path = FIG_DIR / "fig1_framework.pdf"
    png_prefix = FIG_DIR / "fig1_framework"
    subprocess.run(
        ["/opt/homebrew/bin/pdftoppm", "-png", "-r", "400", "-singlefile", str(pdf_path), str(png_prefix)],
        check=True,
    )
    print(f"Rendered {pdf_path} and {png_prefix}.png")


if __name__ == "__main__":
    build()
