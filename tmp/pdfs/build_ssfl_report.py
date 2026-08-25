from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tmp" / "pdfs" / "ssfl_comprehensive_explanation.md"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "ssfl_comprehensive_paper_explanation.pdf"

NAVY = colors.HexColor("#12304A")
TEAL = colors.HexColor("#087E8B")
CYAN = colors.HexColor("#DDF3F5")
PALE = colors.HexColor("#F4F7F9")
MID = colors.HexColor("#637887")
INK = colors.HexColor("#17232D")
GOLD = colors.HexColor("#D59B2D")
LINE = colors.HexColor("#CFD9DF")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    italic_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ]
    mono_candidates = [
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    ]
    regular = next((p for p in candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    italic = next((p for p in italic_candidates if p.exists()), None)
    mono = next((p for p in mono_candidates if p.exists()), None)
    if regular and bold and italic and mono:
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
        pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(italic)))
        pdfmetrics.registerFont(TTFont("ReportMono", str(mono)))
        pdfmetrics.registerFontFamily(
            "ReportSans",
            normal="ReportSans",
            bold="ReportSans-Bold",
            italic="ReportSans-Italic",
            boldItalic="ReportSans-Bold",
        )
        return "ReportSans", "ReportSans-Bold", "ReportSans-Italic", "ReportMono"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


FONT, FONT_BOLD, FONT_ITALIC, FONT_MONO = register_fonts()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        body_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[body_frame], onPage=draw_cover_page),
                PageTemplate(id="Body", frames=[body_frame], onPage=draw_body_page),
            ]
        )

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            if style_name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif style_name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def draw_cover_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, 18 * mm, height, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(18 * mm, height - 18 * mm, width - 18 * mm, 2.2 * mm, fill=1, stroke=0)
    canvas.restoreState()


def draw_body_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    page_no = canvas.getPageNumber()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, height - 18 * mm, width - doc.rightMargin, height - 18 * mm)
    canvas.setFont(FONT_BOLD, 8.2)
    canvas.setFillColor(NAVY)
    canvas.drawString(doc.leftMargin, height - 14.8 * mm, "SSFL FOR IOT INTRUSION DETECTION")
    canvas.setFont(FONT, 7.8)
    canvas.setFillColor(MID)
    canvas.drawRightString(width - doc.rightMargin, height - 14.8 * mm, "COMPREHENSIVE PAPER EXPLANATION")
    canvas.line(doc.leftMargin, 15.5 * mm, width - doc.rightMargin, 15.5 * mm)
    canvas.setFont(FONT, 8)
    canvas.drawString(doc.leftMargin, 10.8 * mm, "Zhao et al. | IEEE Internet of Things Journal | 2023")
    canvas.setFont(FONT_BOLD, 8)
    canvas.setFillColor(NAVY)
    canvas.drawRightString(width - doc.rightMargin, 10.8 * mm, str(page_no))
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName=FONT,
        fontSize=9.25,
        leading=13.2,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=7,
        allowWidows=0,
        allowOrphans=0,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName=FONT_BOLD,
        fontSize=16,
        leading=19,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=1,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=11.4,
        leading=14,
        textColor=TEAL,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=1,
    )
    code = ParagraphStyle(
        "Code",
        parent=body,
        fontName=FONT_MONO,
        fontSize=8.1,
        leading=11.2,
        leftIndent=7,
        rightIndent=7,
        borderColor=LINE,
        borderWidth=0.7,
        borderPadding=7,
        backColor=PALE,
        spaceBefore=3,
        spaceAfter=8,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontName=FONT_ITALIC,
        fontSize=8.1,
        leading=10.5,
        textColor=MID,
        alignment=TA_CENTER,
        spaceAfter=9,
    )
    toc_title = ParagraphStyle(
        "TOCTitle",
        parent=h1,
        fontSize=20,
        leading=23,
        spaceAfter=12,
    )
    return body, h1, h2, code, caption, toc_title


BODY, H1, H2, CODE, CAPTION, TOC_TITLE = styles()


def inline_markup(text: str) -> str:
    safe = escape(text.strip())
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"\*(.+?)\*", r"<i>\1</i>", safe)
    safe = re.sub(r"`(.+?)`", r"<font name='%s'>\1</font>" % FONT_MONO, safe)
    return safe


def make_table(rows: list[list[str]], available_width: float):
    if not rows:
        return Spacer(1, 0)
    ncols = max(len(r) for r in rows)
    padded = [r + [""] * (ncols - len(r)) for r in rows]
    cell_style = ParagraphStyle(
        "TableCell",
        parent=BODY,
        fontSize=7.7,
        leading=9.6,
        spaceAfter=0,
    )
    head_style = ParagraphStyle(
        "TableHead",
        parent=cell_style,
        fontName=FONT_BOLD,
        textColor=WHITE,
    )
    data = []
    for row_idx, row in enumerate(padded):
        style = head_style if row_idx == 0 else cell_style
        data.append([Paragraph(inline_markup(cell), style) for cell in row])
    if ncols == 4:
        widths = [available_width * 0.31] + [available_width * 0.23] * 3
    elif ncols == 6:
        widths = [available_width * 0.24] + [available_width * 0.152] * 5
    else:
        widths = [available_width / ncols] * ncols
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.45, LINE),
                ("BOX", (0, 0), (-1, -1), 0.65, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return KeepTogether([Spacer(1, 3), table, Spacer(1, 8)])


def parse_markdown(md: str, available_width: float):
    lines = md.splitlines()
    story = []
    idx = 0
    paragraph: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            joined = " ".join(part.strip() for part in paragraph)
            story.append(Paragraph(inline_markup(joined), BODY))
            paragraph = []

    while idx < len(lines):
        line = lines[idx]
        if line.startswith("# ") or line.startswith("## A comprehensive explanation"):
            flush_paragraph()
            idx += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[3:]), H1))
            idx += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[4:]), H2))
            idx += 1
            continue
        if line.startswith("    "):
            flush_paragraph()
            code_lines = []
            while idx < len(lines) and (lines[idx].startswith("    ") or lines[idx].strip() == ""):
                if lines[idx].startswith("    "):
                    code_lines.append(escape(lines[idx][4:]))
                elif code_lines:
                    code_lines.append("")
                idx += 1
            while code_lines and code_lines[-1] == "":
                code_lines.pop()
            story.append(Paragraph("<br/>".join(code_lines), CODE))
            continue
        if line.startswith("|"):
            flush_paragraph()
            table_lines = []
            while idx < len(lines) and lines[idx].startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            rows = []
            for table_line in table_lines:
                cells = [c.strip() for c in table_line.strip().strip("|").split("|")]
                if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    continue
                rows.append(cells)
            story.append(make_table(rows, available_width))
            continue
        if line.strip() == "":
            flush_paragraph()
            idx += 1
            continue
        paragraph.append(line)
        idx += 1
    flush_paragraph()
    return story


def cover_story():
    title = ParagraphStyle(
        "CoverTitle",
        fontName=FONT_BOLD,
        fontSize=28,
        leading=33,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=14,
    )
    subtitle = ParagraphStyle(
        "CoverSubtitle",
        fontName=FONT,
        fontSize=13.5,
        leading=19,
        textColor=colors.HexColor("#CFE4EE"),
        alignment=TA_LEFT,
    )
    meta = ParagraphStyle(
        "CoverMeta",
        fontName=FONT,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#D9E7ED"),
    )
    highlight = ParagraphStyle(
        "CoverHighlight",
        fontName=FONT_BOLD,
        fontSize=11,
        leading=16,
        textColor=NAVY,
        alignment=TA_LEFT,
    )
    box = Table(
        [[Paragraph("87.40% / 86.70% / 84.22%<br/><font size='8'>Reported SSFL accuracy across three non-IID scenarios</font>", highlight)]],
        colWidths=[154 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CYAN),
                ("BOX", (0, 0), (-1, -1), 1, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return [
        Spacer(1, 34 * mm),
        Paragraph("SEMISUPERVISED<br/>FEDERATED LEARNING", title),
        HRFlowable(width="100%", thickness=2.4, color=GOLD, spaceBefore=1, spaceAfter=12),
        Paragraph("A comprehensive explanation of the IoT intrusion-detection paper by Zhao et al.", subtitle),
        Spacer(1, 16 * mm),
        box,
        Spacer(1, 30 * mm),
        Paragraph("Paper explained", meta),
        Paragraph("Semisupervised Federated-Learning-Based Intrusion Detection Method for Internet of Things", ParagraphStyle("PaperName", parent=meta, fontName=FONT_BOLD, fontSize=11.5, leading=16, textColor=WHITE)),
        Spacer(1, 5 * mm),
        Paragraph("IEEE Internet of Things Journal, Vol. 10, No. 10, May 2023<br/>DOI: 10.1109/JIOT.2022.3175918", meta),
        Spacer(1, 9 * mm),
        Paragraph("Prepared as a study-oriented technical report | 22 July 2026", meta),
        NextPageTemplate("Body"),
        PageBreak(),
    ]


def toc_story():
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName=FONT_BOLD,
            fontSize=9.4,
            leading=13,
            textColor=NAVY,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        ParagraphStyle(
            "TOC2",
            fontName=FONT,
            fontSize=8.3,
            leading=11,
            textColor=MID,
            leftIndent=14,
            firstLineIndent=0,
            spaceBefore=1,
        ),
    ]
    return [
        Paragraph("Contents", TOC_TITLE),
        Paragraph("A guided map of the motivation, method, experiments, results, and critical assessment.", BODY),
        Spacer(1, 5),
        toc,
        PageBreak(),
    ]


def process_flow_story():
    step_style = ParagraphStyle(
        "FlowStep",
        parent=BODY,
        fontName=FONT_BOLD,
        fontSize=7.6,
        leading=9.3,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    arrow_style = ParagraphStyle(
        "Arrow",
        parent=step_style,
        fontSize=13,
        textColor=TEAL,
    )
    labels = [
        "Private labeled<br/>traffic",
        "Local<br/>classifier",
        "Familiarity<br/>discriminator",
        "Hard-label<br/>vote",
        "Global pseudo-label<br/>training",
    ]
    cells = []
    for pos, label in enumerate(labels):
        cells.append(Paragraph(label, step_style))
        if pos < len(labels) - 1:
            cells.append(Paragraph("&gt;", arrow_style))
    widths = [29 * mm, 8 * mm, 25 * mm, 8 * mm, 30 * mm, 8 * mm, 25 * mm, 8 * mm, 33 * mm]
    flow = Table([cells], colWidths=widths, rowHeights=[17 * mm], hAlign="CENTER")
    style_cmds = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for col in range(0, len(cells), 2):
        style_cmds.extend(
            [
                ("BACKGROUND", (col, 0), (col, 0), CYAN if col != 6 else colors.HexColor("#FFF3D8")),
                ("BOX", (col, 0), (col, 0), 0.8, TEAL if col != 6 else GOLD),
                ("LEFTPADDING", (col, 0), (col, 0), 4),
                ("RIGHTPADDING", (col, 0), (col, 0), 4),
            ]
        )
    flow.setStyle(TableStyle(style_cmds))
    return KeepTogether(
        [
            Spacer(1, 4),
            flow,
            Paragraph("Figure 1. Conceptual information flow in SSFL. Unfamiliar clients abstain before voting.", CAPTION),
        ]
    )


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md = SOURCE.read_text(encoding="utf-8")
    doc = ReportDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
        title="Semisupervised Federated Learning for IoT Intrusion Detection",
        author="Comprehensive study report based on Zhao et al.",
        subject="Explanation and critical analysis of SSFL for IoT intrusion detection",
        creator="Codex",
    )
    story = []
    story.extend(cover_story())
    story.extend(toc_story())
    parsed = parse_markdown(md, doc.width)
    inserted_flow = False
    for item in parsed:
        story.append(item)
        if (
            not inserted_flow
            and isinstance(item, Paragraph)
            and item.style.name == "H1"
            and item.getPlainText() == "1. The problem the paper is trying to solve"
        ):
            story.insert(len(story) - 1, process_flow_story())
            inserted_flow = True
    doc.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
