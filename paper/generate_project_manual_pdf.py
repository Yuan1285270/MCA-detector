from __future__ import annotations

import csv
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/project_manual.md"
OUTPUT = ROOT / "output/pdf/mca_detector_project_manual.pdf"


def register_fonts() -> str:
    candidates = [
        (Path("/System/Library/Fonts/STHeiti Light.ttc"), "STHeitiLight"),
        (Path("/System/Library/Fonts/Supplemental/Songti.ttc"), "Songti"),
    ]
    for path, font_name in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(font_name, str(path), subfontIndex=0))
            return font_name
    font_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def normalize_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font color='#334155'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def split_table_row(line: str) -> list[str]:
    row = line.strip().strip("|")
    return [cell.strip() for cell in row.split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def build_styles(font: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ManualTitle",
            parent=base["Title"],
            fontName=font,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "ManualSubtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=18,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "ManualH1",
            parent=base["Heading1"],
            fontName=font,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "ManualH2",
            parent=base["Heading2"],
            fontName=font,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=10,
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "ManualBody",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=6,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "ManualBullet",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=14,
            leftIndent=12,
            firstLineIndent=-8,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=3,
            wordWrap="CJK",
        ),
        "quote": ParagraphStyle(
            "ManualQuote",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9,
            leading=14,
            leftIndent=10,
            borderColor=colors.HexColor("#93c5fd"),
            borderWidth=1,
            borderPadding=6,
            backColor=colors.HexColor("#eff6ff"),
            textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "ManualCode",
            parent=base["Code"],
            fontName=font,
            fontSize=8,
            leading=11,
            leftIndent=4,
            rightIndent=4,
            borderColor=colors.HexColor("#e2e8f0"),
            borderWidth=0.5,
            borderPadding=6,
            backColor=colors.HexColor("#f8fafc"),
            textColor=colors.HexColor("#334155"),
            spaceBefore=4,
            spaceAfter=8,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "ManualTable",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.4,
            leading=10,
            textColor=colors.HexColor("#1f2937"),
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "ManualTableHeader",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.4,
            leading=10,
            textColor=colors.white,
            wordWrap="CJK",
        ),
    }


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    data = []
    for r, row in enumerate(rows):
        style = styles["table_header"] if r == 0 else styles["table"]
        data.append([Paragraph(normalize_inline(cell), style) for cell in row])

    col_count = max(len(row) for row in rows)
    page_width = A4[0] - 36 * mm
    if col_count == 2:
        col_widths = [page_width * 0.32, page_width * 0.68]
    elif col_count == 3:
        col_widths = [page_width * 0.24, page_width * 0.24, page_width * 0.52]
    elif col_count == 4:
        col_widths = [page_width * 0.22, page_width * 0.24, page_width * 0.27, page_width * 0.27]
    else:
        col_widths = [page_width / col_count] * col_count

    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ffffff")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def parse_markdown(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    lines = markdown.splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    pending_title = True

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(normalize_inline(x) for x in code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        if line.startswith("# "):
            if not pending_title:
                story.append(PageBreak())
            story.append(Paragraph(normalize_inline(line[2:].strip()), styles["title"]))
            pending_title = False
            i += 1
            continue

        if pending_title is False and line.startswith("版本："):
            meta = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("## "):
                meta.append(lines[j].strip())
                j += 1
            story.append(Paragraph("<br/>".join(normalize_inline(x) for x in meta), styles["subtitle"]))
            i = j
            continue

        if line.startswith("## "):
            story.append(Paragraph(normalize_inline(line[3:].strip()), styles["h1"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(normalize_inline(line[4:].strip()), styles["h2"]))
            i += 1
            continue

        if line.startswith("> "):
            story.append(Paragraph(normalize_inline(line[2:].strip()), styles["quote"]))
            i += 1
            continue

        if line.startswith("- "):
            story.append(Paragraph("• " + normalize_inline(line[2:].strip()), styles["bullet"]))
            i += 1
            continue

        if re.match(r"\d+\. ", line):
            story.append(Paragraph(normalize_inline(line.strip()), styles["bullet"]))
            i += 1
            continue

        if line.startswith("|"):
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_table_row(lines[i])
                if not is_separator_row(cells):
                    table_rows.append(cells)
                i += 1
            if table_rows:
                story.append(make_table(table_rows, styles))
                story.append(Spacer(1, 6))
            continue

        para = [line.strip()]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].rstrip()
            if (
                not nxt.strip()
                or nxt.startswith("#")
                or nxt.startswith("```")
                or nxt.startswith("|")
                or nxt.startswith("- ")
                or nxt.startswith("> ")
                or re.match(r"\d+\. ", nxt)
            ):
                break
            para.append(nxt.strip())
            j += 1
        story.append(Paragraph(normalize_inline(" ".join(para)), styles["body"]))
        i = j

    return story


class ManualDocTemplate(BaseDocTemplate):
    def __init__(self, filename: Path, font: str):
        self.font = font
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=17 * mm,
            bottomMargin=17 * mm,
            title="MCA Detector 專案說明書",
            author="MCA Detector Team",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=self.draw_page)])

    def draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(self.font, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 10 * mm, "MCA Detector Project Manual")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    font = register_fonts()
    styles = build_styles(font)
    markdown = SOURCE.read_text(encoding="utf-8")
    story = parse_markdown(markdown, styles)
    doc = ManualDocTemplate(OUTPUT, font)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
