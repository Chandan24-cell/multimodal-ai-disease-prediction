"""Generate professional, printable clinical analysis reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPORT_DIR = Path(__file__).resolve().parents[2] / "generated_reports"
NAVY = colors.HexColor("#102445")
BLUE = colors.HexColor("#2563eb")
LIGHT_BLUE = colors.HexColor("#eaf2ff")
MUTED = colors.HexColor("#475569")


def _as_text(value: object, fallback: str = "Not provided") -> str:
    text = str(value or "").strip()
    return text or fallback


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Research Prototype - Not for Clinical Diagnosis")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def generate_pdf_report(prediction_data: dict, rag_context: list, llm_report: str, grad_cam_path: str | None) -> str:
    """Create an A4 PDF report and return its absolute file path."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"clinical_report_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.pdf"

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=colors.white, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=NAVY, spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=colors.HexColor("#1e293b")))
    styles.add(ParagraphStyle(name="Muted", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=MUTED))

    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="Multimodal Healthcare AI - Clinical Analysis Report")
    story = []
    header = Table([[Paragraph("Multimodal Healthcare AI - Clinical Analysis Report", styles["ReportTitle"])]], colWidths=[174 * mm])
    header.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story.extend([header, Spacer(1, 8)])

    story.append(Paragraph("Patient Information", styles["Section"]))
    patient_table = Table([
        [Paragraph("Date", styles["BodySmall"]), Paragraph(datetime.now().strftime("%B %d, %Y %H:%M"), styles["BodySmall"])],
        [Paragraph("Symptoms provided", styles["BodySmall"]), Paragraph(_as_text(prediction_data.get("symptoms")), styles["BodySmall"])],
    ], colWidths=[42 * mm, 132 * mm])
    patient_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.append(patient_table)

    story.append(Paragraph("AI Findings", styles["Section"]))
    rows = [[Paragraph("Rank", styles["BodySmall"]), Paragraph("Finding", styles["BodySmall"]), Paragraph("Model confidence", styles["BodySmall"])]]
    for rank, prediction in enumerate(prediction_data.get("top_predictions", [])[:5], 1):
        rows.append([str(rank), _as_text(prediction.get("disease")), f"{float(prediction.get('confidence', 0)):.2%}"])
    findings = Table(rows, colWidths=[20 * mm, 112 * mm, 42 * mm], repeatRows=1)
    findings.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]), ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (-1, 1), (-1, -1), "RIGHT"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.append(findings)

    if grad_cam_path and Path(grad_cam_path).is_file():
        story.append(Paragraph("Grad-CAM Visualization", styles["Section"]))
        story.append(Image(grad_cam_path, width=92 * mm, height=70 * mm, kind="proportional"))

    story.append(Paragraph("Clinical Report", styles["Section"]))
    for paragraph in _as_text(llm_report).split("\n"):
        if paragraph.strip():
            story.extend([Paragraph(paragraph.strip().replace("&", "&amp;"), styles["BodySmall"]), Spacer(1, 4)])

    story.append(Paragraph("References", styles["Section"]))
    references = []
    for index, chunk in enumerate(rag_context or [], 1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        source = _as_text(metadata.get("source"), "Medical guideline")
        text = _as_text(chunk.get("text"), "") if isinstance(chunk, dict) else ""
        references.extend([Paragraph(f"{index}. {source}: {text}", styles["Muted"]), Spacer(1, 3)])
    story.extend(references or [Paragraph("No RAG references were retrieved.", styles["Muted"])])

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return str(output_path)