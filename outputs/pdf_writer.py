"""
outputs/pdf_writer.py — Professional clinical PDF report generation.

Produces multi-page A4 clinical documents:

1. write_clinical_pdf() — Full consultation report:
   Page 1: Header, metadata, summary, quality metrics
   Next pages: Full transcript table (Time | Speaker | Text | Confidence)

2. write_pdf_evaluation_report() — Evaluation metrics summary PDF

3. write_pdf_dataset_transcript_report() — Per-case card layout PDF

4. write_benchmark_pdf() — Engine comparison benchmark PDF

All PDFs include:
  - Branded header
  - Confidentiality footer
  - Page numbers

Backward-compatible wrappers delegate to the originals in utils.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


def _check_reportlab():
    try:
        import reportlab  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "\n[ERROR] reportlab is not installed.\n"
            "Run: pip install reportlab\n"
        ) from e


def write_clinical_pdf(
    utterances: list["Utterance"],
    output_path: Path,
    metadata: Optional[dict] = None,
    speaker_stats=None,
    quality_report=None,
    clinical_summary=None,
) -> None:
    """
    Write a full multi-page clinical consultation PDF.

    Layout:
      Page 1: Title, metadata table, speaker summary, quality metrics,
              clinical summary (if provided)
      Subsequent pages: Full transcript table
      Footer: Confidentiality notice + page numbers
    """
    _check_reportlab()

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, HRFlowable, PageBreak, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata or {}

    HDR_COLOR   = colors.HexColor("#1a3a5c")
    DOC_BG      = colors.HexColor("#f0f4f8")
    DOCTOR_BG   = colors.HexColor("#ddeeff")
    PATIENT_BG  = colors.HexColor("#ddffee")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=16,
                             textColor=HDR_COLOR, alignment=TA_CENTER, spaceAfter=4)
    meta_s  = ParagraphStyle("M", parent=styles["Normal"], fontSize=9,
                             textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
    cell_s  = ParagraphStyle("C", parent=styles["Normal"], fontSize=9, leading=13)
    h2_s    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                             textColor=HDR_COLOR, spaceBefore=8, spaceAfter=4)
    small_s = ParagraphStyle("Sm", parent=styles["Normal"], fontSize=7,
                             textColor=colors.HexColor("#888888"), alignment=TA_CENTER)

    story = []

    # ── PAGE 1: Cover ──────────────────────────────────────────────────────────
    story.append(Paragraph("MEDICAL CONSULTATION REPORT", title_s))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
        meta_s,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=HDR_COLOR, spaceAfter=10))
    story.append(Spacer(1, 0.3*cm))

    # Metadata table
    meta_rows = [
        [Paragraph("<b>Field</b>", cell_s), Paragraph("<b>Value</b>", cell_s)],
        [Paragraph("Case ID", cell_s),      Paragraph(str(meta.get("case_id", "N/A")), cell_s)],
        [Paragraph("Audio File", cell_s),   Paragraph(str(meta.get("audio_file", "N/A")), cell_s)],
        [Paragraph("Duration", cell_s),     Paragraph(_fmt_dur(meta.get("duration_seconds")), cell_s)],
        [Paragraph("Engine", cell_s),       Paragraph(str(meta.get("engine", "N/A")), cell_s)],
        [Paragraph("Diarization", cell_s),  Paragraph(str(meta.get("diarization_model", "N/A")), cell_s)],
        [Paragraph("ASR Model", cell_s),    Paragraph(str(meta.get("asr_model", "N/A")), cell_s)],
        [Paragraph("Language", cell_s),     Paragraph(str(meta.get("language", "en")), cell_s)],
    ]
    meta_table = Table(meta_rows, colWidths=[4.5*cm, 12.3*cm], repeatRows=1)
    meta_table.setStyle(_two_col_style(HDR_COLOR))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # Speaker summary
    story.append(Paragraph("CONSULTATION SUMMARY", h2_s))
    if speaker_stats and hasattr(speaker_stats, "speakers"):
        spk_rows = [
            [Paragraph("<b>Role</b>", cell_s), Paragraph("<b>Speaking Time</b>", cell_s),
             Paragraph("<b>%</b>", cell_s), Paragraph("<b>Turns</b>", cell_s)]
        ]
        for s in speaker_stats.speakers:
            spk_rows.append([
                Paragraph(s.role, cell_s),
                Paragraph(f"{s.total_speech_seconds:.1f}s", cell_s),
                Paragraph(f"{s.speech_percentage:.1f}%", cell_s),
                Paragraph(str(s.turn_count), cell_s),
            ])
        spk_table = Table(spk_rows, colWidths=[4*cm, 5*cm, 3*cm, 4.8*cm])
        spk_table.setStyle(_basic_table_style(HDR_COLOR))
        story.append(spk_table)
        story.append(Spacer(1, 0.3*cm))

    # Quality metrics
    story.append(Paragraph("QUALITY METRICS", h2_s))
    if quality_report and hasattr(quality_report, "to_dict"):
        qd = quality_report.to_dict()
        avg_c = qd.get("average_confidence")
        q_rows = [
            [Paragraph("<b>Metric</b>", cell_s), Paragraph("<b>Value</b>", cell_s)],
            [Paragraph("Average Confidence", cell_s),
             Paragraph(f"{avg_c*100:.1f}%" if avg_c else "N/A", cell_s)],
            [Paragraph("Low-Confidence Segments", cell_s),
             Paragraph(str(qd.get("low_confidence_count", 0)), cell_s)],
            [Paragraph("Review Required", cell_s),
             Paragraph(str(qd.get("review_required_count", 0)), cell_s)],
            [Paragraph("Overlap Segments", cell_s),
             Paragraph(str(qd.get("overlap_count", 0)), cell_s)],
        ]
        wer = meta.get("wer")
        der = meta.get("der")
        q_rows.append([Paragraph("WER", cell_s), Paragraph(f"{wer*100:.1f}%" if wer is not None else "N/A", cell_s)])
        q_rows.append([Paragraph("DER", cell_s), Paragraph(f"{der*100:.1f}%" if der is not None else "N/A", cell_s)])
        q_table = Table(q_rows, colWidths=[6*cm, 10.8*cm])
        q_table.setStyle(_two_col_style(HDR_COLOR))
        story.append(q_table)
        story.append(Spacer(1, 0.3*cm))

    # Clinical summary (optional)
    if clinical_summary:
        summary_dict = clinical_summary.to_dict() if hasattr(clinical_summary, "to_dict") else clinical_summary
        story.append(Paragraph("CLINICAL SUMMARY", h2_s))
        story.append(Paragraph(
            "<i>⚠ AI-generated. For review only. Not a clinical diagnosis.</i>",
            ParagraphStyle("warn", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#cc6600")),
        ))
        if summary_dict.get("chief_complaint"):
            story.append(Paragraph(f"<b>Chief Complaint:</b> {summary_dict['chief_complaint'][:200]}", cell_s))
        for section, label in [
            ("symptoms", "Symptoms Mentioned"),
            ("medications_mentioned", "Medications Mentioned"),
            ("investigations_mentioned", "Investigations Mentioned"),
        ]:
            items = summary_dict.get(section, [])
            if items:
                findings_str = "; ".join(i["finding"] for i in items[:10])
                story.append(Paragraph(f"<b>{label}:</b> {findings_str}", cell_s))
        story.append(Spacer(1, 0.3*cm))

    # ── TRANSCRIPT ─────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("TRANSCRIPT", h2_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HDR_COLOR, spaceAfter=6))

    COL_W = [2.2*cm, 2.4*cm, 11.2*cm, 1.0*cm]
    hdr_row = [
        Paragraph("<b>Time</b>", cell_s),
        Paragraph("<b>Speaker</b>", cell_s),
        Paragraph("<b>Transcript</b>", cell_s),
        Paragraph("<b>Conf</b>", cell_s),
    ]
    table_data = [hdr_row]
    row_bgs: list[tuple] = []

    for idx, u in enumerate(utterances, start=1):
        is_doctor = u.speaker_label == "Doctor"
        bg = DOCTOR_BG if is_doctor else PATIENT_BG
        spk_color = "#0a3d8f" if is_doctor else "#0a6b40"
        review_tag = " ⚠" if u.review_required else ""
        overlap_tag = " [OVL]" if u.overlap else ""

        from utils import format_timestamp as _fmt_ts
        table_data.append([
            Paragraph(_fmt_ts(u.start_time), cell_s),
            Paragraph(
                f'<font color="{spk_color}"><b>{u.speaker_label}{review_tag}{overlap_tag}</b></font>',
                cell_s,
            ),
            Paragraph(u.text, cell_s),
            Paragraph(
                f"{u.confidence*100:.0f}%" if u.confidence is not None else "—",
                cell_s,
            ),
        ])
        row_bgs.append((idx, bg))

    ts_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HDR_COLOR),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_idx, bg in row_bgs:
        ts_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    transcript_table = Table(table_data, colWidths=COL_W, repeatRows=1)
    transcript_table.setStyle(TableStyle(ts_cmds))
    story.append(transcript_table)

    # Footer note
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"), spaceAfter=3))
    story.append(Paragraph(
        "CONFIDENTIAL — AI-GENERATED TRANSCRIPT — FOR CLINICAL REVIEW ONLY. "
        "This transcript was produced by an automated system and has not been verified by a clinician.",
        small_s,
    ))

    def _page_num(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawRightString(
            doc.pagesize[0] - 2*cm, 1.2*cm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_num, onLaterPages=_page_num)
    print(f"[pdf_writer] Clinical PDF → {output_path}")


def write_benchmark_pdf(benchmark_results: dict, output_path: Path) -> None:
    """Write a benchmark comparison PDF showing engine metrics side by side."""
    _check_reportlab()

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HDR_COLOR = colors.HexColor("#1a3a5c")

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()
    cell_s = ParagraphStyle("c", parent=styles["Normal"], fontSize=9, leading=12)

    story = [
        Paragraph("ENGINE COMPARISON BENCHMARK", ParagraphStyle(
            "T", parent=styles["Heading1"], fontSize=15,
            textColor=HDR_COLOR, alignment=TA_CENTER, spaceAfter=4,
        )),
        Paragraph(
            f"Dataset: {benchmark_results.get('dataset', 'N/A')} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ParagraphStyle("M", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#555555"), alignment=TA_CENTER),
        ),
        HRFlowable(width="100%", thickness=1, color=HDR_COLOR, spaceAfter=12),
        Spacer(1, 0.3*cm),
    ]

    rows = [
        [Paragraph(f"<b>{h}</b>", cell_s) for h in
         ["Engine", "WER", "DER", "Processing Time", "Cases"]]
    ]
    for e in benchmark_results.get("engines", []):
        wer = e.get("wer")
        der = e.get("der")
        rows.append([
            Paragraph(e.get("engine", "N/A"), cell_s),
            Paragraph(f"{wer*100:.1f}%" if wer is not None else "N/A", cell_s),
            Paragraph(f"{der*100:.1f}%" if der is not None else "N/A", cell_s),
            Paragraph(f"{e.get('processing_time_seconds', 0):.1f}s", cell_s),
            Paragraph(str(e.get("cases_evaluated", 0)), cell_s),
        ])

    t = Table(rows, colWidths=[5*cm, 3*cm, 3*cm, 4*cm, 2.8*cm])
    t.setStyle(_basic_table_style(HDR_COLOR))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "NOTE: N/A indicates metric could not be calculated (no ground truth available). "
        "WER requires reference transcriptions. DER requires speaker-annotated two-speaker data.",
        ParagraphStyle("n", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#666666")),
    ))

    def _pn(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawRightString(doc.pagesize[0] - 2*cm, 1.2*cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_pn, onLaterPages=_pn)
    print(f"[pdf_writer] Benchmark PDF → {output_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_dur(seconds) -> str:
    if seconds is None:
        return "N/A"
    s = float(seconds)
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _two_col_style(hdr_color):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), hdr_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUND",(0, 1), (-1, -1),
         [colors.HexColor("#f5f8ff"), colors.white]),
    ])


def _basic_table_style(hdr_color):
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), hdr_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ])


# ── Backward-compatible delegators ────────────────────────────────────────────

def write_pdf_transcript(utterances, output_path, audio_filename="", engine_name=""):
    """Backward-compatible wrapper — delegates to utils.py."""
    from utils import write_pdf_transcript as _orig
    _orig(utterances, output_path, audio_filename=audio_filename, engine_name=engine_name)


def write_pdf_evaluation_report(aggregate, output_path):
    """Backward-compatible wrapper — delegates to utils.py."""
    from utils import write_pdf_evaluation_report as _orig
    _orig(aggregate, output_path)


def write_pdf_dataset_transcript_report(per_case, aggregate, output_path):
    """Backward-compatible wrapper — delegates to utils.py."""
    from utils import write_pdf_dataset_transcript_report as _orig
    _orig(per_case, aggregate, output_path)
