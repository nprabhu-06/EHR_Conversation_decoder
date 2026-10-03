"""
utils.py — Shared helper utilities.

Covers:
  - Audio file validation
  - Timestamp formatting
  - Multi-speaker audio file merging (for datasets that store each speaker
    in a separate WAV file)
  - Transcript output writers (JSON, TXT, and PDF)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transcriber import Utterance

# Audio extensions we accept directly (pydub can handle all of these via ffmpeg)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus",
}


# ── FFmpeg / environment helpers ───────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH, False otherwise."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# ── Audio validation ───────────────────────────────────────────────────────────

def validate_audio_file(path: str | Path) -> Path:
    """
    Validate that `path` exists and has a supported audio extension.

    Returns the resolved Path on success; raises ValueError on failure.
    """
    audio_path = Path(path).resolve()

    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")

    ext = audio_path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {ext!r}\n"
            f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    return audio_path


def ensure_wav(audio_path: Path, output_dir: Path) -> Path:
    """
    Convert an audio file to 16 kHz mono 16-bit PCM WAV using FFmpeg directly.

    Uses FFmpeg subprocess instead of pydub's Python audio pipeline, which
    avoids the 'pyaudioop' / 'audioop' dependency removed in Python 3.13+.
    FFmpeg is already a required system dependency so this is always safe.

    Returns the path to the normalised WAV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / (audio_path.stem + "_16k.wav")

    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output without asking
        "-i", str(audio_path),  # input file
        "-ac", "1",             # mono
        "-ar", "16000",         # 16 kHz sample rate
        "-sample_fmt", "s16",   # 16-bit PCM
        str(wav_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg audio conversion failed for {audio_path.name}:\n{err}"
        )

    return wav_path


# ── Timestamp formatting ───────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """
    Convert a float number of seconds to HH:MM:SS format.

    Examples:
        format_timestamp(0.0)    → "00:00:00"
        format_timestamp(72.5)   → "00:01:12"
        format_timestamp(3661.0) → "01:01:01"
    """
    seconds = max(0.0, seconds)
    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ── Multi-speaker audio merging ────────────────────────────────────────────────

def merge_speaker_audio_files(
    file_map: dict[str, Path],
    output_path: Path,
    sample_rate: int = 16_000,
) -> Path:
    """
    Merge per-speaker WAV files into a single mixed audio file.

    Some diarization datasets store each speaker's audio in a separate
    channel or file. This function overlays them into a single mono WAV
    so the transcription engines can process it.

    Args:
        file_map: Mapping of speaker label → WAV file path.
                  Example: {"Doctor": Path("doctor.wav"), "Patient": Path("patient.wav")}
        output_path: Where to write the merged WAV.
        sample_rate: Target sample rate (default 16 kHz).

    Returns:
        Path to the merged WAV file.
    """
    from pydub import AudioSegment

    if not file_map:
        raise ValueError("file_map must contain at least one entry.")

    segments = []
    for label, path in file_map.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Speaker audio file not found for {label!r}: {path}")
        seg = AudioSegment.from_file(str(path))
        seg = seg.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
        segments.append(seg)

    # Overlay all segments (mix them at the same time-offset = 0)
    merged = segments[0]
    for seg in segments[1:]:
        # Overlay: pads the shorter segment with silence
        merged = merged.overlay(seg)

    merged.export(str(output_path), format="wav")
    print(f"[utils] Merged {len(file_map)} speaker file(s) → {output_path}")
    return output_path


# ── Transcript writers ─────────────────────────────────────────────────────────

def write_json_transcript(utterances: list[Utterance], output_path: Path) -> None:
    """
    Write a structured JSON transcript.

    Output format (array of objects):
    [
      {
        "speaker_label": "Doctor",
        "text": "How are you feeling today?",
        "start_time": 12.34,
        "end_time": 14.56,
        "confidence": 0.97
      },
      ...
    ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(u) for u in utterances]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[utils] JSON transcript saved → {output_path}")


def write_txt_transcript(utterances: list[Utterance], output_path: Path) -> None:
    """
    Write a human-readable plain-text transcript.

    Format:
        [HH:MM:SS] Doctor: How are you feeling today?
        [HH:MM:SS] Patient: I've had a headache for three days.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for u in utterances:
        ts = format_timestamp(u.start_time)
        lines.append(f"[{ts}] {u.speaker_label}: {u.text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[utils] TXT transcript saved  → {output_path}")


def load_json_transcript(path: Path) -> list[dict]:
    """Load a previously saved JSON transcript for evaluation or inspection."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── PDF output ─────────────────────────────────────────────────────────────────

def write_pdf_transcript(
    utterances: list,
    output_path: Path,
    audio_filename: str = "",
    engine_name: str = "",
) -> None:
    """
    Write a professional clinical PDF transcript.

    Layout:
      - Branded header with title and metadata (audio file, engine, date)
      - One row per utterance: timestamp | speaker (color-coded) | text | confidence
      - Doctor rows in blue, Patient rows in green
      - Footer with page numbers

    Args:
        utterances:     List of Utterance objects.
        output_path:    Where to write the .pdf file.
        audio_filename: Source audio filename shown in the header.
        engine_name:    Transcription engine name shown in the header.
    """
    from datetime import datetime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
    )

    # ── Style definitions ──────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        wordWrap="CJK",
    )

    # ── Build content ──────────────────────────────────────────────────────────
    story = []

    # Header
    story.append(Paragraph("Medical Consultation Transcript", title_style))
    meta_parts = []
    if audio_filename:
        meta_parts.append(f"Audio: {audio_filename}")
    if engine_name:
        meta_parts.append(f"Engine: {engine_name}")
    meta_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    story.append(Paragraph("  |  ".join(meta_parts), meta_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#1a3a5c"), spaceAfter=10))
    story.append(Spacer(1, 0.3 * cm))

    # Table header row
    COL_WIDTHS = [2.2 * cm, 2.4 * cm, 10.8 * cm, 1.4 * cm]  # Time | Speaker | Text | Conf
    header_row = [
        Paragraph("<b>Time</b>",       cell_style),
        Paragraph("<b>Speaker</b>",    cell_style),
        Paragraph("<b>Transcript</b>", cell_style),
        Paragraph("<b>Conf.</b>",      cell_style),
    ]

    # Data rows
    table_data = [header_row]
    row_colors: list[tuple[int, colors.Color]] = []  # (row_index, bg_color)

    DOCTOR_BG  = colors.HexColor("#ddeeff")   # light blue
    PATIENT_BG = colors.HexColor("#ddffee")   # light green
    HEADER_BG  = colors.HexColor("#1a3a5c")

    for idx, u in enumerate(utterances, start=1):
        ts       = format_timestamp(u.start_time)
        conf_str = f"{u.confidence:.2f}" if u.confidence is not None else "-"
        is_doctor = u.speaker_label == "Doctor"

        speaker_color = colors.HexColor("#0a3d8f") if is_doctor else colors.HexColor("#0a6b40")
        speaker_para  = Paragraph(
            f'<font color="{speaker_color.hexval()}"><b>{u.speaker_label}</b></font>',
            cell_style,
        )

        row = [
            Paragraph(ts, cell_style),
            speaker_para,
            Paragraph(u.text, cell_style),
            Paragraph(conf_str, cell_style),
        ]
        table_data.append(row)
        row_colors.append((idx, DOCTOR_BG if is_doctor else PATIENT_BG))

    # Build table style
    table_style_cmds = [
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        # All rows
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUND", (0, 0), (-1, -1), [colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_idx, bg in row_colors:
        table_style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    table = Table(table_data, colWidths=COL_WIDTHS, repeatRows=1)
    table.setStyle(TableStyle(table_style_cmds))
    story.append(table)

    # Footer note
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"), spaceAfter=4))
    story.append(Paragraph(
        "CONFIDENTIAL — For clinical use only. "
        "This transcript was generated by an AI system and should be reviewed by a clinician.",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER),
    ))

    def _add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawRightString(
            doc.pagesize[0] - 2 * cm,
            1.2 * cm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    print(f"[utils] PDF transcript saved  -> {output_path}")


def write_pdf_evaluation_report(
    aggregate: dict,
    output_path: Path,
) -> None:
    """
    Write a professional PDF evaluation report.

    Contains:
      - Summary header (engine, dataset, avg WER / DER)
      - Per-case results table with WER and DER columns

    Args:
        aggregate:   The dict produced by evaluate.run_evaluation().
        output_path: Where to write the .pdf file.
    """
    from datetime import datetime
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

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2.5 * cm,
    )

    styles    = getSampleStyleSheet()
    HDR_COLOR = colors.HexColor("#1a3a5c")
    cell_s    = ParagraphStyle("c", parent=styles["Normal"], fontSize=8, leading=11)

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Medical Transcription — Evaluation Report",
        ParagraphStyle("T", parent=styles["Heading1"], fontSize=15,
                       textColor=HDR_COLOR, alignment=TA_CENTER, spaceAfter=4),
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("M", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#666666"), alignment=TA_CENTER),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=HDR_COLOR, spaceAfter=10))
    story.append(Spacer(1, 0.3 * cm))

    # ── Summary table ─────────────────────────────────────────────────────────
    avg_wer = aggregate.get("average_wer")
    avg_der = aggregate.get("average_der")
    wer_str = f"{avg_wer:.4f} ({avg_wer * 100:.1f}%)" if avg_wer is not None else "N/A"
    der_str = f"{avg_der:.4f} ({avg_der * 100:.1f}%)" if avg_der is not None else "N/A"

    summary_data = [
        [Paragraph("<b>Field</b>", cell_s),     Paragraph("<b>Value</b>", cell_s)],
        [Paragraph("Dataset", cell_s),          Paragraph(aggregate.get("dataset", "-"), cell_s)],
        [Paragraph("Engine", cell_s),           Paragraph(aggregate.get("engine", "-"), cell_s)],
        [Paragraph("Cases Evaluated", cell_s),  Paragraph(str(aggregate.get("cases_evaluated", 0)), cell_s)],
        [Paragraph("Average WER", cell_s),      Paragraph(wer_str, cell_s)],
        [Paragraph("Average DER", cell_s),      Paragraph(der_str, cell_s)],
    ]
    if aggregate.get("note"):
        summary_data.append([
            Paragraph("Note", cell_s),
            Paragraph(aggregate["note"], cell_s),
        ])

    summary_table = Table(summary_data, colWidths=[4 * cm, 12.8 * cm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), HDR_COLOR),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUND",(0, 1), (-1, -1), [colors.HexColor("#f5f8ff"), colors.white]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.6 * cm))

    # ── Per-case table ────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Per-Case Results",
        ParagraphStyle("SH", parent=styles["Heading2"], fontSize=11,
                       textColor=HDR_COLOR, spaceBefore=6, spaceAfter=4),
    ))

    case_header = [
        Paragraph("<b>Case ID</b>",    cell_s),
        Paragraph("<b>WER</b>",        cell_s),
        Paragraph("<b>DER</b>",        cell_s),
        Paragraph("<b>Status</b>",     cell_s),
    ]
    case_data = [case_header]
    row_bgs: list[tuple] = []

    for i, r in enumerate(aggregate.get("per_case", []), start=1):
        if r.get("error"):
            wer_v = "-"
            der_v = "-"
            status = Paragraph(f'<font color="red">ERROR: {r["error"][:60]}</font>', cell_s)
            bg = colors.HexColor("#fff0f0")
        else:
            wer_v  = f"{r['wer']:.3f}" if r.get("wer") is not None else "-"
            der_v  = f"{r['der']:.3f}" if r.get("der") is not None else "N/A"
            status = Paragraph('<font color="green">OK</font>', cell_s)
            bg = colors.HexColor("#f5f8ff") if i % 2 == 0 else colors.white

        case_data.append([
            Paragraph(r["case_id"][:55], cell_s),
            Paragraph(wer_v, cell_s),
            Paragraph(der_v, cell_s),
            status,
        ])
        row_bgs.append((i, bg))

    case_style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0), HDR_COLOR),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
    ]
    for row_idx, bg in row_bgs:
        case_style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    case_table = Table(
        case_data,
        colWidths=[9.5 * cm, 2 * cm, 2 * cm, 3.3 * cm],
        repeatRows=1,
    )
    case_table.setStyle(TableStyle(case_style_cmds))
    story.append(case_table)

    def _page_num(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawRightString(doc.pagesize[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_num, onLaterPages=_page_num)
    print(f"[utils] PDF evaluation report -> {output_path}")


def write_pdf_dataset_transcript_report(
    per_case_results: list[dict],
    aggregate: dict,
    output_path: Path,
) -> None:
    """
    Write a clinical transcript-style PDF for evaluated dataset cases.

    Shows for each case:
      - Medical intent / prompt category
      - Reference phrase (ground truth)
      - Transcribed text (hypothesis)
      - WER score (color-coded: green=0%, amber<30%, red>=30%)
      - Confidence score

    Args:
        per_case_results:  List of dicts with case_id, prompt, reference,
                           hypothesis, wer, der, confidence fields.
        aggregate:         Summary dict (engine, dataset, averages).
        output_path:       Where to write the .pdf file.
    """
    from datetime import datetime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph,
        Spacer, HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2.0 * cm,  bottomMargin=2.5 * cm,
    )

    styles  = getSampleStyleSheet()
    HDR     = colors.HexColor("#1a3a5c")
    cell_s  = ParagraphStyle("c",  parent=styles["Normal"], fontSize=8,  leading=11)
    label_s = ParagraphStyle("lb", parent=styles["Normal"], fontSize=7,
                              textColor=colors.HexColor("#555555"), leading=9)

    story = []

    # -- Title & metadata header ------------------------------------------------
    story.append(Paragraph(
        "Medical Speech Transcription — Dataset Review",
        ParagraphStyle("T", parent=styles["Heading1"], fontSize=15,
                       textColor=HDR, alignment=TA_CENTER, spaceAfter=3),
    ))

    avg_wer = aggregate.get("average_wer")
    wer_str = f"{avg_wer:.1%}" if avg_wer is not None else "N/A"
    engine  = aggregate.get("engine", "-")
    n_cases = len(per_case_results)

    story.append(Paragraph(
        f"Engine: {engine}   |   Cases: {n_cases}   |   "
        f"Avg WER: {wer_str}   |   "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("M", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#555555"), alignment=TA_CENTER),
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=HDR, spaceAfter=10))
    story.append(Spacer(1, 0.2 * cm))

    # -- Legend -----------------------------------------------------------------
    legend_data = [[
        Paragraph("<b>WER Color Key:</b>", cell_s),
        Paragraph('<font color="#006600"><b>0% — Perfect</b></font>', cell_s),
        Paragraph('<font color="#7a5c00"><b>1–29% — Good</b></font>', cell_s),
        Paragraph('<font color="#990000"><b>30%+ — Needs review</b></font>', cell_s),
    ]]
    legend_table = Table(legend_data, colWidths=[3*cm, 3.5*cm, 4*cm, 4*cm])
    legend_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#f0f4f8")),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 0.4 * cm))

    # -- Per-case cards ---------------------------------------------------------
    COL_W = [3.5*cm, 2.0*cm, 1.5*cm, 7.8*cm]  # Label | Value | WER | Transcript

    for idx, r in enumerate(per_case_results, 1):
        wer_val  = r.get("wer")
        wer_disp = f"{wer_val:.1%}" if wer_val is not None else "N/A"

        if wer_val is None or r.get("error"):
            row_bg  = colors.HexColor("#fff0f0")
            wer_col = colors.HexColor("#cc0000")
        elif wer_val == 0.0:
            row_bg  = colors.HexColor("#e6f9ee")
            wer_col = colors.HexColor("#006600")
        elif wer_val < 0.30:
            row_bg  = colors.HexColor("#fffbe6")
            wer_col = colors.HexColor("#7a5c00")
        else:
            row_bg  = colors.HexColor("#fff0f0")
            wer_col = colors.HexColor("#990000")

        case_id  = r.get("case_id", "")
        prompt   = r.get("prompt", "")
        ref_text = r.get("reference", "")
        hyp_text = r.get("hypothesis", "— ERROR —") if not r.get("error") else f"ERROR: {r['error']}"

        card_data = [
            # Header row: case number + ID
            [
                Paragraph(f"<b>Case {idx}</b>", cell_s),
                Paragraph(
                    f'<font color="#1a3a5c"><b>{case_id}</b></font>',
                    ParagraphStyle("id", parent=styles["Normal"], fontSize=7, leading=9),
                ),
                Paragraph(
                    f'<font color="{wer_col.hexval()}"><b>WER {wer_disp}</b></font>',
                    ParagraphStyle("w", parent=styles["Normal"], fontSize=8,
                                   leading=10, alignment=TA_CENTER),
                ),
                Paragraph(
                    f'<font color="#444444"><b>Intent:</b></font> '
                    f'<font color="#1a3a5c"><b>{prompt or "—"}</b></font>',
                    cell_s,
                ),
            ],
            # Reference row
            [
                Paragraph("<i>Reference</i>", label_s),
                Paragraph("", cell_s),
                Paragraph("", cell_s),
                Paragraph(ref_text, cell_s),
            ],
            # Hypothesis row
            [
                Paragraph("<i>Transcribed</i>", label_s),
                Paragraph("", cell_s),
                Paragraph("", cell_s),
                Paragraph(hyp_text, cell_s),
            ],
        ]

        card_style = TableStyle([
            # Header row styling
            ("BACKGROUND",    (0, 0), (-1, 0), HDR if idx % 2 == 1 else colors.HexColor("#2a4a6c")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("BACKGROUND",    (0, 1), (-1, 1), colors.HexColor("#f5f8ff")),
            ("BACKGROUND",    (0, 2), (-1, 2), row_bg),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("SPAN",          (1, 0), (1, 0)),
        ])

        card = Table(card_data, colWidths=COL_W, repeatRows=0)
        card.setStyle(card_style)
        story.append(KeepTogether([card, Spacer(1, 0.25 * cm)]))

    # -- Footer -----------------------------------------------------------------
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"), spaceAfter=3))
    story.append(Paragraph(
        "CONFIDENTIAL — For clinical review only. "
        "AI-generated transcript. Verify before clinical use.",
        ParagraphStyle("F", parent=styles["Normal"], fontSize=7,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER),
    ))

    def _page_num(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawRightString(doc.pagesize[0] - 1.8 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_num, onLaterPages=_page_num)
    print(f"[utils] Dataset transcript PDF -> {output_path}")

