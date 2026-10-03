"""
outputs/txt_writer.py — Enhanced plain-text transcript output.

Generates a professional consultation transcript with:
  - Formatted header (case ID, engine, speakers summary)
  - Timestamped dialogue with speaker roles
  - Quality footer (confidence, overlaps, WER/DER if available)

Also provides backward-compatible write_txt_transcript().
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance

_SEP = "=" * 60
_SUB = "-" * 60


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS."""
    seconds = max(0.0, seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_enhanced_txt(
    utterances: list["Utterance"],
    output_path: Path,
    metadata: Optional[dict] = None,
    speaker_stats=None,
    quality_report=None,
) -> None:
    """
    Write an enhanced plain-text transcript.

    Format:
        ============================================================
        MEDICAL CONSULTATION TRANSCRIPT
        ============================================================
        Case ID: ...
        Engine: ...
        ...

        SPEAKERS
        Doctor:  speaking time | turns
        Patient: speaking time | turns

        ------------------------------------------------------------
        [HH:MM:SS] DOCTOR
        How are you feeling today?
        [Confidence: 97%]

        [HH:MM:SS] PATIENT  ⚠ Review Recommended
        I've had a headache for three days.
        [Confidence: 94%]

        ------------------------------------------------------------
        QUALITY
        Average confidence: ...
        ...
        ============================================================
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata or {}
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        _SEP,
        "MEDICAL CONSULTATION TRANSCRIPT",
        _SEP,
        "",
        f"Case ID    : {meta.get('case_id', 'N/A')}",
        f"Audio File : {meta.get('audio_file', 'N/A')}",
        f"Duration   : {_format_duration(meta.get('duration_seconds'))}",
        f"Engine     : {meta.get('engine', 'N/A')}",
        f"Diarization: {meta.get('diarization_model', 'N/A')}",
        f"ASR Model  : {meta.get('asr_model', 'N/A')}",
        f"Language   : {meta.get('language', 'en')}",
        f"Generated  : {meta.get('created_at', 'N/A')}",
        "",
    ]

    # ── Speakers summary ──────────────────────────────────────────────────────
    lines.append("SPEAKERS")
    lines.append(_SUB)
    if speaker_stats and hasattr(speaker_stats, "speakers"):
        for s in speaker_stats.speakers:
            lines.append(
                f"  {s.role:<10}: {s.total_speech_seconds:.1f}s "
                f"({s.speech_percentage:.1f}%) | {s.turn_count} turn(s)"
            )
    else:
        by_role: dict[str, dict] = {}
        for u in utterances:
            r = u.speaker_label
            if r not in by_role:
                by_role[r] = {"seconds": 0.0, "turns": 0}
            by_role[r]["seconds"] += max(0.0, u.end_time - u.start_time)
            by_role[r]["turns"] += 1
        for role, data in by_role.items():
            lines.append(f"  {role:<10}: {data['seconds']:.1f}s | {data['turns']} turn(s)")
    lines += ["", _SUB, ""]

    # ── Transcript ────────────────────────────────────────────────────────────
    for u in utterances:
        ts = format_timestamp(u.start_time)
        role = u.speaker_label.upper()
        review_flag = "  ⚠ Review Recommended" if u.review_required else ""
        overlap_flag = "  [OVERLAP]" if u.overlap else ""

        lines.append(f"[{ts}] {role}{review_flag}{overlap_flag}")
        lines.append(u.text)
        if u.confidence is not None:
            lines.append(f"[Confidence: {u.confidence * 100:.0f}%]")
        lines.append("")

    # ── Quality footer ────────────────────────────────────────────────────────
    lines += [_SUB, "QUALITY", _SUB]
    if quality_report and hasattr(quality_report, "to_dict"):
        qd = quality_report.to_dict()
        avg_c = qd.get("average_confidence")
        lines.append(f"  Average confidence    : {avg_c * 100:.1f}%" if avg_c else "  Average confidence    : N/A")
        lines.append(f"  Low-confidence segs   : {qd.get('low_confidence_count', 0)}")
        lines.append(f"  Review required       : {qd.get('review_required_count', 0)}")
        lines.append(f"  Overlap segments      : {qd.get('overlap_count', 0)}")
    if meta.get("wer") is not None:
        lines.append(f"  WER                   : {meta['wer'] * 100:.1f}%")
    if meta.get("der") is not None:
        lines.append(f"  DER                   : {meta['der'] * 100:.1f}%")
    else:
        lines.append("  DER                   : N/A")
    lines += [
        "",
        _SEP,
        "CONFIDENTIAL — AI-GENERATED TRANSCRIPT",
        "FOR CLINICAL REVIEW ONLY",
        _SEP,
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[txt_writer] Enhanced TXT → {output_path}")


def _format_duration(seconds) -> str:
    if seconds is None:
        return "N/A"
    return format_timestamp(float(seconds))


# ── Backward-compatible writer ────────────────────────────────────────────────

def write_txt_transcript(utterances: list["Utterance"], output_path: Path) -> None:
    """
    Write a simple plain-text transcript (backward-compatible format).

    Format: [HH:MM:SS] Doctor: How are you feeling today?
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{format_timestamp(u.start_time)}] {u.speaker_label}: {u.text}"
        for u in utterances
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[txt_writer] TXT transcript → {output_path}")
