"""
outputs/json_writer.py — Enhanced structured JSON transcript output.

Produces a rich JSON document with:
  - Metadata (case_id, engine, model, duration, timestamp)
  - Speaker summary (role, speaking time, turn count)
  - Quality metrics (confidence, overlaps, review flags)
  - Full utterances array with all fields
  - Optional clinical summary

Also provides backward-compatible write_json_transcript() that generates
the same simple array format as the original utils.py function.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


def write_enhanced_json(
    utterances: list["Utterance"],
    output_path: Path,
    metadata: Optional[dict] = None,
    speaker_stats=None,
    quality_report=None,
    clinical_summary=None,
) -> None:
    """
    Write an enhanced structured JSON transcript.

    Format:
    {
        "metadata": { ... },
        "speakers": [ ... ],
        "quality": { ... },
        "utterances": [ ... ],
        "clinical_summary": { ... }   (optional)
    }

    Args:
        utterances:       Role-labeled list of Utterance objects.
        output_path:      Where to write the .json file.
        metadata:         Optional dict with case_id, engine, etc.
        speaker_stats:    ConsultationStats from analytics.speaker_stats.
        quality_report:   QualityReport from analytics.quality_metrics.
        clinical_summary: ClinicalSummary from nlp.clinical_summary (optional).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict = {}

    # ── Metadata ──────────────────────────────────────────────────────────────
    doc["metadata"] = {
        "case_id": (metadata or {}).get("case_id", "unknown"),
        "audio_file": (metadata or {}).get("audio_file", "unknown"),
        "duration_seconds": (metadata or {}).get("duration_seconds"),
        "engine": (metadata or {}).get("engine", "unknown"),
        "diarization_model": (metadata or {}).get("diarization_model", "unknown"),
        "asr_model": (metadata or {}).get("asr_model", "unknown"),
        "language": (metadata or {}).get("language", "en"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Speaker summary ───────────────────────────────────────────────────────
    if speaker_stats and hasattr(speaker_stats, "to_dict"):
        doc["speaker_stats"] = speaker_stats.to_dict()
        doc["speakers"] = speaker_stats.to_dict().get("speakers", [])
    else:
        # Compute minimal speaker summary from utterances
        by_role: dict[str, dict] = {}
        for u in utterances:
            r = u.speaker_label
            if r not in by_role:
                by_role[r] = {"id": u.speaker_id, "role": r, "turns": 0, "seconds": 0.0}
            by_role[r]["turns"] += 1
            by_role[r]["seconds"] += max(0.0, u.end_time - u.start_time)
        total_secs = sum(v["seconds"] for v in by_role.values())
        doc["speakers"] = [
            {
                "id": v["id"],
                "role": v["role"],
                "total_speech_seconds": round(v["seconds"], 2),
                "speech_percentage": round(v["seconds"] / total_secs * 100, 1) if total_secs else 0.0,
                "turn_count": v["turns"],
            }
            for v in by_role.values()
        ]

    # ── Quality ───────────────────────────────────────────────────────────────
    if quality_report and hasattr(quality_report, "to_dict"):
        doc["quality"] = quality_report.to_dict()
    else:
        # Minimal quality from utterances
        confs = [u.confidence for u in utterances if u.confidence is not None]
        doc["quality"] = {
            "average_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "low_confidence_segments": sum(1 for c in confs if c < 0.70),
            "overlap_segments": sum(1 for u in utterances if u.overlap),
        }

    # ── Utterances ────────────────────────────────────────────────────────────
    doc["utterances"] = []
    for u in utterances:
        utt_dict = {
            "id": u.id,
            "speaker": u.speaker_label,
            "speaker_id": u.speaker_id,
            "raw_text": u.raw_text,
            "normalized_text": u.normalized_text,
            "start_time": round(u.start_time, 3),
            "end_time": round(u.end_time, 3),
            "confidence": round(u.confidence, 4) if u.confidence is not None else None,
            "overlap": u.overlap,
            "overlapping_speakers": u.overlapping_speakers,
            "review_required": u.review_required,
        }
        doc["utterances"].append(utt_dict)

    # ── Clinical summary ──────────────────────────────────────────────────────
    if clinical_summary is not None:
        if hasattr(clinical_summary, "to_dict"):
            doc["clinical_summary"] = clinical_summary.to_dict()
        elif isinstance(clinical_summary, dict):
            doc["clinical_summary"] = clinical_summary

    # ── Write ─────────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"[json_writer] Enhanced JSON → {output_path}")


# ── Backward-compatible simple writer ────────────────────────────────────────

def write_json_transcript(utterances: list["Utterance"], output_path: Path) -> None:
    """
    Write a simple JSON array transcript (backward-compatible format).

    Output format:
    [
      {
        "speaker_label": "Doctor",
        "text": "...",
        "start_time": 12.34,
        "end_time": 14.56,
        "confidence": 0.97
      }
    ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "speaker_label": u.speaker_label,
            "text": u.text,
            "start_time": round(u.start_time, 3),
            "end_time": round(u.end_time, 3),
            "confidence": u.confidence,
        }
        for u in utterances
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[json_writer] JSON transcript → {output_path}")
