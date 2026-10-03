"""
outputs/conversation_timeline.py — Module 1 → Module 3 Canonical JSON Writer.

THIS IS THE PRIMARY OUTPUT MODULE.

Produces the canonical time-series conversation timeline JSON that
is consumed by Module 3 (EHR NLP / Ontology structuring).

The output is engine-agnostic — Community-1, Sortformer, and AssemblyAI
all produce the same schema.

Schema: schemas/conversation_timeline.json

CRITICAL CONTRACT:
  - Every utterance has: who, role, what, when, language, confidence, review flag
  - Timestamps are in HH:MM:SS.mmm format (millisecond precision)
  - Original audio timestamps are ALWAYS preserved
  - raw_text and normalized_text are both stored
  - No speaker is forced into Doctor/Patient if uncertain
  - Consent status is always recorded

Output file: output/<encounter>_conversation.json
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance
    from diarization.role_classifier import SpeakerProfile

_SCHEMA_VERSION = "1.0"
_EMITTED_TO = "module3.ingestion.conversation"


# ── Timestamp formatting ──────────────────────────────────────────────────────

def _format_ms_timestamp(seconds: float) -> str:
    """
    Convert float seconds to HH:MM:SS.mmm format (millisecond precision).

    CRITICAL: This preserves original audio timestamps exactly.
    Example: 3.12 → "00:00:03.120"
    """
    seconds = max(0.0, seconds)
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_secs = total_ms // 1000
    h, rem = divmod(total_secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ── Encounter ID ──────────────────────────────────────────────────────────────

def generate_encounter_id() -> str:
    """Generate a unique encounter ID with ENC- prefix."""
    return f"ENC-{uuid.uuid4().hex[:12].upper()}"


# ── Main writer ───────────────────────────────────────────────────────────────

def write_conversation_timeline(
    utterances: list["Utterance"],
    output_path: Path,
    *,
    encounter_id: Optional[str] = None,
    audio_ref: str = "",
    capture_mode: str = "batch",
    consent_logged: bool = False,
    engine_info: Optional[dict] = None,
    audio_metadata: Optional[dict] = None,
    speaker_profiles: Optional[list["SpeakerProfile"]] = None,
    language_tags: Optional[list[dict]] = None,
    languages_detected: Optional[list[str]] = None,
    quality_report=None,
    speaker_stats=None,
    clinical_summary=None,
    audit_corrections: Optional[list[dict]] = None,
    validate: bool = True,
) -> dict:
    """
    Write the canonical Module 1 → Module 3 conversation timeline JSON.

    Args:
        utterances:         Role-labeled Utterance objects.
        output_path:        Where to write the .json file.
        encounter_id:       Unique encounter ID. Auto-generated if None.
        audio_ref:          Reference to source audio file.
        capture_mode:       "batch" or "live".
        consent_logged:     Whether consent was obtained.
        engine_info:        Dict describing engine used.
        audio_metadata:     Dict with duration, sample_rate, etc.
        speaker_profiles:   SpeakerProfile list with role assignments.
        language_tags:      Per-utterance language detection results.
        languages_detected: Unique languages detected (list of codes).
        quality_report:     QualityReport from analytics.quality_metrics.
        speaker_stats:      ConsultationStats from analytics.speaker_stats.
        clinical_summary:   ClinicalSummary from nlp.clinical_summary.
        audit_corrections:  List of correction audit records.
        validate:           Whether to validate against JSON schema.

    Returns:
        The generated document dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enc_id = encounter_id or generate_encounter_id()
    now = datetime.now(timezone.utc).isoformat()

    # ── Build language tag lookup ─────────────────────────────────────────────
    lang_by_id: dict[int, dict] = {}
    if language_tags:
        for tag in language_tags:
            lang_by_id[tag["utterance_id"]] = tag

    # ── Build speaker profiles lookup ─────────────────────────────────────────
    if speaker_profiles:
        from diarization.role_classifier import profiles_to_dict
        speakers_list = profiles_to_dict(speaker_profiles)
        profile_map = {p["speaker_id"]: p for p in speakers_list}
    else:
        # Derive minimal speaker info from utterances
        speakers_list = []
        seen: set[str] = set()
        for u in utterances:
            if u.speaker_id not in seen:
                seen.add(u.speaker_id)
                speakers_list.append({
                    "speaker_id": u.speaker_id,
                    "role": u.speaker_label.lower().replace(" ", "_"),
                    "role_confidence": 0.70,
                    "display_label": u.speaker_label,
                    "assignment_method": "auto",
                })
        profile_map = {p["speaker_id"]: p for p in speakers_list}

    # ── Build conversation timeline ───────────────────────────────────────────
    timeline = []
    for seq, u in enumerate(
        sorted(utterances, key=lambda x: (x.start_time, x.id)), start=1
    ):
        lang_tag = lang_by_id.get(u.id, {})
        lang = lang_tag.get("language", "en")
        profile = profile_map.get(u.speaker_id, {})

        entry = {
            "utterance_id": f"u{seq}",
            "seq": seq,
            "start_time": _format_ms_timestamp(u.start_time),
            "end_time": _format_ms_timestamp(u.end_time),
            "speaker_id": u.speaker_id,
            "role": profile.get("display_label", u.speaker_label),
            "language": lang,
            "text": u.normalized_text or u.text,
            "raw_text": u.raw_text or u.text,
            "confidence": round(u.confidence, 4) if u.confidence is not None else None,
            "needs_review": u.review_required,
            "overlap": u.overlap,
            "overlapping_speakers": u.overlapping_speakers or [],
        }
        timeline.append(entry)

    # ── Build quality summary ─────────────────────────────────────────────────
    if quality_report and hasattr(quality_report, "to_dict"):
        quality_dict = quality_report.to_dict()
    else:
        confs = [u.confidence for u in utterances if u.confidence is not None]
        quality_dict = {
            "total_utterances": len(utterances),
            "average_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "low_confidence_count": sum(1 for c in confs if c < 0.75),
            "overlap_count": sum(1 for u in utterances if u.overlap),
            "review_required_count": sum(1 for u in utterances if u.review_required),
        }

    # Add language distribution to quality
    if timeline:
        lang_dist: dict[str, int] = {}
        for entry in timeline:
            lg = entry["language"]
            lang_dist[lg] = lang_dist.get(lg, 0) + 1
        quality_dict["language_distribution"] = lang_dist

    # ── Build analytics ───────────────────────────────────────────────────────
    if speaker_stats and hasattr(speaker_stats, "to_dict"):
        analytics_dict = speaker_stats.to_dict()
    else:
        analytics_dict = {}

    # ── Build clinical summary ────────────────────────────────────────────────
    if clinical_summary is not None:
        if hasattr(clinical_summary, "to_dict"):
            clinical_dict = clinical_summary.to_dict()
        elif isinstance(clinical_summary, dict):
            clinical_dict = clinical_summary
        else:
            clinical_dict = None
    else:
        clinical_dict = None

    # ── Assemble full document ────────────────────────────────────────────────
    doc = {
        "encounter_id": enc_id,
        "schema_version": _SCHEMA_VERSION,
        "capture_mode": capture_mode,
        "audio_ref": audio_ref,
        "consent_logged": consent_logged,
        "generated_at": now,
        "engine": engine_info or {},
        "audio_metadata": audio_metadata or {},
        "languages_detected": languages_detected or sorted(set(e["language"] for e in timeline)),
        "speakers": speakers_list,
        "conversation_timeline": timeline,
        "quality": quality_dict,
        "analytics": analytics_dict,
        "clinical_summary": clinical_dict,
        "audit": {
            "corrections": audit_corrections or [],
        },
        "emitted_to": _EMITTED_TO,
        "emitted_at": now,
    }

    # ── Schema validation ─────────────────────────────────────────────────────
    if validate:
        errors = validate_timeline(doc)
        if errors:
            print(f"[timeline] Schema validation issues [WARN] ({len(errors)}):")
            for err in errors[:5]:
                print(f"  - {err}")
        else:
            print("[timeline] Schema validation passed")

    # ── Write ─────────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"[timeline] Conversation timeline -> {output_path}")
    print(f"           Encounter: {enc_id}")
    print(f"           Utterances: {len(timeline)} | Languages: {doc['languages_detected']}")

    return doc


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_timeline(doc: dict) -> list[str]:
    """
    Validate a conversation timeline document against the schema.

    Uses jsonschema if available, otherwise performs basic structural checks.

    Args:
        doc: Document dict to validate.

    Returns:
        List of validation error strings. Empty list = valid.
    """
    errors: list[str] = []

    # Try jsonschema first
    try:
        import jsonschema  # type: ignore
        schema_path = Path(__file__).parent.parent / "schemas" / "conversation_timeline.json"
        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            validator = jsonschema.Draft7Validator(schema)
            for err in validator.iter_errors(doc):
                errors.append(f"{err.json_path}: {err.message}")
            return errors
    except ImportError:
        pass  # jsonschema not installed — use basic checks

    # Basic structural validation
    required_top = [
        "encounter_id", "schema_version", "capture_mode", "audio_ref",
        "consent_logged", "generated_at", "speakers",
        "conversation_timeline", "emitted_to"
    ]
    for key in required_top:
        if key not in doc:
            errors.append(f"Missing required field: '{key}'")

    if not doc.get("encounter_id", "").startswith("ENC-"):
        errors.append("encounter_id must start with 'ENC-'")

    for i, entry in enumerate(doc.get("conversation_timeline", [])):
        for field in ["utterance_id", "seq", "start_time", "end_time",
                      "speaker_id", "role", "language", "text", "needs_review"]:
            if field not in entry:
                errors.append(f"conversation_timeline[{i}] missing field: '{field}'")

    return errors
