"""
nlp/medical_normalization.py — Safe medical text normalization.

Applies high-confidence, meaning-preserving corrections to ASR output.

Principles:
  - ALWAYS preserves raw_text
  - ONLY applies high-confidence corrections
  - NEVER changes medical meaning
  - NEVER invents missing terms
  - NEVER diagnoses or adds clinical information
  - Normalizations are loaded from medical_vocabulary/ files

Example corrections:
  "hyper tension" → "hypertension"
  "blood presser" → "blood pressure"
  "ECG" → preserved (already correct)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


# ── Vocabulary loading ────────────────────────────────────────────────────────

_VOCAB_DIR = Path(__file__).parent.parent / "medical_vocabulary"

# Common multi-word → single-word normalizations (safe, meaning-preserving)
# Format: (regex_pattern, replacement)
_BUILTIN_PATTERNS: list[tuple[str, str]] = [
    # Common ASR errors for medical compounds
    (r"\bhyper\s+tension\b", "hypertension"),
    (r"\bhypo\s+tension\b", "hypotension"),
    (r"\bdiabetes\s+mellitus\b", "diabetes mellitus"),
    (r"\bheart\s+attack\b", "heart attack"),
    (r"\bblood\s+presser\b", "blood pressure"),
    (r"\bblood\s+pressure\b", "blood pressure"),
    (r"\bbreath\s+ing\b", "breathing"),
    (r"\bheadache\b", "headache"),
    (r"\bhead\s+ache\b", "headache"),
    (r"\bback\s+ache\b", "backache"),
    (r"\bstomach\s+ache\b", "stomach ache"),
    (r"\bshort\s+ness\s+of\s+breath\b", "shortness of breath"),
    (r"\bpalpita\s+tions\b", "palpitations"),
    (r"\bdys\s+pnea\b", "dyspnea"),
    (r"\bdys\s+pnoea\b", "dyspnoea"),
    (r"\becg\b", "ECG"),
    (r"\bekg\b", "EKG"),
    (r"\bmri\b", "MRI"),
    (r"\bct\s+scan\b", "CT scan"),
    (r"\bx\s*-\s*ray\b", "X-ray"),
    (r"\bechocardiog\s+ram\b", "echocardiogram"),
    (r"\belectrocardiog\s+ram\b", "electrocardiogram"),
    (r"\bparacet\s+amol\b", "paracetamol"),
    (r"\bibu\s+profen\b", "ibuprofen"),
    (r"\bmet\s+formin\b", "metformin"),
    (r"\bamox\s+icillin\b", "amoxicillin"),
]

# Compiled patterns cache
_COMPILED_PATTERNS: Optional[list[tuple[re.Pattern, str]]] = None


def _get_patterns() -> list[tuple[re.Pattern, str]]:
    """Load and compile normalization patterns (cached)."""
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is not None:
        return _COMPILED_PATTERNS

    patterns: list[tuple[re.Pattern, str]] = []

    # Add built-in patterns
    for pattern_str, replacement in _BUILTIN_PATTERNS:
        try:
            patterns.append((re.compile(pattern_str, re.IGNORECASE), replacement))
        except re.error:
            pass  # Skip malformed patterns

    _COMPILED_PATTERNS = patterns
    return patterns


def normalize_text(text: str) -> str:
    """
    Apply medical text normalization to a single string.

    Only applies high-confidence, meaning-preserving corrections.
    The original text is NOT modified in place — this returns a new string.

    Args:
        text: Raw ASR text.

    Returns:
        Normalized text string.
    """
    if not text or not text.strip():
        return text

    result = text
    for pattern, replacement in _get_patterns():
        result = pattern.sub(replacement, result)

    return result


def normalize_utterances(
    utterances: list["Utterance"],
) -> list["Utterance"]:
    """
    Apply medical normalization to all utterances.

    Sets normalized_text field while preserving raw_text.
    The text field is updated to normalized_text.

    Args:
        utterances: List of Utterance objects.

    Returns:
        New list of Utterance objects with normalized_text populated.
    """
    from engines.base import Utterance as Utt

    result = []
    changed = 0

    for u in utterances:
        norm = normalize_text(u.raw_text or u.text)
        if norm != (u.raw_text or u.text):
            changed += 1

        result.append(Utt(
            id=u.id,
            speaker_id=u.speaker_id,
            speaker_label=u.speaker_label,
            text=norm,               # Display text uses normalized version
            raw_text=u.raw_text or u.text,  # Always preserve original
            normalized_text=norm,
            start_time=u.start_time,
            end_time=u.end_time,
            confidence=u.confidence,
            overlap=u.overlap,
            overlapping_speakers=u.overlapping_speakers,
            words=u.words,
            review_required=u.review_required,
        ))

    if changed:
        print(f"[normalization] {changed} utterance(s) normalized.")
    return result


def load_vocabulary_terms() -> set[str]:
    """Load all medical terms from vocabulary files into a flat set."""
    terms: set[str] = set()
    if not _VOCAB_DIR.exists():
        return terms

    for fname in _VOCAB_DIR.glob("*.txt"):
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" not in line:
                    terms.add(line.lower())
    return terms
