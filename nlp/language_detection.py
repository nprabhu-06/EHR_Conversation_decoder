"""
nlp/language_detection.py — Per-utterance language identification.

Detects the language of each utterance with support for:
  - English (en)
  - Hindi (hi)
  - Telugu (te)
  - Mixed / code-switching

CRITICAL RULES:
  - Do NOT fabricate language labels.
  - If detection is uncertain, use "unknown" or "mixed".
  - Language tags are informational — they do NOT alter ASR output.
  - Detects based on script/character patterns and heuristics when
    a dedicated library is not available.

Configuration:
    ENABLE_LANGUAGE_DETECTION=true

Optional dependency (for accurate ML-based detection):
    langdetect (pip install langdetect)
    OR
    langid   (pip install langid)

Without these, the module falls back to script-based heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


# ── Availability probes ───────────────────────────────────────────────────────

_LANGDETECT_AVAILABLE = False
_LANGID_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException  # type: ignore
    _LANGDETECT_AVAILABLE = True
except ImportError:
    pass

if not _LANGDETECT_AVAILABLE:
    try:
        import langid  # type: ignore
        _LANGID_AVAILABLE = True
    except ImportError:
        pass


# ── Script / character patterns for heuristic detection ──────────────────────

# Devanagari Unicode range (covers Hindi, Marathi, Sanskrit, etc.)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Telugu Unicode range
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")

# Basic Latin (ASCII letters — covers English and transliterated Indian languages)
_LATIN_RE = re.compile(r"[a-zA-Z]")

# Common Telugu transliteration words (Romanized Telugu)
_TELUGU_LATIN_KEYWORDS = re.compile(
    r"\b(naku|meeru|mee|undi|ledu|ayindi|oka|inka|konni|rojulu|"
    r"vachindi|pothundi|cheyandi|antunnaru|antundi|ante|kadhu|"
    r"ela|evaru|enduku|chudandi|cheppindi|telusaa|telusu|"
    r"ekkuva|takkuva|bayalu|lo|ki|tho|nu|ni|ni|ga|gaa)\b",
    re.IGNORECASE,
)

# Common Hindi transliteration words (Romanized Hindi)
_HINDI_LATIN_KEYWORDS = re.compile(
    r"\b(mujhe|aapko|kya|hai|tha|thi|the|nahi|bahut|thoda|"
    r"dard|bukhar|darr|pehle|baad|theek|abhi|phir|lekin|aur|"
    r"bhi|haan|ji|sir|doctor|madam|hoga|hain|hoti|hota|"
    r"zyada|kam|roz|raat|din|subah|shaam)\b",
    re.IGNORECASE,
)


# ── Language detection result ─────────────────────────────────────────────────

@dataclass
class LanguageDetectionResult:
    """Language detection result for a single utterance."""
    language: str           # ISO 639-1 code: en, hi, te, mixed, unknown
    confidence: float       # Detection confidence [0.0, 1.0]
    method: str             # Detection method used


def detect_language(text: str) -> LanguageDetectionResult:
    """
    Detect the language of a single text string.

    Detection order:
      1. Script-based heuristics (non-Latin scripts — most reliable)
      2. ML-based detection (langdetect / langid) if available
      3. Transliterated keyword heuristics (Romanized Indian languages)
      4. Default to "en" with low confidence for plain Latin text

    Args:
        text: Utterance text to classify.

    Returns:
        LanguageDetectionResult with language code and confidence.
    """
    if not text or not text.strip():
        return LanguageDetectionResult(language="unknown", confidence=0.0, method="empty")

    text_stripped = text.strip()

    # ── Step 1: Script detection (most reliable for native scripts) ───────────
    has_devanagari = bool(_DEVANAGARI_RE.search(text_stripped))
    has_telugu = bool(_TELUGU_RE.search(text_stripped))
    has_latin = bool(_LATIN_RE.search(text_stripped))

    # Pure Devanagari → Hindi (most common)
    if has_devanagari and not has_telugu and not has_latin:
        return LanguageDetectionResult(language="hi", confidence=0.95, method="script")

    # Pure Telugu script
    if has_telugu and not has_devanagari and not has_latin:
        return LanguageDetectionResult(language="te", confidence=0.95, method="script")

    # Mixed scripts → code-switching
    if (has_devanagari or has_telugu) and has_latin:
        return LanguageDetectionResult(language="mixed", confidence=0.90, method="script")

    # ── Step 2: ML-based detection for Latin-script text ─────────────────────
    if _LANGDETECT_AVAILABLE and len(text_stripped.split()) >= 3:
        try:
            lang = detect(text_stripped)
            # Map to our supported codes
            lang_map = {
                "hi": "hi",
                "te": "te",
                "en": "en",
                "mr": "hi",   # Marathi in Devanagari — treat as Hindi-family
            }
            mapped = lang_map.get(lang, "en")
            return LanguageDetectionResult(language=mapped, confidence=0.80, method="langdetect")
        except Exception:
            pass

    if _LANGID_AVAILABLE and len(text_stripped.split()) >= 3:
        try:
            lang, score = langid.classify(text_stripped)
            lang_map = {"hi": "hi", "te": "te", "en": "en"}
            mapped = lang_map.get(lang, "en")
            return LanguageDetectionResult(
                language=mapped,
                confidence=min(1.0, abs(float(score)) / 10.0),
                method="langid",
            )
        except Exception:
            pass

    # ── Step 3: Transliteration keyword heuristics ────────────────────────────
    te_matches = len(_TELUGU_LATIN_KEYWORDS.findall(text_stripped))
    hi_matches = len(_HINDI_LATIN_KEYWORDS.findall(text_stripped))
    word_count = max(1, len(text_stripped.split()))

    te_ratio = te_matches / word_count
    hi_ratio = hi_matches / word_count

    if te_ratio > 0.15 and te_ratio > hi_ratio:
        lang = "te" if te_ratio > 0.30 else "mixed"
        return LanguageDetectionResult(
            language=lang, confidence=min(0.75, te_ratio * 2), method="heuristic"
        )

    if hi_ratio > 0.15:
        lang = "hi" if hi_ratio > 0.30 else "mixed"
        return LanguageDetectionResult(
            language=lang, confidence=min(0.75, hi_ratio * 2), method="heuristic"
        )

    # ── Step 4: Default to English ────────────────────────────────────────────
    return LanguageDetectionResult(language="en", confidence=0.70, method="default")


def tag_utterances_with_language(
    utterances: list["Utterance"],
) -> tuple[list[dict], list[str]]:
    """
    Detect and tag the language of each utterance.

    Args:
        utterances: List of Utterance objects.

    Returns:
        Tuple of:
          - List of dicts: {utterance_id, language, lang_confidence, method}
          - List of unique languages detected across all utterances
    """
    tags: list[dict] = []
    lang_set: set[str] = set()

    for u in utterances:
        result = detect_language(u.text)
        tags.append({
            "utterance_id": u.id,
            "language": result.language,
            "lang_confidence": round(result.confidence, 3),
            "lang_method": result.method,
        })
        if result.language not in ("unknown",):
            lang_set.add(result.language)

    unique_langs = sorted(lang_set)
    if tags:
        lang_dist = {}
        for t in tags:
            lang_dist[t["language"]] = lang_dist.get(t["language"], 0) + 1
        print(f"[language_detection] Language distribution: {lang_dist}")

    return tags, unique_langs
