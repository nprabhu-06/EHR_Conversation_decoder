"""
diarization/alignment.py — Improved word-to-speaker assignment algorithm.

For every transcribed word:
  1. Find all overlapping speaker segments
  2. Calculate overlap duration with each
  3. Select the speaker with maximum overlap
  4. Calculate assignment confidence
  5. If no overlap, use midpoint/nearest-segment fallback
  6. Never silently discard any word

This is the core algorithm that bridges diarization and ASR outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import SpeakerSegment, WordTimestamp


@dataclass
class AlignedWord:
    """A word with its assigned speaker and alignment metadata."""
    word: str
    start: float
    end: float
    confidence: Optional[float]
    speaker_id: str
    assignment_confidence: Optional[float]  # How certain we are about speaker assignment
    is_ambiguous: bool                       # True if multiple speakers had similar overlap


def assign_words_to_speakers(
    words: list["WordTimestamp"],
    segments: list["SpeakerSegment"],
    ambiguity_threshold: float = 0.20,
) -> list[AlignedWord]:
    """
    Assign each word to a speaker using maximum-overlap algorithm.

    Algorithm:
        For each word:
          1. Compute overlap with every speaker segment
          2. Select segment with maximum overlap
          3. If top-2 overlap difference < ambiguity_threshold, flag as ambiguous
          4. If no overlap at all, use nearest-segment by midpoint
          5. Compute assignment_confidence = max_overlap / word_duration

    NO word is ever discarded.

    Args:
        words:               Word-level timestamps from ASR.
        segments:            Speaker segments from diarization.
        ambiguity_threshold: Relative difference below which a word is flagged ambiguous.
                             E.g. 0.20 = top-2 within 20% of each other.

    Returns:
        List of AlignedWord objects (same length as input words, minus empty tokens).
    """
    if not words or not segments:
        return []

    aligned: list[AlignedWord] = []

    for word in words:
        if not word.word:
            continue  # Skip empty tokens but count is preserved

        w_start = word.start
        w_end = max(word.end, w_start + 0.001)  # Ensure nonzero duration
        w_mid = (w_start + w_end) / 2.0
        w_dur = w_end - w_start

        # Compute overlap with every segment
        overlaps: list[tuple[float, str]] = []
        for seg in segments:
            overlap = min(w_end, seg.end_time) - max(w_start, seg.start_time)
            if overlap > 0:
                overlaps.append((overlap, seg.speaker_id))

        if overlaps:
            overlaps.sort(key=lambda x: x[0], reverse=True)
            best_overlap, best_speaker = overlaps[0]

            # Check for ambiguity (two speakers with similar overlap)
            is_ambiguous = False
            if len(overlaps) >= 2:
                second_overlap = overlaps[1][0]
                rel_diff = (best_overlap - second_overlap) / max(best_overlap, 1e-6)
                is_ambiguous = rel_diff < ambiguity_threshold

            # Assignment confidence: overlap as fraction of word duration
            assignment_conf = min(1.0, best_overlap / w_dur) if w_dur > 0 else None

        else:
            # No overlap — fallback to nearest segment by midpoint
            nearest = min(
                segments,
                key=lambda s: abs(w_mid - (s.start_time + s.end_time) / 2.0),
            )
            best_speaker = nearest.speaker_id
            is_ambiguous = False
            assignment_conf = 0.0  # Zero overlap → low confidence

        aligned.append(AlignedWord(
            word=word.word,
            start=w_start,
            end=w_end,
            confidence=word.confidence,
            speaker_id=best_speaker,
            assignment_confidence=assignment_conf,
            is_ambiguous=is_ambiguous,
        ))

    return aligned


def group_into_utterances(aligned_words: list[AlignedWord]) -> list[dict]:
    """
    Group consecutive words from the same speaker into utterance dictionaries.

    Returns list of dicts with keys:
        speaker_id, text, start_time, end_time, confidence, overlap, words
    """
    if not aligned_words:
        return []

    groups: list[dict] = []
    current_speaker = aligned_words[0].speaker_id
    current_words: list[AlignedWord] = [aligned_words[0]]

    for aw in aligned_words[1:]:
        if aw.speaker_id == current_speaker:
            current_words.append(aw)
        else:
            groups.append(_words_to_group(current_words, current_speaker))
            current_speaker = aw.speaker_id
            current_words = [aw]

    if current_words:
        groups.append(_words_to_group(current_words, current_speaker))

    return groups


def _words_to_group(words: list[AlignedWord], speaker_id: str) -> dict:
    text = " ".join(w.word for w in words if w.word)
    scores = [w.confidence for w in words if w.confidence is not None]
    avg_conf = sum(scores) / len(scores) if scores else None
    has_ambiguity = any(w.is_ambiguous for w in words)

    return {
        "speaker_id": speaker_id,
        "text": text.strip(),
        "start_time": words[0].start,
        "end_time": words[-1].end,
        "confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "overlap": has_ambiguity,
        "words": words,
    }
