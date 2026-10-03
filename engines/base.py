"""
engines/base.py — Core data models and abstract interfaces for all engines.

Defines the common SpeakerSegment, WordTimestamp, and abstract base classes
so that Community-1, Sortformer, and AssemblyAI all produce identical output
types that downstream code can consume engine-agnostically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Shared data models ─────────────────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    """
    One contiguous speech segment attributed to a single speaker.

    Produced by any diarization backend (pyannote Community-1, Sortformer,
    AssemblyAI) and converted into this common format before downstream
    processing (alignment, overlap detection, role mapping).
    """
    speaker_id: str               # Raw diarization label: SPEAKER_00, SPEAKER_A, etc.
    start_time: float             # Seconds from start of audio
    end_time: float               # Seconds from start of audio
    confidence: Optional[float] = None  # Segment-level diarization confidence [0,1]
    overlap: bool = False         # True if this segment overlaps with another speaker

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class WordTimestamp:
    """
    One transcribed word with precise timing and optional confidence.

    Produced by WhisperX word-level alignment or AssemblyAI word timestamps.
    """
    word: str
    start: float                  # Seconds from start of audio
    end: float                    # Seconds from start of audio
    confidence: Optional[float] = None  # ASR word-level confidence [0,1]
    speaker_id: Optional[str] = None    # Filled in during word-to-speaker alignment


@dataclass
class OverlapRegion:
    """
    A region of audio where two (or more) speakers are simultaneously active.
    """
    start_time: float
    end_time: float
    speakers: list[str] = field(default_factory=list)
    overlap: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class Utterance:
    """
    Enhanced Utterance — the primary output unit for the complete pipeline.

    Backward compatible with the original transcriber.Utterance (which had
    speaker_label, text, start_time, end_time, confidence).

    New optional fields default to backward-safe values.
    """
    # ── Backward-compatible core fields ────────────────────────────────────────
    speaker_label: str            # "Doctor" | "Patient"
    text: str                     # Transcribed text (normalized if enabled)
    start_time: float
    end_time: float
    confidence: Optional[float]

    # ── Enhanced fields ────────────────────────────────────────────────────────
    id: int = 0
    speaker_id: str = ""          # Raw diarization label (SPEAKER_00, SPEAKER_A)
    raw_text: str = ""            # Pre-normalization text (same as text if no normalization)
    normalized_text: str = ""     # Post-normalization text
    overlap: bool = False         # True if utterance overlaps with another speaker
    overlapping_speakers: list[str] = field(default_factory=list)
    words: list[WordTimestamp] = field(default_factory=list)
    review_required: bool = False  # True if confidence < LOW_CONFIDENCE_THRESHOLD

    def __post_init__(self):
        # Ensure raw_text is populated if not set
        if not self.raw_text:
            self.raw_text = self.text
        # Ensure normalized_text is populated if not set
        if not self.normalized_text:
            self.normalized_text = self.text

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def confidence_tier(
        self,
        high_threshold: float = 0.90,
        low_threshold: float = 0.70,
    ) -> str:
        """Return 'high', 'medium', or 'low' based on confidence score."""
        if self.confidence is None:
            return "unknown"
        if self.confidence >= high_threshold:
            return "high"
        if self.confidence >= low_threshold:
            return "medium"
        return "low"


# ── Abstract engine interfaces ─────────────────────────────────────────────────

class DiarizationEngine(ABC):
    """
    Abstract interface for speaker diarization backends.

    All implementations (pyannote Community-1, Sortformer, AssemblyAI)
    must produce a list[SpeakerSegment] so downstream code is engine-agnostic.
    """

    @abstractmethod
    def diarize(self, audio_path: str | Path) -> list[SpeakerSegment]:
        """
        Perform speaker diarization on the given audio file.

        Args:
            audio_path: Path to a WAV file (16 kHz mono recommended).

        Returns:
            List of SpeakerSegment objects sorted by start_time.
        """


class ASREngine(ABC):
    """
    Abstract interface for Automatic Speech Recognition backends.

    All implementations must produce a list[WordTimestamp] so the alignment
    layer is decoupled from the specific ASR model.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> list[WordTimestamp]:
        """
        Transcribe audio and return word-level timestamps.

        Args:
            audio_path: Path to a WAV file.

        Returns:
            List of WordTimestamp objects sorted by start time.
        """


class CombinedEngine(ABC):
    """
    Abstract interface for engines that perform diarization + ASR together
    (e.g. AssemblyAI, which does both in a single API call).

    Produces Utterance objects directly rather than separate SpeakerSegment
    and WordTimestamp lists.
    """

    @abstractmethod
    def transcribe_and_diarize(self, audio_path: str | Path) -> list[Utterance]:
        """
        Perform combined transcription + diarization.

        Returns Utterance objects with raw speaker labels (SPEAKER_A, etc.),
        not yet mapped to Doctor/Patient roles.
        """
