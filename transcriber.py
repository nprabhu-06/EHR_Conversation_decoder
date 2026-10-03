"""
transcriber.py — Engine factory and backward-compatibility bridge.

This file provides:

  1. Legacy Utterance dataclass — kept for backward compatibility with
     evaluate.py and older code. New code should use engines.base.Utterance.

  2. Legacy TranscriptionEngine ABC — kept for evaluate.py compatibility.

  3. create_engine() factory — updated to support all engines:
       - community-whisperx (local, pyannote + whisperx)
       - pyannote-whisperx  (alias for community-whisperx)
       - assemblyai         (cloud)
       - sortformer-whisper (experimental CUDA)

  All new engines (from engines/) return engines.base.Utterance which is
  compatible with the legacy Utterance interface for display/evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Legacy Utterance (backward-compatible) ─────────────────────────────────────

@dataclass
class Utterance:
    """
    Legacy Utterance dataclass — backward-compatible stub.

    New code should use engines.base.Utterance.
    evaluate.py uses this class for reference utterances from the test dataset.
    """
    speaker_label: str          # "Doctor" | "Patient" | raw label
    text: str                   # Transcribed speech
    start_time: float           # Segment start in seconds
    end_time: float             # Segment end in seconds
    confidence: Optional[float] = None  # ASR confidence [0.0, 1.0]


# ── Legacy TranscriptionEngine ABC ─────────────────────────────────────────────

class TranscriptionEngine(ABC):
    """
    Legacy abstract base class — preserved for evaluate.py compatibility.
    New engines inherit from engines.base.CombinedEngine or DiarizationEngine.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> list:
        """Process audio and return utterances."""

    def map_speakers_to_roles(
        self,
        utterances: list,
        doctor_speaker: Optional[str] = None,
    ) -> list:
        """
        Assign Doctor/Patient roles. Delegates to diarization.speaker_mapping.
        Preserved for evaluate.py compatibility.
        """
        from diarization.speaker_mapping import apply_role_mapping
        strategy = "manual" if doctor_speaker else "first_speaker"
        return apply_role_mapping(utterances, strategy=strategy, doctor_speaker=doctor_speaker)


# ── Engine factory ─────────────────────────────────────────────────────────────

def create_engine(engine_name: str, config) -> object:
    """
    Instantiate and return the appropriate engine.

    Supported engine names:
        community-whisperx  — Local: pyannote.audio 3.1 + WhisperX
        pyannote-whisperx   — Alias for community-whisperx
        assemblyai          — Cloud: AssemblyAI API
        sortformer-whisper  — Experimental: NVIDIA NeMo Sortformer + WhisperX

    Args:
        engine_name: Engine identifier string.
        config:      Config dataclass from config.py.

    Returns:
        Engine instance with .transcribe() and .map_speakers_to_roles() methods.

    Raises:
        ValueError: For unknown engine names.
        RuntimeError: If required dependencies are missing.
    """
    # Normalize alias
    if engine_name == "pyannote-whisperx":
        engine_name = "community-whisperx"

    if engine_name == "assemblyai":
        from engines.assemblyai_engine import AssemblyAIEngineV2
        return AssemblyAIEngineV2(config)

    if engine_name == "community-whisperx":
        from engines.community_whisperx import CommunityWhisperXEngine
        return CommunityWhisperXEngine(config)

    if engine_name == "sortformer-whisper":
        from engines.sortformer_engine import SortformerEngine
        return SortformerEngine(config)

    raise ValueError(
        f"Unknown engine: {engine_name!r}.\n"
        "Valid options: assemblyai | community-whisperx | sortformer-whisper"
    )
