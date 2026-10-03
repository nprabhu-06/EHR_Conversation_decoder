"""
engines/sortformer_engine.py — Optional experimental Sortformer diarization engine.

Uses NVIDIA NeMo Sortformer for end-to-end speaker diarization,
combined with WhisperX for transcription.

IMPORTANT: This engine is OPTIONAL and EXPERIMENTAL.
- Requires nemo_toolkit[asr] and CUDA
- Will NOT be installed by default
- The application continues to function without it
- Shows a clear message when unavailable

Install with:
    pip install -r requirements-sortformer.txt

Usage:
    python main.py --audio consultation.mp3 --engine sortformer-whisper
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from engines.base import SpeakerSegment, Utterance, DiarizationEngine

# ── Availability check ────────────────────────────────────────────────────────

_SORTFORMER_AVAILABLE = False
_SORTFORMER_MISSING_REASON = ""

try:
    import nemo  # type: ignore
    _NEMO_VERSION = getattr(nemo, "__version__", "unknown")
    _SORTFORMER_AVAILABLE = True
except ImportError:
    _SORTFORMER_MISSING_REASON = (
        "nemo_toolkit not installed. "
        "Run: pip install -r requirements-sortformer.txt"
    )

try:
    import torch  # type: ignore
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False

# ── Availability helper ───────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if Sortformer dependencies are installed."""
    return _SORTFORMER_AVAILABLE


def get_unavailable_reason() -> str:
    """Return a human-readable reason why Sortformer is unavailable."""
    return _SORTFORMER_MISSING_REASON or "Unknown reason"


# ── Sortformer diarization ────────────────────────────────────────────────────

class SortformerDiarizer(DiarizationEngine):
    """
    NVIDIA NeMo Sortformer diarization backend.

    Sortformer is an end-to-end neural diarization model that predicts
    speaker activity directly without separate VAD + clustering steps.

    Requires: nemo_toolkit[asr] >= 1.23.0 and CUDA.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if not _SORTFORMER_AVAILABLE:
            raise RuntimeError(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  Sortformer engine: dependencies not installed               ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n\n"
                f"Reason: {_SORTFORMER_MISSING_REASON}\n\n"
                "To install Sortformer dependencies:\n"
                "  pip install -r requirements-sortformer.txt\n\n"
                "NOTE: Sortformer requires CUDA (GPU). CPU inference is not supported.\n"
                "      Use --engine community-whisperx or --engine assemblyai instead.\n"
            )

        if not _CUDA_AVAILABLE:
            raise RuntimeError(
                "\n[ERROR] Sortformer requires CUDA but no GPU is available.\n"
                "Use --engine community-whisperx (CPU supported) or "
                "--engine assemblyai (cloud) instead.\n"
            )

        self._device = device or "cuda"
        self._model = None
        print(
            f"[SortformerDiarizer] Initialized (EXPERIMENTAL) | "
            f"NeMo: {_NEMO_VERSION} | Device: {self._device}"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        print("[SortformerDiarizer] Loading Sortformer model …")
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel  # type: ignore
            self._model = SortformerEncLabelModel.from_pretrained(
                model_name="diar_sortformer_4spk-v1"
            )
            self._model.eval()
            print("[SortformerDiarizer] Model loaded.")
        except Exception as e:
            raise RuntimeError(
                f"[SortformerDiarizer] Failed to load model: {e}\n"
                "Check that nemo_toolkit is correctly installed and "
                "model 'diar_sortformer_4spk-v1' is accessible."
            ) from e

    def diarize(self, audio_path: str | Path) -> list[SpeakerSegment]:
        """
        Run Sortformer diarization and return SpeakerSegments.

        Converts NeMo output format into the common SpeakerSegment format
        so the downstream alignment and role-mapping code is engine-agnostic.
        """
        self._load()
        audio_path = Path(audio_path)
        print(f"[SortformerDiarizer] Diarizing: {audio_path.name}")

        try:
            # NeMo Sortformer inference interface (subject to NeMo version)
            output = self._model.diarize(audio_path=str(audio_path))
            segments: list[SpeakerSegment] = []

            # Convert NeMo output (format may vary by version) to SpeakerSegment
            for item in output:
                # Expected fields: start, end, speaker (varies by NeMo version)
                start = float(item.get("start", 0.0))
                end = float(item.get("end", 0.0))
                speaker = str(item.get("speaker", "SPEAKER_00"))
                segments.append(SpeakerSegment(
                    speaker_id=speaker,
                    start_time=start,
                    end_time=end,
                    confidence=item.get("confidence"),
                ))

            segments.sort(key=lambda s: s.start_time)
            print(f"[SortformerDiarizer] {len(segments)} segment(s) detected.")
            return segments

        except Exception as e:
            raise RuntimeError(
                f"[SortformerDiarizer] Diarization failed: {e}"
            ) from e


# ── Combined Sortformer + WhisperX engine ─────────────────────────────────────

class SortformerEngine:
    """
    Experimental pipeline: NVIDIA Sortformer + WhisperX.

    Architecture:
        Audio → Sortformer (diarization) → SpeakerSegments
             → WhisperX (ASR) → WordTimestamps
             → Word-speaker alignment → Utterances

    This engine is experimental. Validate results before clinical use.
    """

    ENGINE_NAME = "sortformer-whisper"
    MODE = "Experimental Local (CUDA required)"

    def __init__(self, config) -> None:
        if not _SORTFORMER_AVAILABLE:
            raise RuntimeError(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Sortformer engine unavailable                           ║\n"
                "╚══════════════════════════════════════════════════════════╝\n\n"
                "Install optional Sortformer dependencies to enable it:\n"
                "  pip install -r requirements-sortformer.txt\n\n"
                "This engine requires CUDA. Use --engine assemblyai or\n"
                "--engine community-whisperx as alternatives.\n"
            )

        self._diarizer = SortformerDiarizer()
        self._config = config

        # Reuse WhisperX from community_whisperx module
        try:
            from engines.community_whisperx import WhisperXASR, _group_words_into_utterances, _assign_speaker
            whisper_model = getattr(config, "whisper_model", "large-v3")
            self._asr = WhisperXASR(model_name=whisper_model, device="cuda")
            self._group_words = _group_words_into_utterances
            self._assign_speaker = _assign_speaker
        except ImportError as e:
            raise RuntimeError(
                f"[SortformerEngine] WhisperX is also required: {e}\n"
                "Install: pip install -r requirements-local.txt"
            ) from e

        print("[SortformerEngine] Ready (EXPERIMENTAL)")

    def transcribe(self, audio_path: str | Path) -> list[Utterance]:
        """Run Sortformer diarization + WhisperX transcription."""
        audio_path = Path(audio_path)

        # Step 1: Diarization via Sortformer
        segments = self._diarizer.diarize(audio_path)
        if not segments:
            return []

        # Step 2: ASR via WhisperX
        words = self._asr.transcribe(audio_path)
        if not words:
            return []

        # Step 3: Word-to-speaker assignment (same algorithm as Community-1)
        labeled_words = []
        for word in words:
            if word.word:
                speaker_id, is_ambiguous = self._assign_speaker(word, segments)
                labeled_words.append((word, speaker_id, is_ambiguous))

        # Step 4: Group into utterances
        utterances = self._group_words(labeled_words)
        return utterances

    def map_speakers_to_roles(
        self,
        utterances: list[Utterance],
        doctor_speaker: Optional[str] = None,
    ) -> list[Utterance]:
        """Assign Doctor/Patient roles — identical logic to other engines."""
        if not utterances:
            return []
        doctor_raw = doctor_speaker or min(
            utterances, key=lambda u: u.start_time
        ).speaker_id
        all_labels = sorted(set(u.speaker_id for u in utterances))
        label_map = {doctor_raw: "Doctor"}
        for lbl in all_labels:
            if lbl != doctor_raw:
                label_map[lbl] = "Patient"

        return [
            Utterance(
                id=u.id,
                speaker_id=u.speaker_id,
                speaker_label=label_map.get(u.speaker_id, "Patient"),
                text=u.text,
                raw_text=u.raw_text,
                normalized_text=u.normalized_text,
                start_time=u.start_time,
                end_time=u.end_time,
                confidence=u.confidence,
                overlap=u.overlap,
                overlapping_speakers=u.overlapping_speakers,
                words=u.words,
                review_required=u.review_required,
            )
            for u in utterances
        ]
