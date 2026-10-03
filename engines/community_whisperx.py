"""
engines/community_whisperx.py — Local primary pipeline using pyannote Community-1 + WhisperX.

This is the PRIMARY LOCAL pipeline replacing the legacy pyannote 3.1 default.

Pipeline:
    Audio
     ↓
    pyannote Community-1  → SpeakerSegment list
     ↓
    WhisperX (configurable model)  → WordTimestamp list
     ↓
    Word-to-speaker alignment
     ↓
    Utterance grouping
     ↓
    list[Utterance]

IMPORTANT: This module requires torch, pyannote.audio, and whisperx.
These are NOT part of the core requirements because Python 3.14 support
is not yet stable. Install requirements-local.txt when using a compatible
Python version (3.11 or 3.12 recommended).

If any dependency is missing, this engine raises a clear ImportError
with install instructions. It does NOT silently fall back to the old model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from engines.base import (
    SpeakerSegment, WordTimestamp, Utterance,
    DiarizationEngine, ASREngine,
)

# ── Dependency check ───────────────────────────────────────────────────────────

_MISSING_DEPS: list[str] = []

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _MISSING_DEPS.append("torch (PyTorch)")

try:
    import pyannote.audio
    _PYANNOTE_AVAILABLE = True
except ImportError:
    _PYANNOTE_AVAILABLE = False
    _MISSING_DEPS.append("pyannote.audio")

try:
    import whisperx
    _WHISPERX_AVAILABLE = True
except ImportError:
    _WHISPERX_AVAILABLE = False
    _MISSING_DEPS.append("whisperx")


def _require_deps() -> None:
    """Raise a clear error if any required dependency is missing."""
    if _MISSING_DEPS:
        deps_str = "\n  - ".join(_MISSING_DEPS)
        raise RuntimeError(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  Community-WhisperX engine: dependencies not installed       ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n\n"
            f"Missing packages:\n  - {deps_str}\n\n"
            "To install the local pipeline dependencies:\n\n"
            "  1. Install PyTorch first (match your CUDA version):\n"
            "     https://pytorch.org/get-started/locally/\n\n"
            "  2. Then install the rest:\n"
            "     pip install -r requirements-local.txt\n\n"
            "NOTE: Python 3.14 is not yet officially supported by PyTorch.\n"
            "      Python 3.11 or 3.12 is recommended for local engine use.\n\n"
            "IMPORTANT: This engine does NOT silently fall back to pyannote 3.1.\n"
            "           Fix the dependency issue or use --engine assemblyai instead.\n"
        )


# ── Pyannote Community-1 diarization ──────────────────────────────────────────

class PyannoteCommunity1Diarizer(DiarizationEngine):
    """
    Diarization backend using pyannote Community-1 pipeline.

    Model: pyannote/speaker-diarization  (Community-1)
    Requires HuggingFace token + model terms acceptance.

    This replaces the legacy pyannote/speaker-diarization-3.1 pipeline
    as the default local diarization engine.
    """

    MODEL_ID = "pyannote/speaker-diarization"

    def __init__(self, hf_token: str, device: Optional[str] = None) -> None:
        _require_deps()
        self._hf_token = hf_token
        self._device = device
        self._pipeline = None
        print(
            f"[PyannoteCommunity1] Initialized. "
            f"Model: {self.MODEL_ID} | Device: {device or 'auto'}"
        )

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        print("[PyannoteCommunity1] Loading speaker diarization pipeline …")
        self._pipeline = pyannote.audio.Pipeline.from_pretrained(
            self.MODEL_ID,
            use_auth_token=self._hf_token,
        )
        if self._device and self._device.startswith("cuda"):
            self._pipeline = self._pipeline.to(torch.device(self._device))
        print("[PyannoteCommunity1] Pipeline loaded successfully.")

    def diarize(self, audio_path: str | Path) -> list[SpeakerSegment]:
        """Run pyannote Community-1 diarization and return SpeakerSegments."""
        _require_deps()
        self._load()

        audio_path = Path(audio_path)
        print(f"[PyannoteCommunity1] Diarizing: {audio_path.name}")

        diarization = self._pipeline(str(audio_path))

        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker_id=speaker,       # e.g. SPEAKER_00
                start_time=turn.start,
                end_time=turn.end,
                confidence=None,          # pyannote Community-1 may not expose per-segment confidence
            ))

        # Sort by start time
        segments.sort(key=lambda s: s.start_time)
        unique_speakers = len(set(s.speaker_id for s in segments))
        print(
            f"[PyannoteCommunity1] Diarization complete: "
            f"{len(segments)} segments | {unique_speakers} speaker(s)"
        )
        return segments


# ── WhisperX ASR ──────────────────────────────────────────────────────────────

class WhisperXASR(ASREngine):
    """
    ASR backend using WhisperX for word-level transcription + alignment.

    Supports configurable Whisper model sizes: tiny, base, small, medium, large-v3
    """

    def __init__(
        self,
        model_name: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float32",
    ) -> None:
        _require_deps()
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._compute_type = compute_type
        self._model = None
        print(
            f"[WhisperXASR] Initialized. "
            f"Model: {model_name} | Device: {self._device} | "
            f"Compute: {compute_type}"
        )
        if self._device == "cpu" and model_name in ("large-v3", "large-v2"):
            print(
                f"[WhisperXASR] ⚠ WARNING: {model_name} on CPU will be very slow. "
                "Consider WHISPER_MODEL=medium for CPU-only systems."
            )

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"[WhisperXASR] Loading WhisperX model ({self._model_name}) …")
        self._model = whisperx.load_model(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
            language="en",
        )
        print("[WhisperXASR] Model loaded.")

    def transcribe(self, audio_path: str | Path) -> list[WordTimestamp]:
        """
        Transcribe audio and return word-level timestamps.

        Uses WhisperX forced alignment for precise word timing.
        """
        _require_deps()
        self._load()

        audio_path = Path(audio_path)
        print(f"[WhisperXASR] Transcribing: {audio_path.name}")

        audio = whisperx.load_audio(str(audio_path))
        result = self._model.transcribe(audio, batch_size=16)

        print("[WhisperXASR] Aligning word timestamps …")
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language", "en"),
            device=self._device,
        )
        aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, self._device,
            return_char_alignments=False,
        )

        words: list[WordTimestamp] = []
        for w in aligned.get("word_segments", []):
            words.append(WordTimestamp(
                word=w.get("word", "").strip(),
                start=w.get("start", 0.0),
                end=w.get("end", w.get("start", 0.0) + 0.1),
                confidence=w.get("score"),   # WhisperX word-level confidence
            ))

        print(f"[WhisperXASR] {len(words)} word(s) transcribed.")
        return words


# ── Word-to-speaker alignment ──────────────────────────────────────────────────

def _assign_speaker(
    word: WordTimestamp,
    segments: list[SpeakerSegment],
) -> tuple[str, bool]:
    """
    Assign a speaker to a word using maximum overlap.

    Returns (speaker_id, is_ambiguous).
    is_ambiguous=True if the word overlaps multiple speakers significantly.
    """
    w_start, w_end = word.start, word.end
    w_mid = (w_start + w_end) / 2.0

    overlaps: list[tuple[float, str]] = []  # (overlap_duration, speaker_id)
    for seg in segments:
        overlap = min(w_end, seg.end_time) - max(w_start, seg.start_time)
        if overlap > 0:
            overlaps.append((overlap, seg.speaker_id))

    if not overlaps:
        # No overlap — use nearest segment by midpoint
        nearest = min(segments, key=lambda s: abs(w_mid - (s.start_time + s.end_time) / 2.0))
        return nearest.speaker_id, False

    # Sort by overlap descending
    overlaps.sort(key=lambda x: x[0], reverse=True)
    best_speaker = overlaps[0][1]

    # Flag as ambiguous if two speakers have similar overlap (within 20%)
    is_ambiguous = (
        len(overlaps) >= 2
        and overlaps[0][0] > 0
        and (overlaps[0][0] - overlaps[1][0]) / overlaps[0][0] < 0.2
    )

    return best_speaker, is_ambiguous


def _group_words_into_utterances(
    labeled_words: list[tuple[WordTimestamp, str, bool]],  # (word, speaker_id, is_ambiguous)
) -> list[Utterance]:
    """
    Group consecutive words from the same speaker into Utterance objects.

    Args:
        labeled_words: List of (WordTimestamp, speaker_id, is_ambiguous) tuples.

    Returns:
        List of Utterance objects sorted by start_time.
    """
    if not labeled_words:
        return []

    utterances: list[Utterance] = []
    current_speaker = labeled_words[0][1]
    current_words: list[tuple[WordTimestamp, bool]] = [(labeled_words[0][0], labeled_words[0][2])]

    for word, speaker_id, is_ambiguous in labeled_words[1:]:
        if speaker_id == current_speaker:
            current_words.append((word, is_ambiguous))
        else:
            utterances.append(_words_to_utterance(current_words, current_speaker, len(utterances) + 1))
            current_speaker = speaker_id
            current_words = [(word, is_ambiguous)]

    # Flush last group
    if current_words:
        utterances.append(_words_to_utterance(current_words, current_speaker, len(utterances) + 1))

    return utterances


def _words_to_utterance(
    words: list[tuple[WordTimestamp, bool]],
    speaker_id: str,
    utt_id: int,
) -> Utterance:
    """Collapse a group of words into a single Utterance."""
    word_objs = [w for w, _ in words]
    ambiguous_any = any(a for _, a in words)

    text = " ".join(w.word for w in word_objs if w.word)
    scores = [w.confidence for w in word_objs if w.confidence is not None]
    avg_conf = sum(scores) / len(scores) if scores else None

    return Utterance(
        id=utt_id,
        speaker_id=speaker_id,
        speaker_label=speaker_id,    # Remapped to Doctor/Patient later
        text=text.strip(),
        raw_text=text.strip(),
        normalized_text=text.strip(),
        start_time=word_objs[0].start,
        end_time=word_objs[-1].end,
        confidence=avg_conf,
        overlap=ambiguous_any,
        words=word_objs,
    )


# ── Main combined engine ───────────────────────────────────────────────────────

class CommunityWhisperXEngine:
    """
    Primary local pipeline: pyannote Community-1 + WhisperX.

    Architecture:
        Audio → Preprocessing → pyannote Community-1 (diarization)
             → WhisperX (ASR + alignment) → Word-speaker assignment
             → Utterance grouping → list[Utterance]

    This engine is fully local/self-hosted. Audio never leaves your machine.
    """

    ENGINE_NAME = "community-whisperx"
    MODE = "Local (self-hosted)"

    def __init__(self, config) -> None:
        _require_deps()

        if not config.hf_token:
            raise RuntimeError(
                "\n[ERROR] HUGGINGFACE_TOKEN is not set.\n"
                "The community-whisperx engine needs a HuggingFace token to\n"
                "download pyannote Community-1 model weights.\n\n"
                "Steps:\n"
                "  1. Create a token at https://huggingface.co/settings/tokens\n"
                "  2. Accept model terms at:\n"
                "       https://huggingface.co/pyannote/speaker-diarization\n"
                "  3. Add to .env:\n"
                "       HUGGINGFACE_TOKEN=hf_...\n"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = getattr(config, "whisper_model", "large-v3")
        compute_type = "float16" if device == "cuda" else "float32"

        self._diarizer = PyannoteCommunity1Diarizer(hf_token=config.hf_token, device=device)
        self._asr = WhisperXASR(
            model_name=whisper_model,
            device=device,
            compute_type=compute_type,
        )
        self._config = config
        self._last_segments: list[SpeakerSegment] = []  # Stored after each diarize() call

        print(
            f"[CommunityWhisperXEngine] Ready | "
            f"Diarizer: pyannote Community-1 | "
            f"ASR: WhisperX {whisper_model} | "
            f"Device: {device}"
        )

    def transcribe(self, audio_path: str | Path) -> list[Utterance]:
        """
        Run pyannote Community-1 + WhisperX and return Utterances with raw speaker labels.

        Call map_speakers_to_roles() afterwards to assign Doctor/Patient.
        Diarization segments are stored in self._last_segments for overlap detection.
        """
        audio_path = Path(audio_path)

        # Step 1: Diarization
        segments = self._diarizer.diarize(audio_path)
        if not segments:
            print("[CommunityWhisperXEngine] WARNING: No speaker segments from diarization.")
            return []

        # Store segments for downstream overlap detection
        self._last_segments = segments

        # Step 2: ASR + word alignment
        words = self._asr.transcribe(audio_path)
        if not words:
            print("[CommunityWhisperXEngine] WARNING: No words from transcription.")
            return []

        # Step 3: Word-to-speaker assignment
        labeled_words: list[tuple[WordTimestamp, str, bool]] = []
        for word in words:
            if word.word:  # Skip empty tokens
                speaker_id, is_ambiguous = _assign_speaker(word, segments)
                labeled_words.append((word, speaker_id, is_ambiguous))

        # Step 4: Group into utterances
        utterances = _group_words_into_utterances(labeled_words)
        print(f"[CommunityWhisperXEngine] {len(utterances)} utterance(s) produced.")
        return utterances

    def map_speakers_to_roles(
        self,
        utterances: list[Utterance],
        doctor_speaker: Optional[str] = None,
    ) -> list[Utterance]:
        """Assign Doctor/Patient roles to diarized utterances."""
        if not utterances:
            return []

        if doctor_speaker is not None:
            doctor_raw = doctor_speaker
        else:
            doctor_raw = min(utterances, key=lambda u: u.start_time).speaker_id

        all_labels = sorted(set(u.speaker_id for u in utterances))
        non_doctor = [lbl for lbl in all_labels if lbl != doctor_raw]
        label_map = {doctor_raw: "Doctor"}
        for lbl in non_doctor:
            label_map[lbl] = "Patient"

        if len(all_labels) > 2:
            print(
                f"[CommunityWhisperXEngine] WARNING: {len(all_labels)} speakers detected. "
                "All except the Doctor will be labelled 'Patient'."
            )

        result = []
        for u in utterances:
            role = label_map.get(u.speaker_id, "Patient")
            result.append(Utterance(
                id=u.id,
                speaker_id=u.speaker_id,
                speaker_label=role,
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
            ))
        return result
