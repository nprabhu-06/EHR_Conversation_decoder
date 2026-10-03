"""
audio/vad.py — Optional Voice Activity Detection using Silero VAD.

VAD identifies speech vs. non-speech regions in audio.
When enabled, it can be used to pre-filter audio before diarization.

CRITICAL TIMESTAMP RULE:
    VAD does NOT alter original audio timestamps.
    If speech occurs at 125.4 seconds in the original audio,
    the output will still reference 125.4 seconds — not a VAD-relative offset.

OPTIONAL DEPENDENCY:
    Silero VAD requires torch. If unavailable, VAD is simply disabled.
    The application continues without it.

Configuration (in .env):
    ENABLE_VAD=true
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Availability check ────────────────────────────────────────────────────────

_SILERO_AVAILABLE = False
_VAD_MISSING_REASON = ""

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    _VAD_MISSING_REASON = "torch (PyTorch) not installed"

if _TORCH_AVAILABLE:
    try:
        # Silero VAD v5+ installed as silero-vad package
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps  # type: ignore
        _SILERO_AVAILABLE = True
    except ImportError:
        try:
            # Older Silero VAD installed via torch.hub
            _SILERO_HUB = True
            _SILERO_AVAILABLE = True
        except Exception:
            _SILERO_AVAILABLE = False
            _VAD_MISSING_REASON = "silero-vad not installed (pip install silero-vad)"


def is_available() -> bool:
    """Return True if Silero VAD dependencies are installed."""
    return _SILERO_AVAILABLE


@dataclass
class SpeechRegion:
    """
    A detected speech segment in the original audio timeline.

    IMPORTANT: start and end are in ORIGINAL audio seconds,
    not VAD-relative offsets.
    """
    start: float   # Original audio timestamp (seconds)
    end: float     # Original audio timestamp (seconds)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_speech_regions(
    audio_path: Path,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> list[SpeechRegion]:
    """
    Detect speech regions in an audio file using Silero VAD.

    CRITICAL: Returns timestamps relative to the ORIGINAL audio timeline.
    Timestamps are NOT shifted or renumbered.

    Args:
        audio_path: Path to a WAV file (16 kHz mono recommended).
        threshold:  VAD confidence threshold [0, 1]. Higher = more conservative.
        min_speech_duration_ms: Minimum speech segment duration to keep.
        min_silence_duration_ms: Minimum silence to split segments.

    Returns:
        List of SpeechRegion objects in original audio time.

    Raises:
        RuntimeError: If VAD dependencies are not available.
    """
    if not _SILERO_AVAILABLE:
        raise RuntimeError(
            f"\n[VAD] Silero VAD is not available: {_VAD_MISSING_REASON}\n"
            "Disable VAD with ENABLE_VAD=false in .env, or install:\n"
            "  pip install silero-vad\n"
            "(Also requires PyTorch: https://pytorch.org/get-started/locally/)\n"
        )

    audio_path = Path(audio_path)
    print(f"[VAD] Running Silero VAD on: {audio_path.name}")

    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps  # type: ignore

        model = load_silero_vad()
        wav = read_audio(str(audio_path), sampling_rate=16000)

        raw_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=True,  # <-- CRITICAL: ensures original timestamps
        )

        regions = [
            SpeechRegion(start=t["start"], end=t["end"])
            for t in raw_timestamps
        ]
        total_speech = sum(r.duration for r in regions)
        print(
            f"[VAD] Detected {len(regions)} speech region(s) "
            f"totaling {total_speech:.1f}s"
        )
        return regions

    except Exception as e:
        # Fall back gracefully — VAD failure should not stop transcription
        print(f"[VAD] WARNING: VAD failed ({e}). Continuing without VAD.")
        return []


def filter_segments_to_speech(
    segments: list,
    speech_regions: list[SpeechRegion],
    tolerance_sec: float = 0.5,
) -> list:
    """
    Filter diarization segments to only those overlapping with speech regions.

    Args:
        segments:       List of SpeakerSegment objects.
        speech_regions: List of SpeechRegion from detect_speech_regions().
        tolerance_sec:  Extra padding around speech regions.

    Returns:
        Filtered list of segments. Timestamps are PRESERVED (not shifted).
    """
    if not speech_regions:
        return segments  # No VAD data — return all segments

    def overlaps_speech(seg) -> bool:
        seg_start = seg.start_time - tolerance_sec
        seg_end = seg.end_time + tolerance_sec
        for region in speech_regions:
            if region.start < seg_end and region.end > seg_start:
                return True
        return False

    filtered = [s for s in segments if overlaps_speech(s)]
    removed = len(segments) - len(filtered)
    if removed > 0:
        print(f"[VAD] Filtered out {removed} non-speech segment(s).")
    return filtered
