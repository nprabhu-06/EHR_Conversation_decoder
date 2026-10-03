"""
audio/preprocessing.py — Audio preprocessing utilities.

Enhanced audio conversion pipeline:
  1. Validate audio file (format + existence)
  2. Convert to 16 kHz mono 16-bit PCM WAV via FFmpeg
  3. Preserve original audio (never modify the source file)
  4. Detect audio duration
  5. All timestamps reference the ORIGINAL audio timeline

The original audio is NEVER modified.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Audio file extensions supported for input
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac",
    ".aac", ".webm", ".opus", ".wma", ".3gp",
}


def validate_audio_file(path: str | Path) -> Path:
    """
    Validate that `path` exists and has a supported audio extension.

    Returns the resolved Path on success; raises ValueError on failure.
    """
    audio_path = Path(path).resolve()

    if not audio_path.exists():
        raise ValueError(f"Audio file not found: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Path is not a file: {audio_path}")

    ext = audio_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {ext!r}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return audio_path


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is on PATH and runnable."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def normalize_to_wav(
    audio_path: Path,
    output_dir: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_fmt: str = "s16",
) -> Path:
    """
    Convert an audio file to standard WAV format using FFmpeg.

    Output: 16 kHz, mono, 16-bit PCM WAV.
    The output file is placed in output_dir.
    The original audio_path is NEVER modified.

    Args:
        audio_path:  Source audio file (any supported format).
        output_dir:  Directory for the converted WAV.
        sample_rate: Target sample rate (default 16000 Hz).
        channels:    Number of channels (default 1 = mono).
        sample_fmt:  PCM sample format (default s16 = 16-bit).

    Returns:
        Path to the normalized WAV file.

    Raises:
        RuntimeError: If FFmpeg is not available or conversion fails.
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "\n[ERROR] FFmpeg not found on PATH.\n"
            "FFmpeg is required for audio format conversion.\n\n"
            "Install:\n"
            "  Windows: https://www.gyan.dev/ffmpeg/builds/ — add bin/ to PATH\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg\n"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"{audio_path.stem}_16k.wav"

    cmd = [
        "ffmpeg",
        "-y",                     # Overwrite without asking
        "-i", str(audio_path),    # Input file
        "-ac", str(channels),     # Mono
        "-ar", str(sample_rate),  # 16 kHz
        "-sample_fmt", sample_fmt,# 16-bit PCM
        str(wav_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg audio conversion failed for {audio_path.name}:\n{err}"
        )

    return wav_path


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """
    Get the duration of an audio file in seconds using FFmpeg.

    Returns None if duration cannot be determined.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return float(result.stdout.decode().strip())
    except (ValueError, FileNotFoundError):
        pass

    # Fallback: try FFmpeg itself
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path)],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        for line in stderr.split("\n"):
            if "Duration:" in line:
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = duration_str.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass

    return None
