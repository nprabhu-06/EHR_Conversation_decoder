"""
config.py — Configuration loading, API key management, and startup validation.

Loads settings from .env via python-dotenv and validates all runtime
dependencies before the application starts. Fails fast with clear error
messages so users know exactly what to fix.

Extended with:
  - Multi-engine support (community-whisperx, sortformer-whisper, assemblyai)
  - Configurable Whisper model
  - VAD / noise reduction toggles
  - Multi-speaker role classification strategy
  - Language detection, overlap detection, clinical summary toggles
  - Confidence threshold configuration
  - Consent and audit settings
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

# Load .env from the project root (directory containing this file)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

# ── Type aliases ───────────────────────────────────────────────────────────────

EngineChoice = Literal[
    "assemblyai",
    "community-whisperx",
    "pyannote-whisperx",   # backward-compat alias → community-whisperx
    "sortformer-whisper",
]

MappingStrategy = Literal[
    "first_speaker",
    "manual",
    "interactive",
]

ASREngineChoice = Literal["whisper", "indicwhisper"]


# ── Central config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Central configuration object for the application."""

    # ── Engine selection ───────────────────────────────────────────────────────
    engine: EngineChoice = field(
        default_factory=lambda: os.getenv("DEFAULT_ENGINE", "community-whisperx")  # type: ignore
    )

    # ── ASR ───────────────────────────────────────────────────────────────────
    asr_engine: ASREngineChoice = field(
        default_factory=lambda: os.getenv("ASR_ENGINE", "whisper")  # type: ignore
    )
    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3")
    )

    # ── API credentials ────────────────────────────────────────────────────────
    assemblyai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ASSEMBLYAI_API_KEY")
    )
    hf_token: Optional[str] = field(
        default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN")
    )

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output"))
    )

    # ── Speaker role mapping ───────────────────────────────────────────────────
    speaker_mapping_strategy: MappingStrategy = field(
        default_factory=lambda: os.getenv("SPEAKER_MAPPING_STRATEGY", "first_speaker")  # type: ignore
    )
    # Optional override: raw diarization label that maps to "Doctor"
    doctor_speaker_label: Optional[str] = None

    # ── Audio preprocessing toggles ────────────────────────────────────────────
    enable_vad: bool = field(
        default_factory=lambda: os.getenv("ENABLE_VAD", "true").lower() == "true"
    )
    enable_noise_reduction: bool = field(
        default_factory=lambda: os.getenv("ENABLE_NOISE_REDUCTION", "false").lower() == "true"
    )
    enable_echo_cancellation: bool = field(
        default_factory=lambda: os.getenv("ENABLE_ECHO_CANCELLATION", "false").lower() == "true"
    )

    # ── Pipeline feature toggles ───────────────────────────────────────────────
    enable_overlap_detection: bool = field(
        default_factory=lambda: os.getenv("ENABLE_OVERLAP_DETECTION", "true").lower() == "true"
    )
    enable_language_detection: bool = field(
        default_factory=lambda: os.getenv("ENABLE_LANGUAGE_DETECTION", "true").lower() == "true"
    )
    enable_medical_normalization: bool = field(
        default_factory=lambda: os.getenv("ENABLE_MEDICAL_NORMALIZATION", "true").lower() == "true"
    )
    enable_clinical_summary: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CLINICAL_SUMMARY", "true").lower() == "true"
    )
    enable_sortformer: bool = field(
        default_factory=lambda: os.getenv("ENABLE_SORTFORMER", "false").lower() == "true"
    )

    # ── Confidence thresholds ──────────────────────────────────────────────────
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    )
    high_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.90"))
    )
    low_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.70"))
    )

    # ── Consent ───────────────────────────────────────────────────────────────
    consent_logged: bool = False  # Set to True after explicit consent confirmation

    # ── Encounter metadata ─────────────────────────────────────────────────────
    encounter_id: Optional[str] = None

    def __post_init__(self) -> None:
        # Normalize output_dir to absolute path relative to project root
        if not self.output_dir.is_absolute():
            self.output_dir = Path(__file__).parent / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize pyannote-whisperx alias
        if self.engine == "pyannote-whisperx":
            self.engine = "community-whisperx"  # type: ignore


# ── Hardware detection ─────────────────────────────────────────────────────────

def detect_hardware() -> dict:
    """
    Detect available compute hardware.

    Returns:
        dict with keys: device, cuda_available, gpu_name, vram_gb
    """
    info = {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda"
            gpu_id = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(gpu_id)
            vram = torch.cuda.get_device_properties(gpu_id).total_memory
            info["vram_gb"] = round(vram / 1024**3, 1)
    except ImportError:
        pass
    return info


def _warn_model_resources(whisper_model: str, hw: dict) -> None:
    """Warn if selected model may exceed available resources."""
    large_models = {"large-v3", "large-v2", "large"}
    if whisper_model in large_models and not hw["cuda_available"]:
        print(
            f"\n[config] ⚠ WARNING: Whisper model '{whisper_model}' on CPU will be extremely slow.\n"
            "  Recommendation: Set WHISPER_MODEL=medium in .env for CPU-only systems.\n"
            "  Or use --engine assemblyai for fast cloud processing.\n"
        )
    if (whisper_model in large_models and hw["cuda_available"]
            and hw["vram_gb"] is not None and hw["vram_gb"] < 6):
        print(
            f"\n[config] ⚠ WARNING: Whisper '{whisper_model}' requires ~6GB VRAM.\n"
            f"  Detected GPU VRAM: {hw['vram_gb']}GB. Consider WHISPER_MODEL=medium.\n"
        )


# ── Dependency validators ──────────────────────────────────────────────────────

def _check_ffmpeg() -> None:
    """Verify FFmpeg is installed and on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "\n[ERROR] FFmpeg not found on PATH.\n"
            "FFmpeg is required for audio format conversion.\n\n"
            "Install instructions:\n"
            "  Windows : https://www.gyan.dev/ffmpeg/builds/ — add bin/ to PATH\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "[ERROR] FFmpeg was found but failed to execute. "
            "Your FFmpeg installation may be corrupted.",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_assemblyai_key(config: Config) -> None:
    """Verify the AssemblyAI API key is present."""
    if not config.assemblyai_api_key:
        print(
            "\n[ERROR] ASSEMBLYAI_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "  ASSEMBLYAI_API_KEY=your_key_here\n\n"
            "Get a free key at: https://www.assemblyai.com/app/account\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_hf_token(config: Config) -> None:
    """Verify the HuggingFace token is present."""
    if not config.hf_token:
        print(
            "\n[ERROR] HUGGINGFACE_TOKEN is not set.\n"
            "The community-whisperx engine needs a HuggingFace access token to\n"
            "download the diarization model weights on first run.\n\n"
            "Steps:\n"
            "  1. Create a token at https://huggingface.co/settings/tokens\n"
            "  2. Accept model terms at:\n"
            "       https://huggingface.co/pyannote/speaker-diarization\n"
            "  3. Add to .env:\n"
            "       HUGGINGFACE_TOKEN=hf_...\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_local_deps() -> None:
    """Check pyannote + whisperx are importable for local engines."""
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch (PyTorch) — install from https://pytorch.org/get-started/locally/")

    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        missing.append("pyannote.audio — run: pip install pyannote.audio")

    try:
        import whisperx  # noqa: F401
    except ImportError:
        missing.append("whisperx — run: pip install whisperx")

    if missing:
        deps_str = "\n  - ".join(missing)
        print(
            f"\n[ERROR] Local engine dependencies missing:\n  - {deps_str}\n\n"
            "Install with:\n"
            "  pip install -r requirements-local.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_assemblyai_sdk() -> None:
    """Verify the AssemblyAI SDK is importable."""
    try:
        import assemblyai  # noqa: F401
    except ImportError:
        print(
            "\n[ERROR] assemblyai SDK is not installed.\n"
            "Run: pip install assemblyai\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _check_sortformer() -> None:
    """Check Sortformer availability (non-fatal)."""
    try:
        import nemo  # noqa: F401
        print("[config] [OK] NeMo (Sortformer) available")
    except ImportError:
        print(
            "[config] [INFO] Sortformer (NeMo) not installed. "
            "Use --engine community-whisperx or --engine assemblyai instead. "
            "To install: pip install -r requirements-sortformer.txt"
        )


# ── Startup banner ─────────────────────────────────────────────────────────────

def print_startup_banner(config: Config, hw: dict) -> None:
    """Print the startup information panel."""
    device_str = hw["device"].upper()
    if hw["gpu_name"]:
        device_str += f" ({hw['gpu_name']}, {hw['vram_gb']}GB VRAM)"

    mode_map = {
        "community-whisperx": ("Community-1 + WhisperX", "Local (self-hosted)"),
        "assemblyai":          ("AssemblyAI",              "Cloud ⚠️  Audio sent to AssemblyAI"),
        "sortformer-whisper":  ("NVIDIA Sortformer + Whisper", "Experimental Local (CUDA required)"),
    }
    engine_label, mode_label = mode_map.get(
        config.engine, (config.engine, "Unknown")
    )

    banner_lines = [
        "=" * 60,
        "Doctor-Patient Clinical Conversation Intelligence",
        "=" * 60,
        "",
        f"Engine            : {engine_label}",
        f"Mode              : {mode_label}",
        f"ASR Model         : Whisper {config.whisper_model}",
        f"Device            : {device_str}",
        f"VAD               : {'Enabled' if config.enable_vad else 'Disabled'}",
        f"Noise Reduction   : {'Enabled' if config.enable_noise_reduction else 'Disabled'}",
        f"Overlap Detection : {'Enabled' if config.enable_overlap_detection else 'Disabled'}",
        f"Language Detect   : {'Enabled' if config.enable_language_detection else 'Disabled'}",
        f"Med. Normalization: {'Enabled' if config.enable_medical_normalization else 'Disabled'}",
        f"Clinical Summary  : {'Enabled' if config.enable_clinical_summary else 'Disabled'}",
        f"Speaker Mapping   : {config.speaker_mapping_strategy}",
        f"Confidence Thresh : {config.confidence_threshold}",
        "",
        "=" * 60,
    ]
    print("\n".join(banner_lines))


# ── Public API ────────────────────────────────────────────────────────────────

def validate_startup(config: Config) -> None:
    """
    Run all startup dependency checks for the chosen engine.

    Calls sys.exit(1) with a descriptive message on any failure.
    """
    print(f"[config] Validating startup for engine: {config.engine}")

    # FFmpeg is always required
    _check_ffmpeg()
    print("[config] [OK] FFmpeg found")

    # Detect hardware
    hw = detect_hardware()
    print(f"[config] [OK] Device: {hw['device'].upper()}" +
          (f" | GPU: {hw['gpu_name']}" if hw["gpu_name"] else ""))

    if config.engine == "assemblyai":
        _check_assemblyai_sdk()
        _check_assemblyai_key(config)
        print("[config] [OK] AssemblyAI SDK and API key OK")

    elif config.engine in ("community-whisperx", "pyannote-whisperx"):
        _check_hf_token(config)
        _check_local_deps()
        _warn_model_resources(config.whisper_model, hw)
        print("[config] [OK] HuggingFace token and pyannote/whisperx imports OK")

    elif config.engine == "sortformer-whisper":
        _check_sortformer()
        _check_local_deps()
        _warn_model_resources(config.whisper_model, hw)
        print("[config] [OK] Sortformer dependencies checked")

    else:
        print(
            f"\n[ERROR] Unknown engine: {config.engine!r}\n"
            "Valid options: assemblyai | community-whisperx | sortformer-whisper\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print full startup banner
    print_startup_banner(config, hw)
    print("[config] Startup validation passed.\n")
