"""engines/__init__.py — Engine factory and availability detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


def get_available_engines() -> dict[str, bool]:
    """
    Probe which engines are available in the current environment.

    Returns a dict mapping engine name → True/False availability.
    """
    available: dict[str, bool] = {}

    # AssemblyAI — requires assemblyai SDK and API key (key checked at runtime)
    try:
        import assemblyai  # noqa: F401
        available["assemblyai"] = True
    except ImportError:
        available["assemblyai"] = False

    # Community-1 + WhisperX — requires torch, pyannote.audio, whisperx
    try:
        import torch  # noqa: F401
        import pyannote.audio  # noqa: F401
        import whisperx  # noqa: F401
        available["community-whisperx"] = True
    except ImportError:
        available["community-whisperx"] = False

    # Backward-compat alias
    available["pyannote-whisperx"] = available["community-whisperx"]

    # Sortformer — requires nemo_toolkit
    try:
        import nemo  # noqa: F401
        available["sortformer-whisper"] = True
    except ImportError:
        available["sortformer-whisper"] = False

    return available


def create_engine(engine_name: str, config) -> object:
    """
    Factory: instantiate and return the appropriate engine.

    Routes to the new modular engines while maintaining backward
    compatibility with the existing transcriber.create_engine() API.

    Args:
        engine_name: One of assemblyai | community-whisperx |
                     pyannote-whisperx | sortformer-whisper
        config:      Config dataclass instance.

    Returns:
        An engine object with a transcribe() method returning list[Utterance].

    Raises:
        ValueError: If engine_name is unknown.
        RuntimeError: If required dependencies are missing.
    """
    # Normalize aliases
    name = engine_name.lower().strip()
    if name == "pyannote-whisperx":
        print(
            "[engines] NOTE: 'pyannote-whisperx' is now an alias for "
            "'community-whisperx' (pyannote Community-1 pipeline). "
            "Update DEFAULT_ENGINE=community-whisperx in your .env to remove this notice."
        )
        name = "community-whisperx"

    if name == "assemblyai":
        from engines.assemblyai_engine import AssemblyAIEngineV2
        return AssemblyAIEngineV2(config)

    if name == "community-whisperx":
        from engines.community_whisperx import CommunityWhisperXEngine
        return CommunityWhisperXEngine(config)

    if name == "sortformer-whisper":
        from engines.sortformer_engine import SortformerEngine
        return SortformerEngine(config)

    raise ValueError(
        f"Unknown engine: {engine_name!r}. "
        "Valid options: assemblyai | community-whisperx | sortformer-whisper"
    )
