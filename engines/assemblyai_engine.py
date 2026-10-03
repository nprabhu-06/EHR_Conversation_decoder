"""
engines/assemblyai_engine.py — AssemblyAI cloud transcription + diarization engine.

Enhanced version of the original AssemblyAIEngine from transcriber.py.
Uses the new SpeakerSegment / WordTimestamp / Utterance types from engines.base
while preserving all original functionality.

This engine is explicitly cloud-based. Audio is sent to AssemblyAI's servers.
For local/private processing, use community-whisperx instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from engines.base import SpeakerSegment, WordTimestamp, Utterance, CombinedEngine

if TYPE_CHECKING:
    pass


# ── Expanded medical vocabulary ────────────────────────────────────────────────
# Loaded from medical_vocabulary/ files at runtime; these are fallback defaults.

_MEDICAL_WORD_BOOST_DEFAULT = [
    "diagnosis", "prognosis", "prescription", "medication", "dosage",
    "symptom", "chronic", "acute", "hypertension", "diabetes", "tachycardia",
    "bradycardia", "dyspnea", "arrhythmia", "myocardial", "infarction",
    "cerebrovascular", "osteoporosis", "arthritis", "oncology", "radiograph",
    "echocardiogram", "electrocardiogram", "haemoglobin", "hemoglobin",
    "leukocyte", "erythrocyte", "platelet", "biopsy", "metastasis",
    "carcinoma", "lymphoma", "antibiotic", "analgesic", "antipyretic",
    "anticoagulant", "corticosteroid", "bronchodilator", "immunosuppressant",
    "paracetamol", "ibuprofen", "metformin", "amoxicillin", "atorvastatin",
    "amlodipine", "ramipril", "omeprazole", "salbutamol", "prednisolone",
]


def _load_vocabulary_boost() -> list[str]:
    """Load medical terms from vocabulary files for word boost."""
    vocab_dir = Path(__file__).parent.parent / "medical_vocabulary"
    terms: set[str] = set(_MEDICAL_WORD_BOOST_DEFAULT)

    for fname in ("medications.txt", "investigations.txt", "procedures.txt", "diseases.txt"):
        fpath = vocab_dir / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" not in line:
                        # Only include single-word or short two-word terms
                        if len(line.split()) <= 3:
                            terms.add(line)

    # AssemblyAI word boost limit is ~1000 terms; keep the most useful ones
    return sorted(terms)[:500]


class AssemblyAIEngineV2(CombinedEngine):
    """
    Cloud-based transcription + diarization via the AssemblyAI Python SDK.

    ⚠️  CLOUD PROCESSING — Audio is sent to AssemblyAI's servers.
         For patient data, a Business Associate Agreement (BAA) with
         AssemblyAI is required before use with PHI.
         See: https://www.assemblyai.com/hipaa

    Features:
      - Speaker diarization (identifies SPEAKER_A / SPEAKER_B)
      - Medical word boost for improved domain accuracy
      - Per-utterance confidence scores
      - Word-level timestamps
    """

    ENGINE_NAME = "assemblyai"
    MODE = "Cloud"

    def __init__(self, config) -> None:
        try:
            import assemblyai as aai
        except ImportError as e:
            raise RuntimeError(
                "\n[ERROR] assemblyai SDK is not installed.\n"
                "Run: pip install assemblyai\n"
            ) from e

        if not config.assemblyai_api_key:
            raise RuntimeError(
                "\n[ERROR] ASSEMBLYAI_API_KEY is not set.\n"
                "Add it to your .env file:\n"
                "  ASSEMBLYAI_API_KEY=your_key_here\n"
                "Get a free key at: https://www.assemblyai.com/app/account\n"
            )

        aai.settings.api_key = config.assemblyai_api_key
        self._aai = aai
        self._config = config

        # Build medical word boost list
        self._word_boost = _load_vocabulary_boost()
        print(
            f"[AssemblyAIEngineV2] Initialized (cloud mode) | "
            f"API key=***{config.assemblyai_api_key[-4:]} | "
            f"Word boost: {len(self._word_boost)} medical terms"
        )

    def transcribe_and_diarize(self, audio_path: str | Path) -> list[Utterance]:
        """
        Upload audio to AssemblyAI, run diarization + transcription, return Utterances.

        Utterances at this stage have raw speaker labels (SPEAKER_A, SPEAKER_B).
        Call map_speakers_to_roles() to assign Doctor/Patient roles.
        """
        audio_path = Path(audio_path)
        print(f"[AssemblyAIEngineV2] Uploading: {audio_path.name}")

        aai_config = self._aai.TranscriptionConfig(
            speaker_labels=True,
            word_boost=self._word_boost,
            boost_param=self._aai.WordBoost.high,
            punctuate=True,
            format_text=True,
        )

        transcriber = self._aai.Transcriber(config=aai_config)
        transcript = transcriber.transcribe(str(audio_path))

        if transcript.status == self._aai.TranscriptStatus.error:
            raise RuntimeError(
                f"AssemblyAI transcription failed: {transcript.error}"
            )

        duration = transcript.audio_duration or 0.0
        print(
            f"[AssemblyAIEngineV2] Complete — "
            f"Duration: {duration:.1f}s | "
            f"Utterances: {len(transcript.utterances or [])}"
        )

        utterances: list[Utterance] = []
        for idx, utt in enumerate(transcript.utterances or [], start=1):
            # Extract word-level timestamps if available
            words: list[WordTimestamp] = []
            if hasattr(utt, "words") and utt.words:
                for w in utt.words:
                    words.append(WordTimestamp(
                        word=w.text.strip() if hasattr(w, "text") else str(w),
                        start=w.start / 1000.0 if hasattr(w, "start") else 0.0,
                        end=w.end / 1000.0 if hasattr(w, "end") else 0.0,
                        confidence=w.confidence if hasattr(w, "confidence") else None,
                    ))

            raw_label = f"SPEAKER_{utt.speaker}"  # e.g. SPEAKER_A
            text = utt.text.strip()

            utterances.append(Utterance(
                id=idx,
                speaker_id=raw_label,
                speaker_label=raw_label,   # Will be remapped to Doctor/Patient later
                text=text,
                raw_text=text,
                normalized_text=text,
                start_time=utt.start / 1000.0,
                end_time=utt.end / 1000.0,
                confidence=utt.confidence,
                words=words,
                review_required=(
                    utt.confidence is not None
                    and utt.confidence < getattr(self._config, "low_confidence_threshold", 0.70)
                ),
            ))

        return sorted(utterances, key=lambda u: u.start_time)

    # ── Backward-compatible transcribe() method ────────────────────────────────
    def transcribe(self, audio_path: str | Path) -> list[Utterance]:
        """Alias for transcribe_and_diarize() — backward compatibility."""
        return self.transcribe_and_diarize(audio_path)

    def map_speakers_to_roles(
        self,
        utterances: list[Utterance],
        doctor_speaker: Optional[str] = None,
    ) -> list[Utterance]:
        """
        Assign 'Doctor' / 'Patient' labels to speaker-diarized utterances.

        Args:
            utterances:     Raw utterances from transcribe().
            doctor_speaker: If provided, this raw label maps to 'Doctor'.
                            If None, uses first-speaker heuristic.
        """
        if not utterances:
            return []

        if doctor_speaker is not None:
            doctor_raw = doctor_speaker
        else:
            doctor_raw = min(utterances, key=lambda u: u.start_time).speaker_id

        all_labels = sorted(set(u.speaker_id for u in utterances))
        non_doctor = [lbl for lbl in all_labels if lbl != doctor_raw]

        label_map: dict[str, str] = {doctor_raw: "Doctor"}
        for lbl in non_doctor:
            label_map[lbl] = "Patient"

        if len(all_labels) > 2:
            print(
                f"[AssemblyAIEngineV2] WARNING: {len(all_labels)} speakers detected. "
                "All speakers except the Doctor will be labelled 'Patient'."
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
