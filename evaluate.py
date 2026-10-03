"""
evaluate.py — Evaluation script: DER + WER benchmarking against labeled test data.

Metrics:
  DER (Diarization Error Rate) — via pyannote.metrics
      Measures how accurately the system identifies who spoke when.
      DER = (missed speech + false alarms + speaker confusion) / total reference speech

  WER (Word Error Rate) — via jiwer
      Measures transcription accuracy.
      WER = (substitutions + deletions + insertions) / reference word count

Usage:
  python main.py --evaluate --engine assemblyai
  python main.py --evaluate --engine pyannote-whisperx

──────────────────────────────────────────────────────────────────────────────
TEST DATA LOADER — PLACEHOLDER
──────────────────────────────────────────────────────────────────────────────
The load_test_dataset() function below contains a TODO placeholder.
The loader will be implemented once you share the test_data/ folder structure.

Please add your labeled dataset to test_data/ and then run:
  python main.py --evaluate

Then share the structure (file names, annotation format — RTTM, JSON, CSV,
TextGrid, etc.) and I will fill in the loader to match exactly.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transcriber import Utterance


# ── Ground-truth data model ────────────────────────────────────────────────────

@dataclass
class TestCase:
    """
    One labeled test example from test_data/.

    Fields:
        case_id:        Unique identifier (usually the file stem).
        audio_path:     Path to the audio file for this test case.
        reference_utterances: Ground-truth list of utterances
                        (same schema as transcriber.Utterance, but loaded
                         from annotation files rather than ASR output).
        doctor_speaker_label: The raw diarization label in the *hypothesis*
                        that corresponds to "Doctor" (used to seed role mapping
                        from ground truth, bypassing the first-speaker heuristic).
    """

    case_id: str
    audio_path: Path
    reference_utterances: list[Utterance]
    doctor_speaker_label: Optional[str] = None  # e.g. "SPEAKER_A" or "SPEAKER_00"
    prompt: str = ""                             # Medical intent category from CSV


# ── Dataset loader (PLACEHOLDER — implement after confirming folder structure) ─

def load_test_dataset(test_data_dir: Path) -> list[TestCase]:
    """
    Load labeled test cases from the 'Medical Speech, Transcription, and Intent' dataset.

    Dataset structure (auto-detected inside test_data/):
        test_data/
        └── Medical Speech, Transcription, and Intent/
            ├── overview-of-recordings.csv   ← ground-truth transcriptions
            └── recordings/
                ├── train/    ← WAV files  (e.g. 1249120_43453425_58166571.wav)
                ├── test/
                └── validate/

    CSV columns used:
        file_name  — WAV filename (no path)
        phrase     — Ground-truth transcription of the utterance
        prompt     — Symptom/intent category (used as metadata, not for scoring)
        speaker_id — Crowdsource speaker identifier

    NOTE on evaluation scope:
        Each WAV contains a single speaker saying a single medical phrase.
        This dataset is suitable for **WER evaluation only** — there is no
        2-speaker diarization ground truth, so DER is reported as N/A.
        The engine is run on each WAV and its transcript is compared against
        the known `phrase` using jiwer WER.

    Args:
        test_data_dir: Path to the test_data/ directory.

    Returns:
        List of TestCase objects (one per WAV file found in the CSV).
    """
    import csv
    from transcriber import Utterance as _Utterance

    # ── Locate the dataset root ────────────────────────────────────────────────
    dataset_root = test_data_dir / "Medical Speech, Transcription, and Intent"
    csv_path = dataset_root / "overview-of-recordings.csv"
    recordings_root = dataset_root / "recordings"

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected dataset folder not found:\n  {dataset_root}\n\n"
            "Make sure the 'Medical Speech, Transcription, and Intent' folder "
            "is inside test_data/."
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Annotations CSV not found:\n  {csv_path}"
        )

    # Build a lookup: filename → full WAV path (search all splits)
    wav_lookup: dict[str, Path] = {}
    for split in ("train", "test", "validate"):
        split_dir = recordings_root / split
        if split_dir.exists():
            for wav_file in split_dir.glob("*.wav"):
                wav_lookup[wav_file.name] = wav_file

    print(f"[evaluate] WAV files found: {len(wav_lookup)} across train/test/validate splits")

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    test_cases: list[TestCase] = []
    missing: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name", "").strip()
            phrase    = row.get("phrase", "").strip()

            if not file_name or not phrase:
                continue  # Skip malformed rows

            wav_path = wav_lookup.get(file_name)
            if wav_path is None:
                missing.append(file_name)
                continue

            # Each WAV is a single utterance.
            # We label the speaker as "Patient" since these are patient-speech
            # recordings (symptom descriptions). start/end are approximate;
            # pydub can compute exact duration if needed.
            ref_utterance = _Utterance(
                speaker_label="Patient",
                text=phrase,
                start_time=0.0,
                end_time=0.0,
                confidence=None,
            )

            test_cases.append(
                TestCase(
                    case_id=Path(file_name).stem,
                    audio_path=wav_path,
                    reference_utterances=[ref_utterance],
                    doctor_speaker_label=None,  # Single-speaker -- no diarization
                    prompt=row.get("prompt", "").strip(),
                )
            )

    if missing:
        print(
            f"[evaluate] WARNING: {len(missing)} CSV rows had no matching WAV file "
            f"(first 5: {missing[:5]})"
        )

    if not test_cases:
        raise RuntimeError(
            "No valid test cases could be loaded. "
            "Check that WAV files are present in the recordings/ subdirectories."
        )

    print(f"[evaluate] Loaded {len(test_cases)} test case(s) from CSV.")
    return test_cases


# ── DER computation ────────────────────────────────────────────────────────────

def _compute_der(
    reference_utterances: list[Utterance],
    hypothesis_utterances: list[Utterance],
    case_id: str,
) -> float:
    """
    Compute Diarization Error Rate using pyannote.metrics.

    Converts Utterance lists into pyannote Annotation objects for scoring.

    Returns:
        DER as a float in [0.0, ∞). 0.0 = perfect. 1.0 = 100% error.
        Values > 1.0 are possible (false alarms can exceed reference duration).
    """
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def utterances_to_annotation(utterances: list[Utterance]) -> Annotation:
        ann = Annotation(uri=case_id)
        for u in utterances:
            ann[Segment(u.start_time, u.end_time)] = u.speaker_label
        return ann

    reference = utterances_to_annotation(reference_utterances)
    hypothesis = utterances_to_annotation(hypothesis_utterances)

    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)
    return float(der)


# ── WER computation ────────────────────────────────────────────────────────────

def _compute_wer(
    reference_utterances: list[Utterance],
    hypothesis_utterances: list[Utterance],
) -> float:
    """
    Compute Word Error Rate using jiwer.

    Concatenates all reference / hypothesis text into single strings for
    an overall WER across the full transcript.

    Returns:
        WER as a float in [0.0, ∞). 0.0 = perfect.
    """
    import jiwer

    ref_text = " ".join(u.text for u in reference_utterances)
    hyp_text = " ".join(u.text for u in hypothesis_utterances)

    if not ref_text.strip():
        return 0.0  # Empty reference — skip

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    return float(jiwer.wer(ref_text, hyp_text, reference_transform=transformation, hypothesis_transform=transformation))


# ── Main evaluation runner ─────────────────────────────────────────────────────

def run_evaluation(
    engine,
    test_data_dir: Path,
    output_dir: Path,
    max_cases: Optional[int] = None,
) -> None:
    """
    Run DER + WER evaluation for all test cases and write a report.

    For the 'Medical Speech, Transcription, and Intent' dataset, each WAV is
    a single-speaker utterance, so only WER is computed (DER requires 2-speaker
    diarization ground truth). DER is computed only when reference utterances
    have meaningful speaker-labeled time ranges (end_time > start_time).

    Args:
        engine:        Instantiated TranscriptionEngine to evaluate.
        test_data_dir: Path to the labeled test dataset.
        output_dir:    Where to write the evaluation report files.
        max_cases:     If set, limit evaluation to this many cases (for quick runs).
    """
    from utils import ensure_wav, write_json_transcript

    print(f"\n[evaluate] Loading test dataset from: {test_data_dir}")
    test_cases = load_test_dataset(test_data_dir)

    if max_cases and max_cases > 0:
        test_cases = test_cases[:max_cases]
        print(f"[evaluate] Limiting to {max_cases} case(s) (--max-cases).")

    print(f"[evaluate] {len(test_cases)} test case(s) to evaluate.\n")

    per_case_results: list[dict] = []
    total_wer = 0.0
    total_der = 0.0
    der_count = 0
    wer_count = 0

    for i, case in enumerate(test_cases, 1):
        print(f"[evaluate] [{i}/{len(test_cases)}] Case: {case.case_id}")

        # Convert audio to WAV if needed
        try:
            wav_path = ensure_wav(case.audio_path, output_dir)
        except Exception as exc:
            print(f"[evaluate]   ERROR converting audio: {exc}")
            per_case_results.append(
                {"case_id": case.case_id, "error": f"audio conversion: {exc}",
                 "der": None, "wer": None}
            )
            continue

        # Run transcription
        try:
            raw_utterances = engine.transcribe(wav_path)
        except Exception as exc:
            print(f"[evaluate]   ERROR transcribing: {exc}")
            per_case_results.append(
                {"case_id": case.case_id, "error": str(exc), "der": None, "wer": None}
            )
            continue

        # Map speakers — use ground-truth doctor label if provided
        hypothesis = engine.map_speakers_to_roles(
            raw_utterances,
            doctor_speaker=case.doctor_speaker_label,
        )

        # Save hypothesis transcript for inspection
        hyp_path = output_dir / f"eval_{case.case_id}_hypothesis.json"
        write_json_transcript(hypothesis, hyp_path)

        # ── WER (always computed) ───────────────────────────────────────────
        wer = _compute_wer(case.reference_utterances, hypothesis)
        total_wer += wer
        wer_count += 1
        print(f"[evaluate]   WER = {wer:.3f}  ({wer * 100:.1f}%)")

        # ── DER (only when reference has time-stamped speaker segments) ─────
        has_diarization_gt = any(
            u.end_time > u.start_time for u in case.reference_utterances
        )
        der: Optional[float] = None
        if has_diarization_gt:
            der = _compute_der(case.reference_utterances, hypothesis, case.case_id)
            total_der += der
            der_count += 1
            print(f"[evaluate]   DER = {der:.3f}  ({der * 100:.1f}%)")
        else:
            print(f"[evaluate]   DER = N/A (no diarization ground truth for this dataset)")

        # Flatten hypothesis text for storage
        hyp_text = " ".join(u.text for u in hypothesis).strip()
        ref_text = " ".join(u.text for u in case.reference_utterances).strip()

        per_case_results.append(
            {
                "case_id":        case.case_id,
                "prompt":         case.prompt,
                "reference":      ref_text,
                "hypothesis":     hyp_text,
                "der":            round(der, 4) if der is not None else None,
                "wer":            round(wer, 4),
                "hypothesis_path": str(hyp_path),
            }
        )

    avg_wer = round(total_wer / wer_count, 4) if wer_count else None
    avg_der = round(total_der / der_count, 4) if der_count else None

    aggregate = {
        "dataset": "Medical Speech, Transcription, and Intent",
        "engine": engine.__class__.__name__,
        "cases_evaluated": len(per_case_results),
        "wer_cases": wer_count,
        "der_cases": der_count,
        "average_wer": avg_wer,
        "average_der": avg_der,
        "note": (
            "DER is N/A for this dataset — each WAV is a single-speaker utterance. "
            "DER requires 2-speaker diarization ground truth."
            if der_count == 0 else None
        ),
        "per_case": per_case_results,
    }

    # ── Write report ───────────────────────────────────────────────────────────
    report_json = output_dir / "evaluation_report.json"
    report_txt  = output_dir / "evaluation_report.txt"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    der_display = f"{avg_der:.4f} ({avg_der * 100:.1f}%)" if avg_der is not None else "N/A"
    wer_display = f"{avg_wer:.4f} ({avg_wer * 100:.1f}%)" if avg_wer is not None else "N/A"

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("Doctor-Patient Transcription - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset         : Medical Speech, Transcription, and Intent\n")
        f.write(f"Engine          : {engine.__class__.__name__}\n")
        f.write(f"Cases evaluated : {len(per_case_results)}\n")
        f.write(f"Average WER     : {wer_display}\n")
        f.write(f"Average DER     : {der_display}\n")
        if der_count == 0:
            f.write(
                "\nNote: DER is N/A for this dataset. Each recording is a single-speaker\n"
                "      utterance with no diarization ground truth. For DER evaluation,\n"
                "      provide a dataset with multi-speaker, time-stamped annotations.\n"
            )
        f.write("\nPer-Case Results\n")
        f.write("-" * 60 + "\n")
        for r in per_case_results:
            if r.get("error"):
                f.write(f"  {r['case_id'][:45]:45s}  ERROR: {r['error']}\n")
            else:
                der_str = f"{r['der']:.3f}" if r["der"] is not None else " N/A "
                f.write(
                    f"  {r['case_id'][:45]:45s}  "
                    f"WER={r['wer']:.3f}  DER={der_str}\n"
                )

    print(f"\n[evaluate] Report saved:")
    print(f"           JSON -> {report_json}")
    print(f"           TXT  -> {report_txt}")

    # ── PDF evaluation summary report ───────────────────────────────────────
    from utils import write_pdf_evaluation_report, write_pdf_dataset_transcript_report
    report_pdf = output_dir / "evaluation_report.pdf"
    write_pdf_evaluation_report(aggregate, report_pdf)
    print(f"           PDF  -> {report_pdf}")

    # ── PDF dataset transcript report (card-per-case layout) ────────────────
    transcript_pdf = output_dir / "dataset_transcript_report.pdf"
    write_pdf_dataset_transcript_report(per_case_results, aggregate, transcript_pdf)
    print(f"           TRANSCRIPT PDF -> {transcript_pdf}")

    print(
        f"\n[evaluate] Summary: {wer_count} cases scored  |  "
        f"Avg WER={wer_display}  |  Avg DER={der_display}"
    )

