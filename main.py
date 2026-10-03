"""
main.py — CLI entrypoint for the Clinical Conversation Intelligence System.

Module 1: Doctor-Patient / Multi-Party Clinical Conversation Capture,
          Speech-to-Text & Speaker Diarization.

Usage:
    python main.py --audio consultation.mp3
    python main.py --audio consultation.mp3 --engine assemblyai
    python main.py --audio consultation.mp3 --engine community-whisperx
    python main.py --audio consultation.mp3 --engine assemblyai --doctor-speaker SPEAKER_A
    python main.py --audio consultation.mp3 --interactive-speakers
    python main.py --audio consultation.mp3 --clinical-summary
    python main.py --audio consultation.mp3 --whisper-model medium
    python main.py --evaluate --engine assemblyai
    python main.py --benchmark

All options:
    --audio PATH              Path to the input audio file
    --engine ENGINE           assemblyai | community-whisperx | sortformer-whisper
    --whisper-model MODEL     Whisper model: tiny/base/small/medium/large-v3
    --doctor-speaker LABEL    Raw label for the doctor speaker
    --interactive-speakers    Interactively assign roles to each detected speaker
    --clinical-summary        Extract clinical information from transcript
    --no-vad                  Disable Voice Activity Detection pre-filter
    --output-dir PATH         Directory for output files
    --encounter-id ID         Override auto-generated encounter ID
    --consent                 Acknowledge consent before processing
    --evaluate                Run WER/DER evaluation on test_data/
    --benchmark               Compare all engines on a shared test file
    --max-cases N             Limit evaluation to N cases
    --no-validation           Skip startup validation (CI/debug)
"""

from __future__ import annotations

import sys
import io

# ── Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError) ────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
# ──────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ── CLI definition ─────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--audio",
    type=click.Path(exists=False),
    default=None,
    help="Path to the input audio file (.wav, .mp3, .m4a, etc.)",
)
@click.option(
    "--engine",
    type=click.Choice(
        ["assemblyai", "community-whisperx", "pyannote-whisperx", "sortformer-whisper"],
        case_sensitive=False,
    ),
    default=None,
    help="Transcription engine. Overrides DEFAULT_ENGINE in .env",
)
@click.option(
    "--whisper-model",
    default=None,
    metavar="MODEL",
    help="Whisper model size: tiny|base|small|medium|large-v3. Overrides WHISPER_MODEL in .env",
)
@click.option(
    "--doctor-speaker",
    default=None,
    metavar="LABEL",
    help="Raw diarization label for the Doctor (e.g. SPEAKER_A, SPEAKER_00). Overrides heuristic.",
)
@click.option(
    "--interactive-speakers",
    is_flag=True,
    default=False,
    help="Interactively assign clinical roles (doctor/patient/attendant/etc.) to each speaker.",
)
@click.option(
    "--clinical-summary",
    is_flag=True,
    default=False,
    help="Extract structured clinical information from the transcript (for review only).",
)
@click.option(
    "--no-vad",
    is_flag=True,
    default=False,
    help="Disable Voice Activity Detection pre-filter.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Directory for output files (default: output/ or OUTPUT_DIR in .env)",
)
@click.option(
    "--encounter-id",
    default=None,
    metavar="ID",
    help="Override the auto-generated encounter ID (must start with ENC-)",
)
@click.option(
    "--consent",
    is_flag=True,
    default=False,
    help=(
        "Confirm that all parties in the recording have given explicit informed consent "
        "to be recorded and for the audio to be processed. Required before processing."
    ),
)
@click.option(
    "--evaluate",
    is_flag=True,
    default=False,
    help="Run WER/DER evaluation against test_data/ ground-truth files.",
)
@click.option(
    "--benchmark",
    is_flag=True,
    default=False,
    help="Run benchmark comparison of all engines on test_data/ (requires --audio or test_data/).",
)
@click.option(
    "--max-cases",
    default=None,
    type=int,
    metavar="N",
    help="Limit evaluation to the first N cases (useful for quick spot-checks).",
)
@click.option(
    "--no-validation",
    is_flag=True,
    default=False,
    help="Skip startup dependency validation (for debugging / CI use only).",
)
def main(
    audio: Optional[str],
    engine: Optional[str],
    whisper_model: Optional[str],
    doctor_speaker: Optional[str],
    interactive_speakers: bool,
    clinical_summary: bool,
    no_vad: bool,
    output_dir: Optional[str],
    encounter_id: Optional[str],
    consent: bool,
    evaluate: bool,
    benchmark: bool,
    max_cases: Optional[int],
    no_validation: bool,
) -> None:
    """
    \b
    Clinical Conversation Intelligence — Module 1
    -----------------------------------------------
    Captures multi-party clinical conversations and produces a time-ordered,
    speaker-attributed, role-aware conversation timeline for EHR ingestion.

    Primary output: output/<name>_conversation.json (Module 1 → Module 3 contract)

    Examples:
      python main.py --audio consultation.mp3 --consent
      python main.py --audio consult.wav --engine community-whisperx --consent
      python main.py --audio consult.mp3 --clinical-summary --interactive-speakers --consent
      python main.py --evaluate --engine assemblyai
      python main.py --benchmark --audio sample.mp3
    """
    _print_banner()

    # ── Load configuration ─────────────────────────────────────────────────────
    from config import Config, validate_startup

    config = Config()

    # CLI flags override .env defaults
    if engine:
        config.engine = engine  # type: ignore
    if whisper_model:
        config.whisper_model = whisper_model
    if output_dir:
        config.output_dir = Path(output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)
    if doctor_speaker:
        config.doctor_speaker_label = doctor_speaker
        config.speaker_mapping_strategy = "manual"  # type: ignore
    if interactive_speakers:
        config.speaker_mapping_strategy = "interactive"  # type: ignore
    if clinical_summary:
        config.enable_clinical_summary = True
    if no_vad:
        config.enable_vad = False
    if encounter_id:
        config.encounter_id = encounter_id

    # ── Consent gate ───────────────────────────────────────────────────────────
    # Consent is required for recording/processing. In batch mode, --consent
    # flag or CONSENT_LOGGED=true in .env acknowledges it.
    import os
    env_consent = os.getenv("CONSENT_LOGGED", "false").lower() == "true"
    config.consent_logged = consent or env_consent

    if not evaluate and not benchmark and not no_validation:
        if not config.consent_logged and audio:
            console.print(
                "\n[bold yellow]⚠  CONSENT REQUIRED[/bold yellow]\n\n"
                "Processing audio without confirming informed consent is not permitted.\n"
                "Add [bold]--consent[/bold] to confirm that ALL parties in the recording\n"
                "have given explicit informed consent to be recorded and processed.\n\n"
                "Example:\n"
                "  python main.py --audio consultation.mp3 --consent\n"
            )
            sys.exit(1)

    console.print(
        f"[bold cyan]Engine:[/bold cyan] {config.engine}   "
        f"[bold cyan]Output:[/bold cyan] {config.output_dir}   "
        f"[bold cyan]Consent:[/bold cyan] {'✓' if config.consent_logged else '✗'}"
    )

    # ── Startup validation ─────────────────────────────────────────────────────
    if not no_validation:
        try:
            validate_startup(config)
        except SystemExit:
            raise

    # ── Route ─────────────────────────────────────────────────────────────────
    if benchmark:
        _run_benchmark(audio, config)
    elif evaluate:
        _run_evaluate(config, max_cases=max_cases)
    else:
        if not audio:
            console.print(
                "[bold red]Error:[/bold red] --audio is required.\n"
                "Run [bold]python main.py --help[/bold] for usage.",
                style="red",
            )
            sys.exit(1)
        _run_transcribe(audio, config)


# ── Transcription pipeline ─────────────────────────────────────────────────────

def _run_transcribe(audio_str: str, config) -> None:
    """
    Main transcription pipeline.

    Steps:
      1.  Validate & convert audio to 16kHz WAV
      2.  Optional VAD pre-filter
      3.  Run diarization + ASR via selected engine
      4.  Medical text normalization
      5.  Language detection
      6.  Multi-speaker role assignment
      7.  Overlap detection
      8.  Quality metrics
      9.  Speaker analytics
      10. Optional clinical summary extraction
      11. Write all outputs (canonical JSON, enhanced JSON, TXT, PDF)
    """
    from audio.preprocessing import validate_audio_file, normalize_to_wav, get_audio_duration
    from audio.vad import detect_speech_regions, filter_segments_to_speech, is_available as vad_available
    from transcriber import create_engine
    from nlp.medical_normalization import normalize_utterances
    from nlp.language_detection import tag_utterances_with_language
    from diarization.role_classifier import (
        assign_roles_first_speaker, assign_roles_manual,
        assign_roles_interactive, apply_role_profiles,
    )
    from diarization.overlap import detect_overlaps, mark_utterance_overlaps
    from analytics.quality_metrics import compute_quality_metrics
    from analytics.speaker_stats import compute_speaker_stats
    from outputs.conversation_timeline import write_conversation_timeline, generate_encounter_id
    from outputs.json_writer import write_enhanced_json
    from outputs.txt_writer import write_enhanced_txt
    from outputs.pdf_writer import write_clinical_pdf

    # ── Step 1: Validate audio ────────────────────────────────────────────────
    try:
        audio_path = validate_audio_file(audio_str)
    except ValueError as exc:
        console.print(f"[bold red]Audio error:[/bold red] {exc}")
        sys.exit(1)

    console.print(f"\n[bold]Processing:[/bold] {audio_path}")

    stem = audio_path.stem
    enc_id = config.encounter_id or generate_encounter_id()
    console.print(f"[dim]Encounter ID: {enc_id}[/dim]")

    # ── Convert to 16kHz WAV ──────────────────────────────────────────────────
    with console.status("[bold green]Converting audio to 16 kHz WAV…"):
        try:
            wav_path = normalize_to_wav(audio_path, config.output_dir)
        except Exception as exc:
            console.print(f"[bold red]Audio conversion failed:[/bold red] {exc}")
            sys.exit(1)

    audio_duration = get_audio_duration(wav_path)
    console.print(
        f"[dim]Audio: {wav_path.name} | Duration: {audio_duration:.1f}s[/dim]"
        if audio_duration else f"[dim]Audio: {wav_path.name}[/dim]"
    )

    # ── Step 2: VAD (optional) ────────────────────────────────────────────────
    speech_regions = []
    if config.enable_vad and vad_available():
        with console.status("[bold green]Running Voice Activity Detection…"):
            try:
                speech_regions = detect_speech_regions(wav_path)
            except Exception as exc:
                console.print(f"[yellow]VAD warning:[/yellow] {exc} — continuing without VAD.")

    # ── Step 3: Transcribe ────────────────────────────────────────────────────
    engine = create_engine(config.engine, config)
    with console.status(f"[bold green]Transcribing with {config.engine}…"):
        try:
            raw_utterances = engine.transcribe(wav_path)
        except Exception as exc:
            console.print(f"[bold red]Transcription failed:[/bold red] {exc}")
            sys.exit(1)

    if not raw_utterances:
        console.print(
            "[yellow]Warning:[/yellow] No utterances returned. "
            "Audio may be silent or too short."
        )
        sys.exit(0)

    console.print(f"[green]✓[/green] Transcription: {len(raw_utterances)} utterances")

    # ── Step 4: Medical text normalization ────────────────────────────────────
    if config.enable_medical_normalization:
        with console.status("[bold green]Normalizing medical terminology…"):
            raw_utterances = normalize_utterances(raw_utterances)

    # ── Step 5: Language detection ────────────────────────────────────────────
    language_tags, languages_detected = [], []
    if config.enable_language_detection:
        with console.status("[bold green]Detecting languages…"):
            language_tags, languages_detected = tag_utterances_with_language(raw_utterances)
        console.print(f"[green]✓[/green] Languages detected: {languages_detected}")

    # ── Step 6: Multi-speaker role assignment ─────────────────────────────────
    strategy = config.speaker_mapping_strategy

    with console.status("[bold green]Assigning speaker roles…") if strategy != "interactive" else console.status(""):
        if strategy == "manual" and config.doctor_speaker_label:
            role_map = {config.doctor_speaker_label: "doctor"}
            profiles = assign_roles_manual(raw_utterances, role_map)
        elif strategy == "interactive":
            profiles = assign_roles_interactive(raw_utterances)
        else:
            profiles = assign_roles_first_speaker(raw_utterances)

    utterances = apply_role_profiles(raw_utterances, profiles)
    console.print(f"[green]✓[/green] Speaker roles assigned: {[p.display_label for p in profiles]}")

    # ── Step 7: Overlap detection ─────────────────────────────────────────────
    if config.enable_overlap_detection and hasattr(engine, "_diarizer"):
        # Use diarization segments directly for overlap detection if available
        try:
            segments = getattr(engine, "_last_segments", [])
            if segments:
                overlap_regions = detect_overlaps(segments)
                utterances = mark_utterance_overlaps(utterances, overlap_regions)
        except Exception:
            pass  # Non-fatal

    # ── Step 8: Quality metrics ───────────────────────────────────────────────
    quality_report = compute_quality_metrics(
        utterances,
        high_threshold=config.high_confidence_threshold,
        low_threshold=config.low_confidence_threshold,
    )

    # ── Step 9: Speaker analytics ─────────────────────────────────────────────
    speaker_stats = compute_speaker_stats(utterances, audio_duration=audio_duration)

    # ── Step 10: Clinical summary (optional) ──────────────────────────────────
    clinical_summary_obj = None
    if config.enable_clinical_summary:
        with console.status("[bold green]Extracting clinical information…"):
            from nlp.clinical_summary import extract_clinical_summary
            clinical_summary_obj = extract_clinical_summary(utterances)

    # ── Step 11: Build engine info metadata ───────────────────────────────────
    engine_info = {
        "diarization": getattr(engine, "ENGINE_NAME", config.engine),
        "asr": config.asr_engine,
        "asr_model": config.whisper_model,
        "mode": getattr(engine, "MODE", "Unknown"),
    }
    audio_metadata = {
        "duration_seconds": audio_duration,
        "sample_rate": 16000,
        "channels": 1,
        "format": "wav",
        "original_filename": audio_path.name,
    }
    metadata = {
        "case_id": enc_id,
        "audio_file": audio_path.name,
        "duration_seconds": audio_duration,
        "engine": config.engine,
        "diarization_model": engine_info["diarization"],
        "asr_model": config.whisper_model,
        "language": ",".join(languages_detected) if languages_detected else "en",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    # Primary: Module 1 → Module 3 canonical JSON
    timeline_path = config.output_dir / f"{stem}_conversation.json"
    with console.status("[bold green]Writing conversation timeline…"):
        write_conversation_timeline(
            utterances,
            timeline_path,
            encounter_id=enc_id,
            audio_ref=str(audio_path),
            capture_mode="batch",
            consent_logged=config.consent_logged,
            engine_info=engine_info,
            audio_metadata=audio_metadata,
            speaker_profiles=profiles,
            language_tags=language_tags,
            languages_detected=languages_detected,
            quality_report=quality_report,
            speaker_stats=speaker_stats,
            clinical_summary=clinical_summary_obj,
        )

    # Secondary: Enhanced JSON (for review tools)
    json_path = config.output_dir / f"{stem}_transcript.json"
    with console.status("[bold green]Writing enhanced JSON…"):
        write_enhanced_json(
            utterances,
            json_path,
            metadata=metadata,
            speaker_stats=speaker_stats,
            quality_report=quality_report,
            clinical_summary=clinical_summary_obj,
        )

    # TXT
    txt_path = config.output_dir / f"{stem}_transcript.txt"
    with console.status("[bold green]Writing TXT transcript…"):
        write_enhanced_txt(
            utterances,
            txt_path,
            metadata=metadata,
            speaker_stats=speaker_stats,
            quality_report=quality_report,
        )

    # PDF
    pdf_path = config.output_dir / f"{stem}_transcript.pdf"
    with console.status("[bold green]Generating PDF report…"):
        try:
            write_clinical_pdf(
                utterances,
                pdf_path,
                metadata=metadata,
                speaker_stats=speaker_stats,
                quality_report=quality_report,
                clinical_summary=clinical_summary_obj,
            )
        except Exception as exc:
            console.print(f"[yellow]PDF warning:[/yellow] {exc}")
            pdf_path = None

    # Print transcript preview
    _print_transcript_table(utterances)
    _print_quality_summary(quality_report, speaker_stats)

    # Done
    outputs = [
        f"  [bold]Conversation JSON[/bold] → [cyan]{timeline_path}[/cyan]  [dim](Module 1→3 contract)[/dim]",
        f"  Enhanced JSON     → [cyan]{json_path}[/cyan]",
        f"  TXT Transcript    → [cyan]{txt_path}[/cyan]",
    ]
    if pdf_path:
        outputs.append(f"  PDF Report        → [cyan]{pdf_path}[/cyan]")

    console.print(
        Panel(
            f"[bold green]Transcription complete![/bold green]\n\n"
            + "\n".join(outputs),
            title=f"Output — {enc_id}",
            border_style="green",
        )
    )


# ── Evaluation mode ────────────────────────────────────────────────────────────

def _run_evaluate(config, max_cases: Optional[int] = None) -> None:
    """Run DER + WER evaluation against test_data/ ground-truth labels."""
    import evaluate as eval_module

    console.print("\n[bold]Running evaluation against test_data/...[/bold]")
    test_data_dir = Path(__file__).parent / "test_data"

    if not test_data_dir.exists() or not any(test_data_dir.iterdir()):
        console.print(
            "[yellow]Warning:[/yellow] test_data/ is empty or does not exist.\n"
            "Add labeled audio files and annotations before running --evaluate.\n"
            "See README.md for the expected folder structure."
        )
        sys.exit(0)

    if max_cases is None:
        console.print(
            "[yellow]Tip:[/yellow] Running on all available files. "
            "Use [bold]--max-cases N[/bold] for a quick spot-check."
        )

    try:
        from transcriber import create_engine
        engine = create_engine(config.engine, config)
        eval_module.run_evaluation(engine, test_data_dir, config.output_dir, max_cases=max_cases)
    except Exception as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        raise


# ── Benchmark mode ─────────────────────────────────────────────────────────────

def _run_benchmark(audio_str: Optional[str], config) -> None:
    """
    Run a benchmark comparison of all available engines.

    Runs the same audio through Community-1, Sortformer (if available),
    and AssemblyAI (if API key is set), then produces a comparison report.
    """
    console.print("\n[bold]Running engine benchmark…[/bold]")

    if not audio_str:
        # Look for a sample file in test_data/
        test_data_dir = Path(__file__).parent / "test_data"
        sample_files = list(test_data_dir.glob("**/*.wav"))[:1]
        if not sample_files:
            console.print(
                "[bold red]Error:[/bold red] --benchmark requires --audio or a WAV file in test_data/\n"
                "Example: python main.py --benchmark --audio sample.mp3"
            )
            sys.exit(1)
        audio_str = str(sample_files[0])
        console.print(f"[dim]Using test file: {audio_str}[/dim]")

    from audio.preprocessing import validate_audio_file, normalize_to_wav, get_audio_duration
    from transcriber import create_engine

    try:
        audio_path = validate_audio_file(audio_str)
    except ValueError as exc:
        console.print(f"[bold red]Audio error:[/bold red] {exc}")
        sys.exit(1)

    with console.status("[bold green]Converting audio…"):
        wav_path = normalize_to_wav(audio_path, config.output_dir)

    duration = get_audio_duration(wav_path) or 0.0

    # Which engines to benchmark
    candidates: list[dict] = [
        {"name": "community-whisperx", "requires": None},
        {"name": "assemblyai",         "requires": "assemblyai_api_key"},
        {"name": "sortformer-whisper", "requires": "sortformer"},
    ]

    results = []
    for candidate in candidates:
        engine_name = candidate["name"]
        req = candidate.get("requires")

        # Skip if prerequisites not met
        if req == "assemblyai_api_key" and not config.assemblyai_api_key:
            console.print(f"[dim]Skipping {engine_name}: no API key[/dim]")
            continue
        if req == "sortformer":
            try:
                import nemo  # noqa: F401
            except ImportError:
                console.print(f"[dim]Skipping {engine_name}: NeMo not installed[/dim]")
                continue

        console.print(f"\n[bold cyan]▶ Benchmarking: {engine_name}[/bold cyan]")

        import time
        cfg_copy = type(config).__new__(type(config))
        cfg_copy.__dict__.update(config.__dict__)
        cfg_copy.engine = engine_name  # type: ignore

        try:
            eng = create_engine(engine_name, cfg_copy)
            t0 = time.perf_counter()
            utterances = eng.transcribe(wav_path)
            elapsed = time.perf_counter() - t0
            rtf = elapsed / max(duration, 0.001)  # Real-time factor

            from analytics.quality_metrics import compute_quality_metrics
            qr = compute_quality_metrics(utterances)

            results.append({
                "engine": engine_name,
                "utterances": len(utterances),
                "elapsed_s": round(elapsed, 2),
                "rtf": round(rtf, 3),
                "avg_confidence": round(qr.average_confidence, 3) if qr.average_confidence else None,
                "low_conf_count": qr.low_confidence_count,
                "status": "ok",
            })
            console.print(
                f"  ✓ {len(utterances)} utterances | {elapsed:.1f}s | RTF={rtf:.2f}x | "
                f"Avg conf={qr.average_confidence:.2f}" if qr.average_confidence else
                f"  ✓ {len(utterances)} utterances | {elapsed:.1f}s | RTF={rtf:.2f}x"
            )
        except Exception as exc:
            results.append({
                "engine": engine_name,
                "status": "error",
                "error": str(exc),
            })
            console.print(f"  ✗ Failed: {exc}")

    # Write benchmark report
    import json
    report = {
        "audio_file": str(audio_path),
        "duration_seconds": duration,
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = config.output_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print comparison table
    table = Table(title="Engine Benchmark", box=box.ROUNDED, border_style="cyan")
    table.add_column("Engine", style="bold")
    table.add_column("Utterances", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Avg Conf", justify="right")
    table.add_column("Status")

    for r in results:
        if r["status"] == "ok":
            table.add_row(
                r["engine"],
                str(r.get("utterances", "-")),
                str(r.get("elapsed_s", "-")),
                str(r.get("rtf", "-")),
                str(r.get("avg_confidence", "-")) if r.get("avg_confidence") else "-",
                "[green]✓ OK[/green]",
            )
        else:
            table.add_row(r["engine"], "-", "-", "-", "-", "[red]✗ Error[/red]")

    console.print(table)
    console.print(f"\n[dim]Full report: {report_path}[/dim]")


# ── Pretty output helpers ──────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print(
        Panel(
            "[bold cyan]Clinical Conversation Intelligence — Module 1[/bold cyan]\n"
            "[dim]Diarization | ASR | Speaker Role Mapping | EHR Timeline[/dim]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _print_transcript_table(utterances) -> None:
    from outputs.txt_writer import format_timestamp

    # Role color map
    ROLE_COLORS = {
        "Doctor": "bold blue",
        "Patient": "bold green",
        "Attendant": "bold yellow",
        "Nurse": "bold magenta",
        "Interpreter": "bold cyan",
        "Second Clinician": "bold blue",
        "Unknown": "dim",
    }

    table = Table(
        title="Transcript Preview",
        box=box.ROUNDED,
        show_lines=True,
        border_style="cyan",
    )
    table.add_column("Time", style="dim", width=10)
    table.add_column("Speaker", width=16)
    table.add_column("Text", ratio=1)
    table.add_column("Conf.", width=6, justify="right")

    # Show at most 30 rows in preview
    preview = utterances[:30]
    for u in preview:
        color = ROLE_COLORS.get(u.speaker_label, "white")
        conf_str = f"{u.confidence:.2f}" if u.confidence is not None else "—"
        review_icon = " ⚠" if u.review_required else ""
        table.add_row(
            format_timestamp(u.start_time),
            f"[{color}]{u.speaker_label}[/{color}]",
            u.text + review_icon,
            conf_str,
        )

    if len(utterances) > 30:
        table.add_row("…", "…", f"(+{len(utterances) - 30} more utterances)", "")

    console.print()
    console.print(table)


def _print_quality_summary(quality_report, speaker_stats) -> None:
    """Print a brief quality and analytics summary."""
    if not quality_report:
        return

    lines = []
    if quality_report.average_confidence is not None:
        lines.append(f"  Avg confidence : {quality_report.average_confidence * 100:.1f}%")
    lines.append(f"  Review needed  : {quality_report.review_required_count} utterance(s)")
    lines.append(f"  Overlaps       : {quality_report.overlap_count}")

    if speaker_stats and speaker_stats.speakers:
        lines.append("")
        for s in speaker_stats.speakers:
            lines.append(
                f"  {s.role:<18}: {s.total_speech_seconds:.1f}s "
                f"({s.speech_percentage:.0f}%) | {s.turn_count} turns"
            )

    console.print(
        Panel("\n".join(lines), title="Quality & Analytics", border_style="dim")
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
