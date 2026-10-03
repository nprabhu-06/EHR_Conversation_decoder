"""
diarization/speaker_mapping.py — Doctor/Patient role assignment strategies.

Supports three strategies:

1. first_speaker (default):
   The speaker heard first (earliest start_time) is mapped to "Doctor".
   Rationale: clinicians typically open consultations.
   IMPORTANT: This is a heuristic — it does NOT prove who is the doctor.

2. manual:
   The user specifies the raw diarization label for the doctor
   via --doctor-speaker SPEAKER_A (CLI) or doctor_speaker field (API).

3. interactive:
   Shows a Rich-based interactive prompt in the terminal.
   The user confirms or adjusts the Doctor/Patient assignment.
   Suitable for CLI mode.

IMPORTANT DISCLAIMER:
   This system does NOT infer speaker roles from voice characteristics,
   gender, age, accent, or any biometric property.
   Role assignment is based ONLY on temporal position (first_speaker)
   or explicit user confirmation (manual / interactive).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


# ── Core mapping logic ─────────────────────────────────────────────────────────

def apply_role_mapping(
    utterances: list["Utterance"],
    strategy: str = "first_speaker",
    doctor_speaker: Optional[str] = None,
) -> list["Utterance"]:
    """
    Apply Doctor/Patient role mapping to a list of Utterances.

    Args:
        utterances:      List of utterances with raw speaker_id labels.
        strategy:        One of: first_speaker | manual | interactive
        doctor_speaker:  Raw label for the doctor (used in manual strategy).

    Returns:
        New list of Utterances with speaker_label set to Doctor/Patient.
    """
    from engines.base import Utterance as Utt

    if not utterances:
        return []

    strategy = strategy.lower().strip()

    if strategy == "manual":
        if not doctor_speaker:
            raise ValueError(
                "Strategy 'manual' requires doctor_speaker to be set.\n"
                "Use --doctor-speaker SPEAKER_A (or SPEAKER_00) on the CLI."
            )
        doctor_raw = doctor_speaker

    elif strategy == "interactive":
        doctor_raw = _interactive_mapping(utterances)

    else:
        # first_speaker (default) — earliest utterance speaker is Doctor
        if strategy not in ("first_speaker", "auto"):
            print(
                f"[speaker_mapping] Unknown strategy {strategy!r}. "
                "Falling back to first_speaker."
            )
        doctor_raw = _first_speaker(utterances)
        print(
            f"[speaker_mapping] first_speaker heuristic → "
            f"'{doctor_raw}' assigned as Doctor. "
            "(Override with --doctor-speaker or SPEAKER_MAPPING_STRATEGY=manual)"
        )

    return _remap(utterances, doctor_raw)


def _first_speaker(utterances: list) -> str:
    """Return the speaker_id of the utterance with the smallest start_time."""
    return min(utterances, key=lambda u: u.start_time).speaker_id


def _remap(utterances: list, doctor_raw: str) -> list:
    """Apply the label_map to utterances and return new Utterance objects."""
    from engines.base import Utterance as Utt

    all_labels = sorted(set(u.speaker_id for u in utterances))
    non_doctor = [lbl for lbl in all_labels if lbl != doctor_raw]
    label_map = {doctor_raw: "Doctor"}
    for lbl in non_doctor:
        label_map[lbl] = "Patient"

    if len(all_labels) > 2:
        print(
            f"[speaker_mapping] WARNING: {len(all_labels)} speaker(s) detected. "
            "Only 2-speaker consultations are fully supported. "
            "All speakers except the Doctor will be labelled 'Patient'."
        )

    result = []
    for u in utterances:
        role = label_map.get(u.speaker_id, "Patient")
        result.append(Utt(
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


def _interactive_mapping(utterances: list) -> str:
    """
    Show an interactive Rich terminal prompt for speaker role assignment.

    Displays a preview of detected speakers and asks the user to confirm
    which raw label corresponds to the Doctor.

    Returns the confirmed raw speaker label for the Doctor.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from rich.prompt import Prompt
        console = Console()
    except ImportError:
        # Fallback to plain input if rich is unavailable
        return _plain_interactive_mapping(utterances)

    # Build speaker summary
    speaker_stats: dict[str, dict] = {}
    for u in utterances:
        sid = u.speaker_id
        if sid not in speaker_stats:
            speaker_stats[sid] = {"first_time": u.start_time, "count": 0, "sample": u.text}
        speaker_stats[sid]["count"] += 1

    console.print()
    console.print(
        "[bold cyan]── Interactive Speaker Mapping ──────────────────────────────────[/bold cyan]"
    )
    console.print(
        "[dim]Detected speakers are listed below. Please confirm which is the Doctor.[/dim]\n"
    )

    table = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
    table.add_column("Speaker Label", style="bold")
    table.add_column("First heard at")
    table.add_column("Turns")
    table.add_column("Sample text (first utterance)")

    speakers_sorted = sorted(speaker_stats.items(), key=lambda x: x[1]["first_time"])
    for sid, stats in speakers_sorted:
        from utils import format_timestamp
        table.add_row(
            sid,
            format_timestamp(stats["first_time"]),
            str(stats["count"]),
            (stats["sample"] or "")[:60] + ("…" if len(stats.get("sample", "")) > 60 else ""),
        )
    console.print(table)

    # Show first-speaker suggestion
    first = speakers_sorted[0][0] if speakers_sorted else list(speaker_stats.keys())[0]
    all_ids = [s[0] for s in speakers_sorted]

    console.print(
        f"\n[dim]Heuristic suggestion: [bold]{first}[/bold] (spoke first).[/dim]"
    )
    console.print(
        "[yellow]NOTE:[/yellow] Role assignment is based solely on your selection. "
        "The system does not infer roles from voice characteristics.\n"
    )

    while True:
        choice = Prompt.ask(
            f"Which label is the [bold blue]Doctor[/bold blue]?",
            choices=all_ids,
            default=first,
        )
        if choice in speaker_stats:
            console.print(
                f"\n[green]✓[/green] '{choice}' will be assigned as [bold]Doctor[/bold]. "
                f"All other speakers → [bold]Patient[/bold].\n"
            )
            return choice
        console.print(f"[red]Invalid choice.[/red] Please select one of: {all_ids}")


def _plain_interactive_mapping(utterances: list) -> str:
    """Fallback interactive mapping without rich."""
    all_ids = sorted(set(u.speaker_id for u in utterances))
    first = min(utterances, key=lambda u: u.start_time).speaker_id

    print("\n── Interactive Speaker Mapping ──")
    print(f"Detected speakers: {', '.join(all_ids)}")
    print(f"Suggestion: {first} (spoke first)")
    while True:
        choice = input(f"Enter Doctor speaker label [{first}]: ").strip() or first
        if choice in all_ids:
            return choice
        print(f"Invalid. Choose from: {all_ids}")
