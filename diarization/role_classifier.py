"""
diarization/role_classifier.py — Multi-speaker role classification.

Assigns clinical roles to diarized speakers beyond the legacy
Doctor / Patient binary.

Supported roles:
  doctor
  patient
  attendant       (caregiver / family member)
  nurse           (clinical assistant)
  interpreter     (language interpreter)
  second_clinician
  other
  unknown         (uncertain — never force assignment)

IMPORTANT DISCLAIMERS:
  - Roles are NOT inferred from gender, age, accent, or appearance.
  - Role confidence is always reported.
  - A speaker CAN remain "unknown" if classification is uncertain.
  - The first_speaker heuristic is kept for backward compatibility
    but should not be assumed to be reliable.

Strategies:
  1. first_speaker — first speaker heard = Doctor (heuristic only)
  2. manual        — user explicitly specifies role assignments
  3. interactive   — Rich terminal prompt for user confirmation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance

# ── Supported roles ───────────────────────────────────────────────────────────

VALID_ROLES = {
    "doctor",
    "patient",
    "attendant",
    "nurse",
    "interpreter",
    "second_clinician",
    "other",
    "unknown",
}

# Human-readable display labels
ROLE_DISPLAY = {
    "doctor": "Doctor",
    "patient": "Patient",
    "attendant": "Attendant",
    "nurse": "Nurse",
    "interpreter": "Interpreter",
    "second_clinician": "Second Clinician",
    "other": "Other",
    "unknown": "Unknown",
}


@dataclass
class SpeakerProfile:
    """Role assignment for a single diarized speaker."""
    speaker_id: str
    role: str                         # One of VALID_ROLES
    role_confidence: float            # 0.0–1.0; how certain the assignment is
    assignment_method: str            # "first_speaker" | "manual" | "interactive" | "auto"
    display_label: str = ""           # Human-readable: "Doctor", "Patient", etc.

    def __post_init__(self):
        if not self.display_label:
            self.display_label = ROLE_DISPLAY.get(self.role, self.role.capitalize())


# ── Role assignment functions ─────────────────────────────────────────────────

def assign_roles_first_speaker(
    utterances: list["Utterance"],
) -> list[SpeakerProfile]:
    """
    Heuristic: first speaker heard = Doctor, second = Patient, rest = Unknown.

    This is a HEURISTIC only. It does NOT prove clinical role.
    Role confidence is intentionally set below 1.0 to reflect uncertainty.
    """
    if not utterances:
        return []

    speaker_order: list[str] = []
    seen: set[str] = set()
    for u in sorted(utterances, key=lambda x: x.start_time):
        if u.speaker_id not in seen:
            speaker_order.append(u.speaker_id)
            seen.add(u.speaker_id)

    profiles: list[SpeakerProfile] = []
    for i, spk_id in enumerate(speaker_order):
        if i == 0:
            role, conf = "doctor", 0.70   # Heuristic — not certain
        elif i == 1:
            role, conf = "patient", 0.65
        else:
            role, conf = "unknown", 0.0   # Extra speakers → unknown

        profiles.append(SpeakerProfile(
            speaker_id=spk_id,
            role=role,
            role_confidence=conf,
            assignment_method="first_speaker",
        ))

    return profiles


def assign_roles_manual(
    utterances: list["Utterance"],
    role_map: dict[str, str],  # {speaker_id: role}
) -> list[SpeakerProfile]:
    """
    Assign roles from a user-provided mapping.

    Args:
        utterances: List of Utterance objects.
        role_map:   Explicit mapping of speaker_id → role string.
                    Example: {"SPEAKER_00": "doctor", "SPEAKER_01": "patient"}

    Unknown speakers not in role_map get role="unknown".
    """
    all_speakers = sorted(set(u.speaker_id for u in utterances))
    profiles = []
    for spk_id in all_speakers:
        raw_role = role_map.get(spk_id, "unknown").lower().strip()
        role = raw_role if raw_role in VALID_ROLES else "unknown"
        conf = 0.99 if role != "unknown" else 0.0
        profiles.append(SpeakerProfile(
            speaker_id=spk_id,
            role=role,
            role_confidence=conf,
            assignment_method="manual",
        ))
    return profiles


def assign_roles_interactive(
    utterances: list["Utterance"],
) -> list[SpeakerProfile]:
    """
    Interactive Rich terminal prompt for role assignment.

    Shows a preview of each speaker (first heard time, turn count, sample text)
    and asks the user to assign a role from the supported list.

    Returns list of SpeakerProfile with user-confirmed roles.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.prompt import Prompt
        from rich import box
        console = Console()
        rich_available = True
    except ImportError:
        rich_available = False

    # Build speaker summary
    speaker_info: dict[str, dict] = {}
    for u in sorted(utterances, key=lambda x: x.start_time):
        sid = u.speaker_id
        if sid not in speaker_info:
            speaker_info[sid] = {
                "first_time": u.start_time,
                "turn_count": 0,
                "sample": u.text[:80],
                "total_seconds": 0.0,
            }
        speaker_info[sid]["turn_count"] += 1
        speaker_info[sid]["total_seconds"] += max(0.0, u.end_time - u.start_time)

    speakers_sorted = sorted(speaker_info.items(), key=lambda x: x[1]["first_time"])
    all_speaker_ids = [s[0] for s in speakers_sorted]

    if rich_available:
        console.print()
        console.print(
            "[bold cyan]── Multi-Speaker Role Assignment ──────────────────────────────[/bold cyan]"
        )
        console.print(
            "[dim]Detected speakers are listed below. Assign a clinical role to each.[/dim]\n"
        )

        table = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
        table.add_column("Speaker ID", style="bold")
        table.add_column("First heard")
        table.add_column("Turns")
        table.add_column("Speech (s)")
        table.add_column("Sample text")

        for sid, info in speakers_sorted:
            from utils import format_timestamp as fmt_ts
            try:
                ts = fmt_ts(info["first_time"])
            except Exception:
                ts = f"{info['first_time']:.1f}s"
            table.add_row(
                sid,
                ts,
                str(info["turn_count"]),
                f"{info['total_seconds']:.1f}",
                (info["sample"] or "")[:60] + ("…" if len(info.get("sample", "")) > 60 else ""),
            )
        console.print(table)

        console.print(
            "\n[yellow]NOTE:[/yellow] Role assignment is based ONLY on your selection.\n"
            "The system does NOT infer roles from voice, gender, age, or accent.\n"
        )

        role_choices = list(VALID_ROLES)
        profiles = []

        for sid in all_speaker_ids:
            first = "doctor" if sid == speakers_sorted[0][0] else "unknown"
            while True:
                console.print(
                    f"\nAssign role to [bold]{sid}[/bold] "
                    f"(spoke first at {speaker_info[sid]['first_time']:.1f}s):"
                )
                console.print(
                    f"  Valid roles: {', '.join(sorted(VALID_ROLES))}"
                )
                choice = Prompt.ask(
                    f"  Role for [bold]{sid}[/bold]",
                    default=first,
                )
                choice = choice.lower().strip()
                if choice in VALID_ROLES:
                    conf = 0.98 if choice != "unknown" else 0.0
                    profiles.append(SpeakerProfile(
                        speaker_id=sid,
                        role=choice,
                        role_confidence=conf,
                        assignment_method="interactive",
                    ))
                    console.print(
                        f"  [green]✓[/green] {sid} → [bold]{ROLE_DISPLAY.get(choice, choice)}[/bold]"
                    )
                    break
                else:
                    console.print(
                        f"  [red]Invalid role '{choice}'.[/red] "
                        f"Choose from: {', '.join(sorted(VALID_ROLES))}"
                    )

        return profiles

    else:
        # Plain terminal fallback
        print("\n── Multi-Speaker Role Assignment ──")
        print(f"Detected speakers: {', '.join(all_speaker_ids)}")
        print(f"Valid roles: {', '.join(sorted(VALID_ROLES))}")

        profiles = []
        for sid in all_speaker_ids:
            while True:
                default = "doctor" if sid == all_speaker_ids[0] else "unknown"
                role = input(f"Role for {sid} [{default}]: ").strip().lower() or default
                if role in VALID_ROLES:
                    conf = 0.98 if role != "unknown" else 0.0
                    profiles.append(SpeakerProfile(
                        speaker_id=sid,
                        role=role,
                        role_confidence=conf,
                        assignment_method="interactive",
                    ))
                    break
                print(f"Invalid. Choose from: {', '.join(sorted(VALID_ROLES))}")
        return profiles


# ── Apply role profiles to utterances ────────────────────────────────────────

def apply_role_profiles(
    utterances: list["Utterance"],
    profiles: list[SpeakerProfile],
) -> list["Utterance"]:
    """
    Apply SpeakerProfile role assignments to utterances.

    Updates speaker_label to the display label (Doctor, Patient, etc.)
    from each profile.

    Args:
        utterances: Input utterances with raw speaker_id labels.
        profiles:   Role profiles from any assignment strategy.

    Returns:
        New list of Utterances with speaker_label updated to role display name.
    """
    from engines.base import Utterance as Utt

    profile_map = {p.speaker_id: p for p in profiles}

    result = []
    for u in utterances:
        profile = profile_map.get(u.speaker_id)
        if profile:
            new_label = profile.display_label
        else:
            new_label = "Unknown"

        result.append(Utt(
            id=u.id,
            speaker_id=u.speaker_id,
            speaker_label=new_label,
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


# ── Profiles to dict (for JSON output) ───────────────────────────────────────

def profiles_to_dict(profiles: list[SpeakerProfile]) -> list[dict]:
    """Convert list of SpeakerProfile to JSON-serializable list of dicts."""
    return [
        {
            "speaker_id": p.speaker_id,
            "role": p.role,
            "role_confidence": round(p.role_confidence, 3),
            "display_label": p.display_label,
            "assignment_method": p.assignment_method,
        }
        for p in profiles
    ]
