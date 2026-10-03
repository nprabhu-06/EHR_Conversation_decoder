"""
analytics/speaker_stats.py — Speaker-level conversation analytics.

Computes objective, measurable statistics about the consultation.

IMPORTANT DISCLAIMER:
  These metrics describe observable conversation patterns only.
  Do NOT make psychological, clinical, or medical conclusions from them.
  Metrics such as "Doctor speaking time" reflect what was transcribed,
  not clinical quality or patient outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


@dataclass
class SpeakerStats:
    """Statistics for a single speaker in the consultation."""
    speaker_id: str
    role: str                      # Doctor | Patient
    total_speech_seconds: float
    speech_percentage: float
    turn_count: int
    avg_turn_duration: float
    longest_turn_seconds: float
    longest_turn_utterance_id: Optional[int]


@dataclass
class ConsultationStats:
    """Full consultation analytics."""
    total_duration_seconds: float
    total_speech_seconds: float
    silence_seconds: float

    speakers: list[SpeakerStats] = field(default_factory=list)

    overlap_count: int = 0
    total_overlap_seconds: float = 0.0

    turn_count_total: int = 0
    avg_turn_duration: float = 0.0

    # Possible interruptions = utterances that start within 0.5s of another speaker ending
    possible_interruption_count: int = 0

    def to_dict(self) -> dict:
        return {
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "total_speech_seconds": round(self.total_speech_seconds, 2),
            "silence_seconds": round(self.silence_seconds, 2),
            "overlap_count": self.overlap_count,
            "total_overlap_seconds": round(self.total_overlap_seconds, 2),
            "possible_interruption_count": self.possible_interruption_count,
            "speakers": [
                {
                    "speaker_id": s.speaker_id,
                    "role": s.role,
                    "total_speech_seconds": round(s.total_speech_seconds, 2),
                    "speech_percentage": round(s.speech_percentage, 1),
                    "turn_count": s.turn_count,
                    "avg_turn_duration": round(s.avg_turn_duration, 2),
                    "longest_turn_seconds": round(s.longest_turn_seconds, 2),
                    "longest_turn_utterance_id": s.longest_turn_utterance_id,
                }
                for s in self.speakers
            ],
        }


def compute_speaker_stats(
    utterances: list["Utterance"],
    audio_duration: Optional[float] = None,
    interruption_gap_sec: float = 0.5,
) -> ConsultationStats:
    """
    Compute comprehensive speaker statistics from utterances.

    Args:
        utterances:          Role-labeled utterances.
        audio_duration:      Total audio duration (seconds). Auto-estimated if None.
        interruption_gap_sec: Max gap between turns to flag as possible interruption.

    Returns:
        ConsultationStats object.
    """
    if not utterances:
        return ConsultationStats(
            total_duration_seconds=audio_duration or 0.0,
            total_speech_seconds=0.0,
            silence_seconds=audio_duration or 0.0,
        )

    # Estimate total duration if not provided
    if audio_duration is None:
        audio_duration = max(u.end_time for u in utterances)

    # Per-speaker aggregation
    by_role: dict[str, dict] = {}
    for u in utterances:
        role = u.speaker_label
        sid = u.speaker_id
        if role not in by_role:
            by_role[role] = {
                "speaker_id": sid,
                "role": role,
                "total_seconds": 0.0,
                "turns": [],
                "turn_ids": [],
            }
        dur = u.end_time - u.start_time
        by_role[role]["total_seconds"] += max(0.0, dur)
        by_role[role]["turns"].append(max(0.0, dur))
        by_role[role]["turn_ids"].append(u.id)

    total_speech = sum(r["total_seconds"] for r in by_role.values())
    overlap_utterances = [u for u in utterances if u.overlap]
    overlap_seconds = sum(max(0.0, u.end_time - u.start_time) for u in overlap_utterances)

    # Build per-speaker stats
    speaker_stats: list[SpeakerStats] = []
    for role, data in by_role.items():
        turns = data["turns"]
        turn_count = len(turns)
        avg_dur = sum(turns) / turn_count if turn_count else 0.0
        max_dur_idx = turns.index(max(turns)) if turns else 0
        max_dur = max(turns) if turns else 0.0
        max_uid = data["turn_ids"][max_dur_idx] if data["turn_ids"] else None

        pct = (data["total_seconds"] / total_speech * 100) if total_speech > 0 else 0.0

        speaker_stats.append(SpeakerStats(
            speaker_id=data["speaker_id"],
            role=role,
            total_speech_seconds=data["total_seconds"],
            speech_percentage=pct,
            turn_count=turn_count,
            avg_turn_duration=avg_dur,
            longest_turn_seconds=max_dur,
            longest_turn_utterance_id=max_uid,
        ))

    # Possible interruptions
    interruptions = 0
    sorted_utts = sorted(utterances, key=lambda u: u.start_time)
    for i in range(1, len(sorted_utts)):
        prev = sorted_utts[i - 1]
        curr = sorted_utts[i]
        if (
            prev.speaker_label != curr.speaker_label
            and curr.start_time - prev.end_time < interruption_gap_sec
            and curr.start_time > prev.start_time  # Not an overlap
        ):
            interruptions += 1

    return ConsultationStats(
        total_duration_seconds=audio_duration,
        total_speech_seconds=total_speech,
        silence_seconds=max(0.0, audio_duration - total_speech),
        speakers=speaker_stats,
        overlap_count=len(overlap_utterances),
        total_overlap_seconds=overlap_seconds,
        turn_count_total=sum(s.turn_count for s in speaker_stats),
        avg_turn_duration=(total_speech / len(utterances)) if utterances else 0.0,
        possible_interruption_count=interruptions,
    )
