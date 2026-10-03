"""
diarization/overlap.py — Explicit overlapping speech detection.

Detects regions where two or more speakers are simultaneously active
in the diarization output.

Example:
    Doctor:  10.20 → 13.00
    Patient: 12.40 → 14.10
    Overlap: 12.40 → 13.00

Overlapping speech is flagged explicitly rather than forced into one speaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import SpeakerSegment


@dataclass
class OverlapRegion:
    """A region where multiple speakers are simultaneously active."""
    start_time: float
    end_time: float
    speakers: list[str] = field(default_factory=list)
    overlap: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def detect_overlaps(
    segments: list["SpeakerSegment"],
    min_duration: float = 0.05,
) -> list[OverlapRegion]:
    """
    Detect all regions where speaker segments from different speakers overlap.

    Args:
        segments:     Diarization speaker segments.
        min_duration: Minimum overlap duration to report (seconds).
                      Very short overlaps (<50ms) are often boundary artifacts.

    Returns:
        List of OverlapRegion objects sorted by start_time.
    """
    if len(segments) < 2:
        return []

    overlaps: list[OverlapRegion] = []

    # Compare every pair of segments
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1:]:
            # Quick skip: if A ends before B starts, no more overlaps possible
            if seg_a.end_time <= seg_b.start_time:
                break

            # Check if different speakers
            if seg_a.speaker_id == seg_b.speaker_id:
                continue

            # Compute overlap interval
            overlap_start = max(seg_a.start_time, seg_b.start_time)
            overlap_end = min(seg_a.end_time, seg_b.end_time)
            duration = overlap_end - overlap_start

            if duration >= min_duration:
                overlaps.append(OverlapRegion(
                    start_time=overlap_start,
                    end_time=overlap_end,
                    speakers=[seg_a.speaker_id, seg_b.speaker_id],
                ))

    # Sort by start time
    overlaps.sort(key=lambda o: o.start_time)

    if overlaps:
        total_dur = sum(o.duration for o in overlaps)
        print(
            f"[overlap] Detected {len(overlaps)} overlap region(s) "
            f"totaling {total_dur:.2f}s"
        )
    return overlaps


def mark_utterance_overlaps(
    utterances: list,
    overlap_regions: list[OverlapRegion],
    tolerance: float = 0.1,
) -> list:
    """
    Mark utterances that fall within detected overlap regions.

    Sets utterance.overlap = True and populates utterance.overlapping_speakers.

    Args:
        utterances:      List of Utterance objects.
        overlap_regions: From detect_overlaps().
        tolerance:       Extra seconds around overlap region boundaries.

    Returns:
        Modified utterances (in-place modification + returned).
    """
    from engines.base import Utterance

    marked = []
    for u in utterances:
        u_start = u.start_time - tolerance
        u_end = u.end_time + tolerance
        overlapping_with = []

        for region in overlap_regions:
            # Does the utterance fall within this overlap region?
            if region.start_time < u_end and region.end_time > u_start:
                for spk in region.speakers:
                    if spk != u.speaker_id and spk not in overlapping_with:
                        overlapping_with.append(spk)

        if overlapping_with:
            marked.append(Utterance(
                id=u.id,
                speaker_id=u.speaker_id,
                speaker_label=u.speaker_label,
                text=u.text,
                raw_text=u.raw_text,
                normalized_text=u.normalized_text,
                start_time=u.start_time,
                end_time=u.end_time,
                confidence=u.confidence,
                overlap=True,
                overlapping_speakers=overlapping_with,
                words=u.words,
                review_required=u.review_required,
            ))
        else:
            marked.append(u)

    return marked
