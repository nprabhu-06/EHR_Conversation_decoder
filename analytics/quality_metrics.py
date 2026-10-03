"""
analytics/quality_metrics.py — Transcript quality assessment.

Classifies transcription confidence into tiers and identifies
segments that warrant clinical review.

Thresholds (configurable via .env):
    HIGH:   >= HIGH_CONFIDENCE_THRESHOLD (default 0.90)
    MEDIUM: >= LOW_CONFIDENCE_THRESHOLD and < HIGH (default 0.70-0.89)
    LOW:    < LOW_CONFIDENCE_THRESHOLD (default 0.70)

IMPORTANT: Confidence scores come from the ASR model.
They indicate transcription uncertainty, NOT clinical significance.
A low-confidence segment may be perfectly correct — it simply warrants review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance


@dataclass
class QualityReport:
    """Transcript quality summary."""
    total_utterances: int
    utterances_with_confidence: int

    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    unknown_confidence_count: int

    average_confidence: Optional[float]
    min_confidence: Optional[float]
    max_confidence: Optional[float]

    review_required_count: int
    overlap_count: int

    high_threshold: float
    low_threshold: float

    low_confidence_utterance_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_utterances": self.total_utterances,
            "average_confidence": round(self.average_confidence, 4) if self.average_confidence is not None else None,
            "min_confidence": round(self.min_confidence, 4) if self.min_confidence is not None else None,
            "max_confidence": round(self.max_confidence, 4) if self.max_confidence is not None else None,
            "high_confidence_count": self.high_confidence_count,
            "medium_confidence_count": self.medium_confidence_count,
            "low_confidence_count": self.low_confidence_count,
            "unknown_confidence_count": self.unknown_confidence_count,
            "review_required_count": self.review_required_count,
            "overlap_count": self.overlap_count,
            "low_confidence_utterance_ids": self.low_confidence_utterance_ids,
            "thresholds": {
                "high": self.high_threshold,
                "low": self.low_threshold,
            },
        }


def compute_quality_metrics(
    utterances: list["Utterance"],
    high_threshold: float = 0.90,
    low_threshold: float = 0.70,
) -> QualityReport:
    """
    Compute quality metrics for a list of utterances.

    Args:
        utterances:      List of Utterance objects.
        high_threshold:  Minimum confidence for HIGH tier.
        low_threshold:   Maximum confidence for LOW tier.

    Returns:
        QualityReport with all quality metrics.
    """
    if not utterances:
        return QualityReport(
            total_utterances=0,
            utterances_with_confidence=0,
            high_confidence_count=0,
            medium_confidence_count=0,
            low_confidence_count=0,
            unknown_confidence_count=0,
            average_confidence=None,
            min_confidence=None,
            max_confidence=None,
            review_required_count=0,
            overlap_count=0,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    confidences = [u.confidence for u in utterances if u.confidence is not None]
    high = sum(1 for c in confidences if c >= high_threshold)
    medium = sum(1 for c in confidences if low_threshold <= c < high_threshold)
    low = sum(1 for c in confidences if c < low_threshold)
    unknown = len(utterances) - len(confidences)

    avg_conf = sum(confidences) / len(confidences) if confidences else None
    min_conf = min(confidences) if confidences else None
    max_conf = max(confidences) if confidences else None

    review_ids = [
        u.id for u in utterances
        if u.confidence is not None and u.confidence < low_threshold
    ]

    overlap_count = sum(1 for u in utterances if u.overlap)

    # Mark utterances as review_required
    for u in utterances:
        if u.confidence is not None and u.confidence < low_threshold:
            u.review_required = True

    return QualityReport(
        total_utterances=len(utterances),
        utterances_with_confidence=len(confidences),
        high_confidence_count=high,
        medium_confidence_count=medium,
        low_confidence_count=low,
        unknown_confidence_count=unknown,
        average_confidence=avg_conf,
        min_confidence=min_conf,
        max_confidence=max_conf,
        review_required_count=len(review_ids),
        overlap_count=overlap_count,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        low_confidence_utterance_ids=review_ids,
    )
