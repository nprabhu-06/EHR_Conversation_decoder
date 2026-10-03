"""
nlp/clinical_summary.py — Optional rule-based clinical information extraction.

Extracts structured information EXPLICITLY PRESENT in the transcript.

CRITICAL DISCLAIMER:
  - This module does NOT diagnose patients.
  - This module does NOT invent medical information.
  - This module does NOT infer test results not mentioned in the transcript.
  - Every extracted item is linked to a source utterance ID.
  - All output should be reviewed by a qualified clinician.

Sections extracted:
  1. Chief Complaint
  2. Symptoms Mentioned
  3. Relevant History Mentioned
  4. Medications Mentioned
  5. Investigations Mentioned
  6. Doctor Statements (key utterances)
  7. Patient Statements (key utterances)
  8. Follow-up Instructions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engines.base import Utterance

_VOCAB_DIR = Path(__file__).parent.parent / "medical_vocabulary"


@dataclass
class ClinicalFinding:
    """A single extracted clinical finding with source reference."""
    finding: str
    source_utterance_id: int
    category: str    # symptom | medication | investigation | instruction | history


@dataclass
class ClinicalSummary:
    """Structured clinical information extracted from the transcript."""
    chief_complaint: Optional[str] = None
    chief_complaint_utterance_id: Optional[int] = None

    symptoms: list[ClinicalFinding] = field(default_factory=list)
    history: list[ClinicalFinding] = field(default_factory=list)
    medications: list[ClinicalFinding] = field(default_factory=list)
    investigations: list[ClinicalFinding] = field(default_factory=list)
    follow_up_instructions: list[ClinicalFinding] = field(default_factory=list)

    doctor_key_statements: list[dict] = field(default_factory=list)
    patient_key_statements: list[dict] = field(default_factory=list)

    disclaimer: str = (
        "AI-GENERATED SUMMARY — FOR CLINICAL REVIEW ONLY. "
        "This summary was extracted automatically from transcript text. "
        "It has NOT been verified by a clinician. "
        "Do not use for diagnosis or treatment decisions without clinical review."
    )

    def to_dict(self) -> dict:
        return {
            "disclaimer": self.disclaimer,
            "chief_complaint": self.chief_complaint,
            "chief_complaint_utterance_id": self.chief_complaint_utterance_id,
            "symptoms": [
                {"finding": f.finding, "source_utterance_id": f.source_utterance_id}
                for f in self.symptoms
            ],
            "history_mentioned": [
                {"finding": f.finding, "source_utterance_id": f.source_utterance_id}
                for f in self.history
            ],
            "medications_mentioned": [
                {"finding": f.finding, "source_utterance_id": f.source_utterance_id}
                for f in self.medications
            ],
            "investigations_mentioned": [
                {"finding": f.finding, "source_utterance_id": f.source_utterance_id}
                for f in self.investigations
            ],
            "follow_up_instructions": [
                {"finding": f.finding, "source_utterance_id": f.source_utterance_id}
                for f in self.follow_up_instructions
            ],
            "doctor_key_statements": self.doctor_key_statements,
            "patient_key_statements": self.patient_key_statements,
        }


# ── Vocabulary loading ─────────────────────────────────────────────────────────

def _load_vocab_file(filename: str) -> set[str]:
    path = _VOCAB_DIR / filename
    terms: set[str] = set()
    if not path.exists():
        return terms
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#") and "=" not in line:
                terms.add(line)
    return terms


def _build_vocab_pattern(terms: set[str]) -> Optional[re.Pattern]:
    """Build a compiled regex that matches any term in the set."""
    if not terms:
        return None
    # Sort by length descending so longer phrases match before shorter substrings
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


# Lazy-loaded vocab patterns
_SYMPTOMS_RE: Optional[re.Pattern] = None
_MEDICATIONS_RE: Optional[re.Pattern] = None
_INVESTIGATIONS_RE: Optional[re.Pattern] = None
_HISTORY_KEYWORDS = re.compile(
    r"\b(history of|diagnosed with|known|previously|used to|had a|suffered from)\b",
    re.IGNORECASE,
)
_FOLLOWUP_KEYWORDS = re.compile(
    r"\b(follow up|come back|return in|see you in|review in|appointment|refer|referral|"
    r"scan|test|blood test|monitor|check|next week|next month|return)\b",
    re.IGNORECASE,
)
_COMPLAINT_KEYWORDS = re.compile(
    r"\b(I have|I've had|I'm having|my .* is|I feel|I am feeling|came in with|"
    r"presenting with|main complaint|problem is|bothering me)\b",
    re.IGNORECASE,
)


def _get_symptoms_re() -> Optional[re.Pattern]:
    global _SYMPTOMS_RE
    if _SYMPTOMS_RE is None:
        _SYMPTOMS_RE = _build_vocab_pattern(_load_vocab_file("symptoms.txt"))
    return _SYMPTOMS_RE


def _get_medications_re() -> Optional[re.Pattern]:
    global _MEDICATIONS_RE
    if _MEDICATIONS_RE is None:
        _MEDICATIONS_RE = _build_vocab_pattern(_load_vocab_file("medications.txt"))
    return _MEDICATIONS_RE


def _get_investigations_re() -> Optional[re.Pattern]:
    global _INVESTIGATIONS_RE
    if _INVESTIGATIONS_RE is None:
        _INVESTIGATIONS_RE = _build_vocab_pattern(_load_vocab_file("investigations.txt"))
    return _INVESTIGATIONS_RE


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_clinical_summary(utterances: list["Utterance"]) -> ClinicalSummary:
    """
    Extract structured clinical information from a list of utterances.

    Only extracts information EXPLICITLY present in the text.
    Does NOT diagnose, infer, or invent any medical information.

    Args:
        utterances: Role-labeled utterances (Doctor/Patient).

    Returns:
        ClinicalSummary dataclass.
    """
    summary = ClinicalSummary()
    symptoms_re = _get_symptoms_re()
    medications_re = _get_medications_re()
    investigations_re = _get_investigations_re()

    seen_symptoms: set[str] = set()
    seen_medications: set[str] = set()
    seen_investigations: set[str] = set()

    for u in utterances:
        text = u.text
        uid = u.id
        role = u.speaker_label

        # ── Chief complaint: first patient utterance mentioning a complaint
        if role == "Patient" and summary.chief_complaint is None:
            if _COMPLAINT_KEYWORDS.search(text):
                # Take the sentence mentioning the complaint
                summary.chief_complaint = text[:200]
                summary.chief_complaint_utterance_id = uid

        # ── Symptoms (primarily from Patient utterances)
        if symptoms_re:
            for m in symptoms_re.finditer(text):
                term = m.group(0).lower()
                if term not in seen_symptoms:
                    seen_symptoms.add(term)
                    summary.symptoms.append(ClinicalFinding(
                        finding=m.group(0),
                        source_utterance_id=uid,
                        category="symptom",
                    ))

        # ── Medications
        if medications_re:
            for m in medications_re.finditer(text):
                term = m.group(0).lower()
                if term not in seen_medications:
                    seen_medications.add(term)
                    summary.medications.append(ClinicalFinding(
                        finding=m.group(0),
                        source_utterance_id=uid,
                        category="medication",
                    ))

        # ── Investigations
        if investigations_re:
            for m in investigations_re.finditer(text):
                term = m.group(0).lower()
                if term not in seen_investigations:
                    seen_investigations.add(term)
                    summary.investigations.append(ClinicalFinding(
                        finding=m.group(0),
                        source_utterance_id=uid,
                        category="investigation",
                    ))

        # ── History
        if _HISTORY_KEYWORDS.search(text):
            summary.history.append(ClinicalFinding(
                finding=text[:200],
                source_utterance_id=uid,
                category="history",
            ))

        # ── Follow-up instructions (primarily from Doctor)
        if role == "Doctor" and _FOLLOWUP_KEYWORDS.search(text):
            summary.follow_up_instructions.append(ClinicalFinding(
                finding=text[:200],
                source_utterance_id=uid,
                category="instruction",
            ))

        # ── Key statements
        if role == "Doctor" and len(u.text.split()) > 5:
            summary.doctor_key_statements.append({
                "utterance_id": uid,
                "text": text,
                "start_time": u.start_time,
            })
        elif role == "Patient" and len(u.text.split()) > 3:
            summary.patient_key_statements.append({
                "utterance_id": uid,
                "text": text,
                "start_time": u.start_time,
            })

    return summary
