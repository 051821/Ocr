"""
Stage 5-8: builds structured, in-memory ClinicalEvent objects from a
visit's clean text + diagnosis fields.

No new database table is created here — events are plain dataclasses /
dicts, optionally serializable to JSON. This module is the seam between
"Stage 4 Clinical Record" (your existing OCR + DB-join output) and
everything downstream (dedup, timeline, trends, report).

Expected input shape for `build_events_for_visit` (one visit):

    {
        "patient_id": "...",
        "visit_id": "...",
        "visit_date": "2026-05-05",
        "provisional_diagnosis": "Htn t2dm",       # optional
        "confirmed_diagnosis": None,               # optional
        "documents": [
            {"document_id": "docA", "document_type": "prescription",
             "clean_text": "..."},
            {"document_id": "docB", "document_type": "lab_report",
             "clean_text": "..."},
        ],
    }

If you only have a single merged `medical_text` per visit (as in the
original analysis.py), wrap it as one synthetic document with
document_id=None — see longitudinal/adapters.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import logging

from clinical_nlp.labs import extract_labs
from clinical_nlp.medications import extract_medications
from clinical_nlp.diagnoses import extract_diagnoses_from_field
from clinical_nlp.symptoms import extract_symptoms
from clinical_nlp.temporal import resolve_event_date
from clinical_nlp.assertion import detect_assertion, is_in_reference_text
from clinical_nlp.ner import extract_supplementary_entities
from normalization.medication import normalize_medication
from normalization.diagnosis import normalize_diagnosis

logger = logging.getLogger(__name__)



@dataclass
class ClinicalEvent:
    patient_id: str
    visit_id: str
    visit_date: Optional[str]
    document_id: Optional[str]
    document_type: Optional[str]
    event_type: str                 # 'laboratory' | 'medication' | 'diagnosis' | 'symptom'
    event_date: Optional[str]
    name: str
    name_raw: str
    assertion: str = "present"      # present | negated | historical | suspected
    temporality: str = "current"
    confidence: float = 0.5
    needs_verification: bool = False
    source_text: str = ""
    source_documents: list = field(default_factory=list)

    # event-type-specific fields (kept flat/optional rather than a new
    # table/class per type, to stay simple and JSON-serializable)
    value: Optional[float] = None
    unit: Optional[str] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    abnormal: Optional[bool] = None
    dose: Optional[float] = None
    frequency: Optional[str] = None
    qualitative_value: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_unclaimed_text(full_text: str, claimed_snippets: list[str]) -> str:
    """
    Returns the portion of full_text NOT already covered by any
    extractor's source_text. This is a line-level approximation (not
    exact character-span tracking) — good enough to decide "is this line
    worth sending to the LLM fallback" without needing every extractor to
    report precise offsets.
    """
    claimed_lines = set()
    for snippet in claimed_snippets:
        for line in snippet.splitlines():
            line = line.strip()
            if line:
                claimed_lines.add(line)

    unclaimed_lines = []
    for line in full_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped in claimed or claimed in stripped for claimed in claimed_lines):
            continue
        unclaimed_lines.append(stripped)

    return "\n".join(unclaimed_lines)


def build_events_for_visit(visit: dict) -> list[ClinicalEvent]:
    """
    Runs Stage 5 (NLP extraction) + Stage 6 (normalization, done inside
    the extractor modules) for every document in a visit and returns a
    flat list of ClinicalEvent objects. Deduplication (Stage 9) happens
    later, in timeline/deduplicate.py — this function intentionally does
    not dedup, so every document's contribution stays traceable.
    """
    events: list[ClinicalEvent] = []

    patient_id = visit["patient_id"]
    visit_id = visit["visit_id"]
    visit_date = visit.get("visit_date")

    # --- diagnoses: come from the visit-level prescription fields, not
    # a specific document, so document_id is left None but the visit_id
    # still anchors it to the source record.
    for field_name, field_value in (
        ("provisional_diagnosis", visit.get("provisional_diagnosis")),
        ("confirmed_diagnosis", visit.get("confirmed_diagnosis")),
    ):
        for d in extract_diagnoses_from_field(field_value, field_name):
            events.append(ClinicalEvent(
                patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                document_id=None, document_type="prescription_record",
                event_type="diagnosis",
                event_date=visit_date,
                name=d["name"], name_raw=d["name_raw"],
                assertion=d["assertion"], confidence=d["confidence"],
                needs_verification=not d["name_matched"],
                source_text=d["source_text"],
                source_documents=[],
            ))

    # --- per-document extraction: labs, medications, symptoms, and
    # diagnosis mentions embedded in free-text clinical notes.
    for doc in visit.get("documents", []):
        doc_id = doc.get("document_id")
        doc_type = doc.get("document_type")
        text = doc.get("clean_text") or ""
        if not text.strip():
            continue

        event_date = resolve_event_date(text, visit_date)

        try:
            for lab in extract_labs(text):
                events.append(ClinicalEvent(
                    patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                    document_id=doc_id, document_type=doc_type,
                    event_type="laboratory", event_date=event_date,
                    name=lab["name"], name_raw=lab["name_raw"],
                    confidence=lab["confidence"],
                    needs_verification=not lab["name_matched"],
                    source_text=lab["source_text"], source_documents=[],
                    value=lab["value"], unit=lab["unit"],
                    reference_low=lab["reference_low"], reference_high=lab["reference_high"],
                    abnormal=lab["abnormal"],
                    qualitative_value=lab.get("qualitative_value"),
                ))
        except Exception:
            logger.exception("Lab extraction failed for document %s", doc_id)

        try:
            for med in extract_medications(text):
                events.append(ClinicalEvent(
                    patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                    document_id=doc_id, document_type=doc_type,
                    event_type="medication", event_date=event_date,
                    name=med["name"], name_raw=med["name_raw"],
                    confidence=med["confidence"],
                    needs_verification=med["needs_verification"],
                    source_text=med["source_text"], source_documents=[],
                    dose=med["dose"], unit=med["unit"], frequency=med["frequency"],
                ))
        except Exception:
            logger.exception("Medication extraction failed for document %s", doc_id)

        try:
            for sym in extract_symptoms(text):
                events.append(ClinicalEvent(
                    patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                    document_id=doc_id, document_type=doc_type,
                    event_type="symptom", event_date=event_date,
                    name=sym["name"], name_raw=sym["name"],
                    assertion=sym["assertion"], confidence=sym["confidence"],
                    source_text=sym["source_text"], source_documents=[],
                ))
        except Exception:
            logger.exception("Symptom extraction failed for document %s", doc_id)

        try:
            for d in extract_diagnoses_from_field(text, "clinical_note"):
                events.append(ClinicalEvent(
                    patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                    document_id=doc_id, document_type=doc_type,
                    event_type="diagnosis", event_date=event_date,
                    name=d["name"], name_raw=d["name_raw"],
                    assertion=d["assertion"], confidence=d["confidence"] * 0.7,
                    needs_verification=True,  # free-text note mentions need review
                    source_text=d["source_text"], source_documents=[],
                ))
        except Exception:
            logger.exception("Diagnosis extraction failed for document %s", doc_id)

        try:
            # medications already captured with full dose/frequency detail
            # by the rule-based extractor for THIS document — skip re-adding
            # them as a bare, dose-less NER duplicate.
            already_found_dx = {
                e.name.lower() for e in events
                if e.event_type == "diagnosis" and e.visit_id == visit_id
            }

            for ent in extract_supplementary_entities(text):
                category = ent["category"]  # 'symptom' | 'diagnosis' | 'medication' | 'lab_finding'

                # Unstructured lab_finding entities are not direct reportable events
                if category == "lab_finding":
                    continue

                # Exclude any entities appearing in reference / educational / guideline text
                if is_in_reference_text(text, ent["start"], ent["end"]):
                    continue

                assertion = detect_assertion(text, ent["start"], ent["end"])
                if assertion == "suspected" and is_in_reference_text(text, ent["start"], ent["end"]):
                    continue

                if category == "medication":
                    name, _ = normalize_medication(ent["name"])
                    if name.lower() in already_found_meds:
                        continue
                elif category == "diagnosis":
                    name, _ = normalize_diagnosis(ent["name"])
                    if name.lower() in already_found_dx:
                        continue
                else:  # symptom
                    name = ent["name"].title()

                events.append(ClinicalEvent(
                    patient_id=patient_id, visit_id=visit_id, visit_date=visit_date,
                    document_id=doc_id, document_type=doc_type,
                    event_type=category, event_date=event_date,
                    name=name, name_raw=ent["name"],
                    assertion=assertion,
                    confidence=round(ent["score"] * 0.7, 2),
                    needs_verification=True,
                    source_text=text[max(0, ent["start"] - 30):ent["end"] + 20].strip(),
                    source_documents=[],
                ))
        except Exception:
            logger.exception("NER supplementary extraction failed for document %s", doc_id)

    return events