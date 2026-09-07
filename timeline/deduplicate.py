"""
Stage 9: Deduplication.

Same patient + visit + event_date + event_type + normalized name + value
+ unit => same clinical measurement, even if it appears in multiple
documents (e.g. a lab value repeated across a scanned report and a typed
summary). Source documents are merged, never discarded.
"""
from __future__ import annotations
from clinical_nlp.events import ClinicalEvent


def _dedup_key(e: ClinicalEvent):
    # value is rounded to avoid float-precision false negatives (e.g.
    # 1.250000001 vs 1.25 from different OCR passes)
    value_key = round(e.value, 3) if e.value is not None else None
    return (
        e.patient_id, e.visit_id, e.event_date, e.event_type,
        e.name, value_key, e.unit,
    )


def deduplicate_events(events: list[ClinicalEvent]) -> list[ClinicalEvent]:
    merged: dict[tuple, ClinicalEvent] = {}

    for e in events:
        key = _dedup_key(e)
        if key not in merged:
            # start source_documents with this event's own document
            e.source_documents = [e.document_id] if e.document_id else []
            merged[key] = e
            continue

        existing = merged[key]
        if e.document_id and e.document_id not in existing.source_documents:
            existing.source_documents.append(e.document_id)

        # keep the higher-confidence extraction as the "primary" record,
        # but always retain all source documents either way
        if e.confidence > existing.confidence:
            new_sources = existing.source_documents
            e.source_documents = new_sources + (
                [e.document_id] if e.document_id and e.document_id not in new_sources else []
            )
            merged[key] = e

    return list(merged.values())
