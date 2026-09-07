"""
Diagnosis extraction from provisional/confirmed diagnosis fields and
free-text clinical notes, combined with assertion detection.
"""
from __future__ import annotations
from normalization.diagnosis import normalize_diagnosis, split_diagnosis_string
from clinical_nlp.assertion import detect_assertion


def extract_diagnoses_from_field(raw_text: str, field_source: str):
    """
    field_source: 'provisional_diagnosis' | 'confirmed_diagnosis' | 'clinical_note'

    provisional_diagnosis / confirmed_diagnosis are dedicated, short
    diagnosis fields, so every token is treated as a diagnosis candidate
    (matched against the controlled dictionary; unmatched tokens are
    still returned but flagged, since the field itself is diagnosis-only
    and dropping them would silently lose information).

    clinical_note is free-running prose. Splitting it into pseudo
    "candidates" the same way would flag arbitrary phrases ("Medications:",
    lab lines, units) as diagnoses. Instead we only scan for known
    diagnosis vocabulary explicitly present in the text (matched=True),
    which is far more conservative and avoids inventing diagnoses from
    unrelated text — consistent with "do not invent diagnoses".
    """
    results = []
    if not raw_text:
        return results

    if field_source == "clinical_note":
        from normalization.diagnosis import _REVERSE  # controlled vocabulary
        lower = raw_text.lower()
        seen = set()
        for surface, canonical in _REVERSE.items():
            if surface and surface in lower and canonical not in seen:
                start = lower.find(surface)
                end = start + len(surface)
                assertion = detect_assertion(raw_text, start, end)
                results.append({
                    "name_raw": surface,
                    "name": canonical,
                    "name_matched": True,
                    "assertion": assertion,
                    "field_source": field_source,
                    "confidence": 0.6,  # lower confidence: incidental mention in free text
                    "source_text": raw_text[max(0, start - 30):end + 20].strip(),
                })
                seen.add(canonical)
        return results

    for candidate in split_diagnosis_string(raw_text):
        canonical, matched = normalize_diagnosis(candidate)
        if not canonical:
            continue

        start = raw_text.lower().find(candidate.lower())
        end = start + len(candidate) if start >= 0 else len(raw_text)
        assertion = detect_assertion(raw_text, max(start, 0), max(end, 0))

        results.append({
            "name_raw": candidate,
            "name": canonical,
            "name_matched": matched,
            "assertion": assertion,
            "field_source": field_source,
            "confidence": 0.85 if matched else 0.5,
            "source_text": raw_text.strip(),
        })

    return results
