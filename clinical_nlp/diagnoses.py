"""
Diagnosis extraction from provisional/confirmed diagnosis fields and
free-text clinical notes, combined with assertion detection.
"""
from __future__ import annotations
from normalization.diagnosis import normalize_diagnosis, split_diagnosis_string
from clinical_nlp.assertion import detect_assertion, is_in_reference_text
import re


def extract_diagnoses_from_field(raw_text: str, field_source: str):
    """
    field_source: 'provisional_diagnosis' | 'confirmed_diagnosis' | 'clinical_note'

    provisional_diagnosis / confirmed_diagnosis are dedicated, short
    diagnosis fields, so every token is treated as a diagnosis candidate
    (matched against the controlled dictionary; unmatched tokens are
    still returned but flagged, since the field itself is diagnosis-only
    and dropping them would silently lose information).

    clinical_note is free-running prose. We scan for known diagnosis
    vocabulary using strict word boundaries (re.search with \b) and
    exclude any matches that appear inside educational/reference text
    (e.g. "may indicate liver disease").
    """
    results = []
    if not raw_text:
        return results

    if field_source == "clinical_note":
        from normalization.diagnosis import _REVERSE  # controlled vocabulary
        lower = raw_text.lower()
        seen = set()

        # Sort surfaces by length descending so longer phrases match before abbreviations
        sorted_surfaces = sorted(_REVERSE.items(), key=lambda kv: -len(kv[0]))

        for surface, canonical in sorted_surfaces:
            if not surface or canonical in seen:
                continue

            # Strict word-boundary search prevents 'oa' matching inside 'investioation'
            pattern = rf"\b{re.escape(surface)}\b"
            for m in re.finditer(pattern, lower):
                start, end = m.start(), m.end()

                # Exclude mentions that sit inside educational or guideline text
                if is_in_reference_text(raw_text, start, end):
                    continue

                assertion = detect_assertion(raw_text, start, end)
                # If assertion is 'suspected' purely because of reference text, skip it
                if assertion == "suspected" and is_in_reference_text(raw_text, start, end):
                    continue

                results.append({
                    "name_raw": surface,
                    "name": canonical,
                    "name_matched": True,
                    "assertion": assertion,
                    "field_source": field_source,
                    "confidence": 0.7,
                    "source_text": raw_text[max(0, start - 30):end + 20].strip(),
                })
                seen.add(canonical)
                break  # match once per canonical diagnosis in clinical note

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
            "confidence": 0.9 if matched else 0.7,
            "source_text": raw_text.strip(),
        })

    return results

