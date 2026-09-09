"""
Optional generic biomedical NER layer.
"""
from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

_ner_pipeline = None
_load_attempted = False


def _get_pipeline():
    global _ner_pipeline, _load_attempted
    if _ner_pipeline is not None or _load_attempted:
        return _ner_pipeline

    _load_attempted = True
    try:
        from transformers import pipeline
        _ner_pipeline = pipeline(
            "token-classification",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple",
        )
        logger.info("Loaded biomedical NER model d4data/biomedical-ner-all")
    except Exception as e:  # noqa: BLE001
        logger.warning("Biomedical NER model unavailable (%s). "
                        "Falling back to rule-based extraction only.", e)
        _ner_pipeline = None

    return _ner_pipeline


def run_ner(text: str):
    if not text:
        return []
    nlp = _get_pipeline()
    if nlp is None:
        return []
    try:
        return nlp(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("NER inference failed: %s", e)
        return []


def _snap_to_word_boundary(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and re.match(r"\w", text[start - 1]):
        start -= 1
    while end < len(text) and re.match(r"\w", text[end]):
        end += 1
    return start, end


# d4data/biomedical-ner-all's full label set includes several categories
# beyond symptom/diagnosis/medication. Lab_value and Diagnostic_procedure
# are added here as a supplementary signal for report types (CBC, urine,
# etc.) the dictionary-driven lab extractor doesn't cover yet — these are
# NOT parsed into name/value/unit here (NER gives spans, not structured
# fields), they're just surfaced as flagged candidates for the LLM
# fallback pass or manual review to pick up.
LABEL_MAP = {
    "Sign_symptom": "symptom",
    "Disease_disorder": "diagnosis",
    "Medication": "medication",
    "Lab_value": "lab_finding",
    "Diagnostic_procedure": "lab_finding",
}

MIN_SCORE_BY_CATEGORY = {
    "medication": 0.45,
    "diagnosis": 0.70,
    "symptom": 0.65,
    "lab_finding": 0.55,
}


def extract_supplementary_entities(text: str):
    """
    Returns a list of dicts:
    {category: 'symptom'|'diagnosis'|'medication'|'lab_finding', name, start, end, score}

    'lab_finding' entities are NOT structured (no value/unit parsed) —
    they exist so events.py can (a) mark that region of text as "already
    seen, low-value to re-send to the LLM fallback" and (b) optionally
    surface them for review even without a parsed numeric value.
    """
    entities = run_ner(text)
    if not entities:
        return []

    results = []
    seen_spans = set()

    for ent in entities:
        group = ent.get("entity_group")
        category = LABEL_MAP.get(group)
        if category is None:
            continue

        score = float(ent.get("score", 0))
        if score < MIN_SCORE_BY_CATEGORY.get(category, 0.70):
            continue

        start, end = ent.get("start", 0), ent.get("end", 0)
        start, end = _snap_to_word_boundary(text, start, end)

        name = text[start:end].strip()
        if len(name) < 2 or not re.search(r"[a-zA-Z]", name):
            continue

        span_key = (start, end)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        results.append({
            "category": category,
            "name": name,
            "start": start,
            "end": end,
            "score": score,
        })

    return results