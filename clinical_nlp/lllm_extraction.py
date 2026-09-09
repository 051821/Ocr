"""
LLM-based fallback extraction for text not claimed by any rule-based or
NER pass.

This is intentionally the LAST resort, not the primary extractor:
regex/dictionary passes stay first because they're fast, deterministic,
and already reliable on well-structured lines. This module only ever
sees whatever's left over — so it exists specifically to catch report
formats (CBC panels, culture/sensitivity, urine routine, anything not
yet special-cased) without writing a new parser for every report type.

Every result from this module is unconditionally flagged
needs_verification=True and tagged extraction_source="llm_fallback" so
it's never silently treated as equivalent-confidence to a dictionary
match downstream.
"""
from __future__ import annotations
import json
import logging
import re
from ollama import chat

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You extract structured clinical findings from raw, possibly OCR-damaged
medical report text.

Return ONLY a JSON array. No prose, no markdown fences, nothing else.
If nothing relevant is found, return exactly: []

Each element must be one of these shapes:

Lab result:
{"type": "lab", "name": "<test name as written>", "value": <number or string if qualitative, e.g. "ABSENT">, "unit": "<unit or null>", "reference_low": <number or null>, "reference_high": <number or null>}

Medication:
{"type": "medication", "name": "<drug name as written>", "dose": <number or null>, "unit": "<unit or null>", "frequency": "<frequency code as written, or null>"}

Diagnosis:
{"type": "diagnosis", "name": "<condition as written>"}

Symptom:
{"type": "symptom", "name": "<symptom as written>"}

Rules:
- Only extract what is explicitly present in the text. Never invent, infer, or guess a value that isn't written.
- If a value is qualitative (ABSENT, PRESENT, CLEAR, Yellow, etc.), put it in "value" as a string and leave "unit" null.
- Skip section headers, method names, lab equipment names, and demographic fields (age, sex, patient ID, doctor name).
- Skip anything that is guideline/reference text about what a finding "may indicate" in general — only extract this patient's actual results.
- If OCR garbling makes something unreadable/ambiguous, skip it rather than guessing.
""".strip()


def _build_prompt(text: str) -> str:
    return f"{_SYSTEM_PROMPT}\n\nTEXT:\n{text.strip()}"


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"```json|```", "", raw).strip()


def extract_with_llm(text: str, model: str = "qwen3:4b", source_label: str = "unclaimed_text"):
    """
    Returns a list of dicts, normalized to the same general shape used
    elsewhere in the pipeline:
    {category, name, value, unit, dose, frequency, reference_low,
     reference_high, confidence, needs_verification, source_text,
     extraction_source}

    Never raises — returns [] on any failure (missing ollama, malformed
    JSON, empty input), consistent with how ner.py degrades gracefully
    when its model is unavailable.
    """
    if not text or not text.strip():
        return []

    try:
        response = chat(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(text)}],
        )
        raw = response.message.content
    except Exception as e:  # noqa: BLE001 - optional dependency, same pattern as ner.py
        logger.warning("LLM fallback extraction unavailable (%s). Skipping.", e)
        return []

    cleaned = _strip_json_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM fallback returned non-JSON output, discarding: %r", raw[:200])
        return []

    if not isinstance(parsed, list):
        return []

    results = []
    for item in parsed:
        if not isinstance(item, dict) or "type" not in item or "name" not in item:
            continue

        item_type = item.get("type")
        name = str(item.get("name") or "").strip()
        if not name or len(name) < 2:
            continue

        category_map = {
            "lab": "laboratory",
            "medication": "medication",
            "diagnosis": "diagnosis",
            "symptom": "symptom",
        }
        category = category_map.get(item_type)
        if category is None:
            continue

        value = item.get("value")
        # Keep numeric values numeric, everything else (qualitative
        # results, or malformed types) stays as a string / None rather
        # than raising.
        numeric_value = None
        qualitative_value = None
        if isinstance(value, (int, float)):
            numeric_value = float(value)
        elif isinstance(value, str) and value.strip():
            qualitative_value = value.strip()

        results.append({
            "category": category,
            "name": name,
            "value": numeric_value,
            "qualitative_value": qualitative_value,
            "unit": item.get("unit"),
            "dose": item.get("dose") if isinstance(item.get("dose"), (int, float)) else None,
            "frequency": item.get("frequency"),
            "reference_low": item.get("reference_low") if isinstance(item.get("reference_low"), (int, float)) else None,
            "reference_high": item.get("reference_high") if isinstance(item.get("reference_high"), (int, float)) else None,
            "confidence": 0.4,  # deliberately low: unverified LLM extraction
            "needs_verification": True,
            "source_text": text[:200].strip(),
            "extraction_source": source_label,
        })

    return results