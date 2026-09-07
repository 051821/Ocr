"""
Symptom extraction from clinical notes. Uses a controlled symptom
vocabulary + assertion detection so "denies chest pain" is never reported
as an active symptom.

The vocabulary is intentionally short and extensible — add entries here,
or swap this module out for a biomedical NER model (see ner.py) once one
is available in your environment.
"""
from __future__ import annotations
import re
from clinical_nlp.assertion import detect_assertion

SYMPTOM_VOCAB = [
    "chest pain", "shortness of breath", "breathlessness", "cough",
    "fever", "headache", "dizziness", "nausea", "vomiting",
    "abdominal pain", "back pain", "neck pain", "knee pain",
    "joint pain", "fatigue", "weakness", "swelling", "numbness",
    "tingling", "blurred vision", "palpitations", "weight loss",
    "weight gain", "loss of appetite",
]


def extract_symptoms(text: str):
    """
    Returns a list of dicts:
    {name, assertion, confidence, source_text}
    """
    results = []
    if not text:
        return results

    lower = text.lower()
    for symptom in SYMPTOM_VOCAB:
        for m in re.finditer(re.escape(symptom), lower):
            assertion = detect_assertion(text, m.start(), m.end())
            results.append({
                "name": symptom.title() if symptom != "chest pain" else "Chest pain",
                "assertion": assertion,
                "confidence": 0.75,
                "source_text": text[max(0, m.start() - 40):m.end() + 10].strip(),
            })

    # "bilateral X pain" style — normalize e.g. "both knee pain" ->
    # "Bilateral knee pain" for readability
    for m in re.finditer(r"\bboth\s+(\w+)\s+pain\b", lower):
        results.append({
            "name": f"Bilateral {m.group(1)} pain".title(),
            "assertion": detect_assertion(text, m.start(), m.end()),
            "confidence": 0.7,
            "source_text": text[max(0, m.start() - 20):m.end() + 10].strip(),
        })

    return results
