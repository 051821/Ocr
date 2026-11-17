"""Complaint phrases -> symptoms -> body systems, from the registry data/reference/symptom_map.json.

A complaint is read through the registry, not by keyword: each entry consumes the words it matches, so a broader entry
later in the list never re-reads them ('swelling on both eyes' is periorbital oedema; its 'eyes' is not a visual symptom).
Rules match a complaint by symptom tags; their own patterns only see the words no entry explains (remainder()).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "data" / "reference" / "symptom_map.json"


@lru_cache(maxsize=1)
def _entries():
    with _PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return [{**e, "regex": re.compile(rf"\b(?:{e['pattern']})", re.I)} for e in data["symptoms"]]


def classify(phrase):
    """[{symptom, system, tags, text}] for one complaint phrase, in phrase order; the words each entry matched are
    consumed so no later entry reads them again."""
    text = str(phrase or "")
    consumed = [False] * len(text)
    found = []
    for e in _entries():
        for m in e["regex"].finditer(text):
            if any(consumed[m.start():m.end()]):
                continue
            consumed[m.start():m.end()] = [True] * (m.end() - m.start())
            found.append({"symptom": e["symptom"], "system": e["system"], "tags": list(e["tags"]),
                          "excluded_by": list(e.get("excluded_by") or []),
                          "text": m.group(0).strip(), "start": m.start(), "end": m.end()})
    return sorted(found, key=lambda x: x["start"])


def tags(phrase):
    return {t for hit in classify(phrase) for t in hit["tags"]}


def systems(phrase):
    return list(dict.fromkeys(hit["system"] for hit in classify(phrase)))


def remainder(phrase):
    """The phrase with every registry-explained span blanked out: what a rule's own pattern may still read."""
    text = str(phrase or "")
    for hit in sorted(classify(phrase), key=lambda h: -h["start"]):
        text = text[:hit["start"]] + " " * (hit["end"] - hit["start"]) + text[hit["end"]:]
    return text
