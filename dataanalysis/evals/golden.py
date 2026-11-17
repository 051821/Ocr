"""Golden regression fixtures: de-identified patient records with their verified rule-based outputs.

A fixture (tests/golden/<id>.json) holds
  input     the record as core.fetching returns it, de-identified: IDs replaced, doctor and referral names removed,
            scanned-document text removed (its extracted lab results / medicines / diagnosis stay), every date shifted
            by a fixed per-patient number of days (intervals, and so every rule, are unchanged);
  terms     the term resolutions (clinical_terms) the record had when frozen, so tests replay them without a model;
  expected  the outputs a reviewer checked against the source: the rule-based brief, the snapshot flags and the
            brief checks.
outputs() recomputes 'expected' from 'input' + 'terms'; tests/test_golden.py fails on any difference.
"""
import copy
import datetime
import hashlib
import re
from unittest import mock

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})")
_DROP_DOC_KEYS = {"extracted_text", "clean_text", "raw_text", "text"}
_DROP_SUMMARY_KEYS = {"patient_name", "name", "address", "doctor", "doctor_name", "hospital", "phone"}


def fixture_id(patient_id):
    return "g" + hashlib.sha256(str(patient_id).encode()).hexdigest()[:10]


def _shift(value, days):
    def move(m):
        try:
            d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) - datetime.timedelta(days=days)
        except ValueError:
            return m.group(0)
        return d.isoformat()
    if isinstance(value, str):
        return _ISO.sub(move, value)
    if isinstance(value, list):
        return [_shift(v, days) for v in value]
    if isinstance(value, dict):
        return {k: _shift(v, days) for k, v in value.items()}
    return value


def _clean_doc(doc, n):
    out = {k: v for k, v in doc.items() if k not in _DROP_DOC_KEYS}
    out["doc_id"] = f"d{n}"
    if isinstance(out.get("clinical_summary"), dict):
        out["clinical_summary"] = {k: v for k, v in out["clinical_summary"].items() if k not in _DROP_SUMMARY_KEYS}
    return out


def deidentify(record):
    """A copy of a fetched record that names no one and dates nothing real (see module doc)."""
    record = copy.deepcopy(record)
    fid = fixture_id(record.get("patient_id"))
    days = 30 + int(fid[1:], 16) % 300
    record["patient_id"] = fid
    n = 0
    for i, v in enumerate(record.get("visits") or [], 1):
        v["visit_id"] = f"v{i}"
        v["doctor_id"] = None
        if v.get("referral_specialist_name"):
            v["referral_specialist_name"] = "specialist"
        docs = []
        for d in v.get("documents") or []:
            n += 1
            docs.append(_clean_doc(d, n))
        v["documents"] = docs
    unlinked = []
    for d in record.get("unlinked_documents") or []:
        n += 1
        unlinked.append(_clean_doc(d, n))
    record["unlinked_documents"] = unlinked
    return _shift(record, days)


def run(record, terms):
    """(patient_data, view): the dashboard pipeline over a fixture record, term resolution replayed from 'terms'."""
    import insight_view
    from services import patient_context

    def replay(patient_data):
        patient_data["_terms"] = copy.deepcopy(terms)
        return patient_data["_terms"]

    patient = copy.deepcopy(record)
    with mock.patch.object(patient_context, "resolve_terms", replay):
        patient_context.prepare_patient_data(patient)
    return patient, insight_view.build_view(patient, None)["patient"]


def outputs(record, terms):
    """What a fixture pins: rule-based brief lines, snapshot flags (level, title, evidence), brief-check results."""
    from clinical_insights import build_insight_evidence, deterministic_insights
    patient, view = run(record, terms)
    return {
        "brief": deterministic_insights(build_insight_evidence(patient)),
        "flags": [[f.get("level"), f.get("title"), f.get("evidence")] for f in view.get("flags") or []],
        "checks": [[i["check"], i["severity"], i["detail"]] for i in view.get("brief_checks") or []],
    }
