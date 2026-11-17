"""Patient preparation shared by the Streamlit UI and the FastAPI layer."""
import copy
import threading
import time

from clinical_terms import resolve_terms
import data_quality
from core.encounters import unify_specializations
from core.tracing import patient_context_outputs, traced
from core.fetching import fetch_patient_history
from services.chat.retrieval import build_patient_index
from services.medication_journey import compute_medication_journey
from services.review_facts import build_review_facts, compute_source_hash, visit_hashes

CACHE_TTL_SECONDS = 300

_cache = {}
_lock = threading.Lock()


def prepare_patient_data(patient_data):
    """Attach review facts, hashes and medication journey to patient_data in place."""
    patient_id = str(patient_data.get("patient_id", ""))
    # A source MISSED follow-up is re-checked against the actual visits before anything reads it.
    data_quality.prepare(patient_data)  # follow-up re-check, duplicate visits
    unify_specializations(patient_data.get("visits"))  # also for records cached before the spellings were merged
    # What each medicine, diagnosis entry and complaint is, in the rules' vocabulary (cached per term).
    resolve_terms(patient_data)
    patient_data["_medication_journey_cache"] = compute_medication_journey(patient_data)
    facts = build_review_facts(patient_data)
    hashes = visit_hashes(patient_data, facts)
    patient_data["_review_facts"] = facts
    patient_data["_review_visit_hashes"] = hashes
    patient_data["_review_source_hash"] = compute_source_hash(patient_id, hashes)
    return patient_data


@traced(name="context.load_patient_context", process_outputs=patient_context_outputs)
def load_patient_context(patient_id, force_refresh=False):
    """Return (patient_data, chat_index) for a patient, cached for CACHE_TTL_SECONDS."""
    patient_id = str(patient_id)
    now = time.monotonic()
    with _lock:
        entry = _cache.get(patient_id)
        # timestamp None = edited in place (AI assist / side-effect lookup); kept until an explicit refresh.
        if entry and not force_refresh and (entry[0] is None or now - entry[0] < CACHE_TTL_SECONDS):
            return entry[1], entry[2]

    patient_data = prepare_patient_data(fetch_patient_history(patient_id))
    chat_index = build_patient_index(patient_data)
    with _lock:
        _cache[patient_id] = (now, patient_data, chat_index)
    return patient_data, chat_index


def _store_edited(patient_id, patient_data):
    """Re-derive facts/index for an edited record and pin it in the cache."""
    for key in ("_medication_journey_cache", "_review_facts", "_review_visit_hashes", "_review_source_hash",
                "_insight_evidence", "_terms"):
        patient_data.pop(key, None)
    prepare_patient_data(patient_data)
    chat_index = build_patient_index(patient_data)
    with _lock:
        _cache[str(patient_id)] = (None, patient_data, chat_index)
    return patient_data, chat_index


@traced(name="context.ai_assist_extraction", process_outputs=patient_context_outputs)
def ai_assist_extraction(patient_id):
    """Re-run AI extraction on low-confidence documents (moved from app.py)."""
    from analysis import _augment_with_ai_if_gaps, extract_medications_from_text, get_possible_side_effects
    patient_data = copy.deepcopy(load_patient_context(patient_id)[0])
    document_meds = set()
    for visit in patient_data.get("visits", []):
        visit_meds = set()
        for doc in visit.get("documents", []) or []:
            lines = doc.get("extracted_text") or []
            summary = doc.get("clinical_summary") or {}
            summary = _augment_with_ai_if_gaps(lines, summary, patient_id=patient_data.get("patient_id"))
            doc["clinical_summary"] = summary
            doc["lab_results"] = summary.get("lab_results") or doc.get("lab_results", [])
            names = set(extract_medications_from_text(lines))
            names.update(m.get("name") for m in summary.get("medications", []) if m.get("name"))
            doc["medications_found"] = sorted(names)
            visit_meds.update(names)
        visit["document_medications"] = sorted(visit_meds)
        document_meds.update(visit_meds)
        visit_db_meds = {m.get("name") for m in visit.get("db_medications", []) if m.get("name")}
        visit["medications_found"] = sorted(visit_db_meds | visit_meds)
    patient_data["all_document_medications"] = sorted(document_meds)
    all_meds = set(patient_data.get("all_db_medications", [])) | document_meds
    patient_data["all_medications"] = sorted(all_meds)
    patient_data["possible_side_effects"] = get_possible_side_effects(sorted(all_meds), use_ai=False)
    return _store_edited(patient_id, patient_data)


@traced(name="context.lookup_side_effects", process_outputs=patient_context_outputs)
def lookup_side_effects(patient_id):
    """AI lookup for medicines missing from the local reference list; result is kept in the cache."""
    from analysis import get_possible_side_effects
    patient_data = copy.deepcopy(load_patient_context(patient_id)[0])
    patient_data["possible_side_effects"] = get_possible_side_effects(patient_data.get("all_medications", []), use_ai=True)
    return _store_edited(patient_id, patient_data)


def invalidate_patient(patient_id=None):
    """Drop one patient (or everyone) from the context cache."""
    with _lock:
        if patient_id is None:
            _cache.clear()
        else:
            _cache.pop(str(patient_id), None)
