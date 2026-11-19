"""Patient analysis endpoints (wraps dataanalysis/services)."""
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder

import clinical_snapshot
import insight_view
import snapshot_view
import patient_types
from analysis import format_age_sex
from review_rules import deterministic_review
from core.demographics import fetch_demographics
from core.fetching import fetch_all_patient_ids
from core.referrals import fetch_referrals
from core.vital_flags import fetch_recent_vital_flags
from services.chat.answers import answer_patient_question
from services.graph_service import is_neo4j_configured, sync_patient_to_graph
from services.patient_context import ai_assist_extraction, invalidate_patient, lookup_side_effects
from services.review_service import get_or_generate_review, is_review_current
from services.timeline import create_patient_timeline
from storage.cache_db import get_top_questions, record_chat_question

from api.deps import get_patient_context, public_view, require_api_key
from api.schemas import ChatRequest, ChatResponse, FaqResponse, PatientList, ReviewResponse

try:
    from config import OPENROUTER_API_KEY as CHAT_API_KEY
except Exception:
    CHAT_API_KEY = None

router = APIRouter(prefix="/patients", tags=["patients"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=PatientList)
def list_patients():
    ids = [str(pid) for pid in fetch_all_patient_ids()]
    return PatientList(patient_ids=ids, total=len(ids))


@router.get("/{patient_id}")
def get_patient(ctx=Depends(get_patient_context)):
    return public_view(ctx[0])


@router.post("/{patient_id}/refresh")
def refresh_patient(patient_id: str):
    invalidate_patient(patient_id)
    return {"patient_id": patient_id, "status": "Cache cleared; next request refetches."}


@router.get("/{patient_id}/demographics")
def get_demographics(patient_id: UUID):
    """Age and sex only, read straight from the patient table (no record load, no LLM)."""
    row = fetch_demographics(str(patient_id))
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found.")
    age_years, sex = row
    return {"patient_id": str(patient_id), "age_years": age_years, "sex": sex,
            "age_sex": format_age_sex(age_years, sex)}


def _snapshot(patient_data):
    """Clinical snapshot with the timeline grid spelled out: each row gets one cell per
    encounter (number, date, recorded yes/no), matching the dashboard grid column for column."""
    snap = clinical_snapshot.build_snapshot(patient_data) or {}
    if not snap:
        return {}
    dates = snap.get("dates") or []
    snap["timeline"] = [
        {"label": row["label"],
         "kind": row.get("kind", "condition"),
         "encounters": sorted(row["encounters"]),
         "cells": [{"encounter": i, "date": d, "recorded": i in row["encounters"]}
                   for i, d in enumerate(dates, 1)]}
        for row in snap.get("timeline") or []]
    snap["encounter_columns"] = [{"encounter": i, "date": d} for i, d in enumerate(dates, 1)]
    return snap


@router.get("/{patient_id}/snapshot")
def get_snapshot(ctx=Depends(get_patient_context)):
    patient_data = ctx[0]
    snap = _snapshot(patient_data)
    brief = deterministic_review(patient_data)
    return {
        "snapshot": snap,
        "clinical_course": clinical_snapshot.build_clinical_course(patient_data),
        # Rule-based doctor's brief, shown until an AI review is generated.
        "brief": brief,
        # Tile area of each brief point and lab detail (snapshot_view.AREAS keys; 'general' when none fits).
        "brief_tags": snapshot_view.tag_review(brief, snap.get("flags") or []),
        "patient_types": patient_types.classify_patient(patient_data),
        "visit_clusters": patient_types.classify_visits(patient_data),
        "exercise": patient_types.exercise_summary(patient_data),
    }


@router.get("/{patient_id}/insight-view")
def get_insight_view(ctx=Depends(get_patient_context),
                     view: Optional[str] = Query(None, pattern="^(quick|detailed)$",
                                                 description="Tier to open; overrides the doctor's default"),
                     doctor_id: Optional[str] = Query(None, pattern=r"^[0-9A-Za-z-]{1,64}$",
                                                      description="Doctor whose default view applies")):
    """Data for the AI Analysis & Summary and Lab Results tabs: the two-tier summary (quick + detailed), merged AI +
    rule brief, concerns with their raw rows and rules, visits, lab rows. Never runs the LLM: the AI parts are the
    cached review, when one is current. 'view' is the tier to open: ?view=, else the doctor's saved default, else
    quick."""
    import summary_tiers
    from storage.cache_db import get_doctor_settings
    patient_data = ctx[0]
    review = get_or_generate_review(patient_data) if is_review_current(patient_data) else None
    out = insight_view.build_view(patient_data, review)
    saved = get_doctor_settings(doctor_id) if doctor_id else None
    out["view"] = summary_tiers.resolve_view(view, (saved or {}).get("default_view"))
    return jsonable_encoder(out, custom_encoder={set: sorted, frozenset: sorted})


def _patient_type(patient_data):
    types = patient_types.classify_patient(patient_data)
    exercise = patient_types.exercise_summary(patient_data)
    return {
        # The type that decided exercise suitability (not just the most frequent one).
        "patient_type": exercise["patient_type"],
        "patient_type_reason": exercise["type_reason"],
        # True/False for a classified patient; None when no type matched.
        "exercise_suitable": {"Yes": True, "No": False}.get(exercise["exercise"]),
        "exercise_reason": exercise["reason"],
        "types": types,
    }


@router.get("/{patient_id}/patient-type")
def get_patient_type(patient_id: str, ctx=Depends(get_patient_context)):
    """Body-system patient type and whether exercise is suitable (rule-based, no LLM)."""
    return {"patient_id": patient_id, **_patient_type(ctx[0]),
            "visits": patient_types.classify_visits(ctx[0])}


@router.get("/{patient_id}/summary")
def get_summary(patient_id: str, ctx=Depends(get_patient_context)):
    """Everything the dashboard shows for one patient, in one call. Never runs the LLM:
    ai_review is the cached review, or null until one is generated (POST /review)."""
    return build_summary(patient_id, ctx[0])


def build_summary(patient_id, patient_data, include_ai=True):
    """The /summary body. include_ai=False (hosted public API) leaves out ai_review and never reads the cache."""
    visits = patient_data.get("visits") or []
    journey = patient_data.get("_medication_journey_cache") or []
    review = get_or_generate_review(patient_data) if include_ai and is_review_current(patient_data) else None
    out = {
        "patient_id": patient_id,
        "age_years": patient_data.get("age_years"),
        "sex": patient_data.get("sex"),
        "long_term_conditions": patient_data.get("long_term_conditions") or [],
        # patient.health_history_json as stored: {key: {"present", "notes" | "qty"}, "other_notes": str}.
        "health_history": patient_data.get("health_history") or {},
        # Last 10 visits, newest first; each visit has its own assigned doctor.
        # is_emergency / referral fields come from the visit's prescription(s).
        "visits": [{"visit_id": v["visit_id"], "visit_date": str(v.get("visit_date") or "")[:10],
                    "doctor_id": v.get("doctor_id"),
                    "is_emergency": bool(v.get("is_emergency")),
                    "is_referral_recommended": bool(v.get("is_referral_recommended")),
                    "referral_specialist_name": v.get("referral_specialist_name"),
                    "referral_treatment_name": v.get("referral_treatment_name")}
                   for v in list(reversed(visits))[:10]],
        **_patient_type(patient_data),
        "ongoing_medicines": journey[-1]["all_active_meds"] if journey else [],
        # Newest visit first; visits without structured vitals are left out.
        "vitals": [{"visit_id": v["visit_id"], "visit_date": str(v.get("visit_date") or "")[:10],
                    "readings": v["db_vitals_rows"]}
                   for v in reversed(visits) if v.get("db_vitals_rows")],
        # visit_vital_flag rows for the last 5 completed visits, newest first (IMPLAUSIBLE tier left out).
        "vital_flags": fetch_recent_vital_flags(patient_id),
        "clinical_snapshot": _snapshot(patient_data),
        "ai_review": None if review is None else {
            "markdown": review.get("raw_markdown", ""),
            "model_name": review.get("model_name", ""),
            "generated_at": review.get("generated_at", ""),
        },
    }
    if not include_ai:
        del out["ai_review"]
    return out


@router.get("/{patient_id}/referrals")
def get_referrals(patient_id: UUID):
    """Where the patient was referred out to and for what (DB read, no record load, no LLM)."""
    return {"patient_id": str(patient_id), "referrals": jsonable_encoder(fetch_referrals(str(patient_id)))}


@router.get("/{patient_id}/long-term-conditions")
def get_long_term_conditions(patient_id: str, ctx=Depends(get_patient_context)):
    """Long-term conditions recorded for the patient (patient_condition -> long_term_condition)."""
    return {"patient_id": patient_id, "long_term_conditions": ctx[0].get("long_term_conditions") or []}


@router.get("/{patient_id}/medications")
def get_medication_journey(ctx=Depends(get_patient_context)):
    return ctx[0].get("_medication_journey_cache") or {}


@router.get("/{patient_id}/timeline")
def get_timeline(ctx=Depends(get_patient_context)):
    return {"timeline": create_patient_timeline(ctx[0])}


@router.get("/{patient_id}/side-effects")
def get_side_effects(patient_id: str, use_ai: bool = False, ctx=Depends(get_patient_context)):
    """use_ai=true looks up unlisted medicines and keeps the result for chat/review."""
    patient_data = lookup_side_effects(patient_id)[0] if use_ai else ctx[0]
    return patient_data.get("possible_side_effects") or {}


@router.post("/{patient_id}/ai-assist")
def ai_assist(patient_id: str, ctx=Depends(get_patient_context)):
    """Re-run AI extraction on low-confidence documents; returns the updated record."""
    return public_view(ai_assist_extraction(patient_id)[0])


@router.get("/{patient_id}/review", response_model=ReviewResponse)
def get_review(patient_id: str, ctx=Depends(get_patient_context)):
    """Return the cached review only; 404 if none is current (use POST to generate)."""
    patient_data = ctx[0]
    if not is_review_current(patient_data):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No current review cached. POST to generate one.")
    return _review_response(patient_id, patient_data, get_or_generate_review(patient_data))


@router.post("/{patient_id}/review", response_model=ReviewResponse)
def generate_review(patient_id: str, force_refresh: bool = False, ctx=Depends(get_patient_context)):
    review = get_or_generate_review(ctx[0], force_refresh=force_refresh)
    return _review_response(patient_id, ctx[0], review)


def _review_response(patient_id, patient_data, review):
    """The review with each AI point and lab detail tagged with its tile area."""
    flags = (clinical_snapshot.build_snapshot(patient_data) or {}).get("flags") or []
    tags = snapshot_view.tag_review(review.get("raw_markdown", ""), flags)
    return ReviewResponse(patient_id=patient_id, **{**review, "area_tags": tags})


@router.post("/{patient_id}/chat", response_model=ChatResponse)
def chat(patient_id: str, body: ChatRequest, ctx=Depends(get_patient_context)):
    patient_data, chat_index = ctx
    meta = {}
    answer, evidence = answer_patient_question(
        patient_data, patient_id, body.question,
        history=[m.model_dump() for m in body.history],
        api_key=CHAT_API_KEY,
        retrieval_index=chat_index,
        source_hash=patient_data.get("_review_source_hash"),
        meta=meta,
    )
    if evidence:
        try:
            record_chat_question(patient_id, body.question)
        except Exception:
            pass
    return ChatResponse(patient_id=patient_id, answer=answer, evidence=evidence or [], meta=meta)


@router.get("/{patient_id}/faq", response_model=FaqResponse)
def get_faq(patient_id: str, limit: int = 5):
    return FaqResponse(patient_id=patient_id, questions=get_top_questions(patient_id, limit=limit))


@router.post("/{patient_id}/graph-sync")
def graph_sync(patient_id: str, ctx=Depends(get_patient_context)):
    if not is_neo4j_configured():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Neo4j is not configured.")
    if not sync_patient_to_graph(ctx[0]):
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Graph sync failed. Check the Neo4j connection.")
    return {"patient_id": patient_id, "status": "Graph is current."}
