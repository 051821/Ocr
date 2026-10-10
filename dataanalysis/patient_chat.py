"""Patient-scoped clinical answers and bounded local retrieval."""
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path
import re
import time
from collections import Counter
from datetime import datetime, timezone

from core.llm_client import complete as llm_complete
from medication_journey import compute_medication_journey
from storage.ai_analysis_csv import get_llm_cache, set_llm_cache
from neo4j_client import is_neo4j_configured, query_patient_graph


CHAT_LOG_QUESTIONS = True

logger = logging.getLogger(__name__)

audit_logger = logging.getLogger("patient_chat.audit")
audit_logger.propagate = False
audit_logger.setLevel(logging.INFO)
if not audit_logger.handlers:
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _handler = RotatingFileHandler(
        str(_log_dir / "patient_chat.log"),
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    audit_logger.addHandler(_handler)

_BASE_PROMPT = (
    "You are an expert Clinical AI Assistant specializing in longitudinal patient records, medical journeys, and condition tracking.\n"
    "Format: '### Answer' then '### Evidence' with ONE short line per source "
    "(Encounter #, date, Visit ID, category). Never copy record text verbatim.\n"
    "Never call a value abnormal/high/low unless the evidence explicitly does.\n"
    "Respect visit constraints ('in visit 1', 'after visit 3', 'latest').\n"
    "State medication shifts (started/dose changed/continued/discontinued) with encounter+date.\n"
    "Do not suggest new prescriptions or speculative diagnoses."
)

_SIDE_EFFECT_RULE = (
    "\nMedication Side Effects & Safety: When asked about side effects or adverse reactions for any medication:\n"
    "- Actively list the established clinical side effects, adverse reactions, and safety warnings from medical pharmacology knowledge. Do not state that side effects are absent from the record; explain them for the clinician.\n"
    "- Connect directly to this patient's record: specify when the medication was prescribed (Encounter #, Date, dosage/instructions), and note whether any adverse reactions were reported in the clinical notes.\n"
    "- Outline practical warning signs and red flags the patient should monitor."
)

_LAB_RULE = (
    "\nLaboratory Values & Clinical Indications: When asked what lab findings or test results indicate, mean, or signify:\n"
    "- Provide the medical significance and clinical causes of the abnormal value (whether elevated/high or decreased/low), using your medical knowledge.\n"
    "- Do not restrict your answer to the literal text of lab notes or report footnotes; actively explain what the low or high finding clinically implies (potential etiologies, metabolic causes, medication effects).\n"
    "- Connect and correlate the findings with the patient's documented diagnoses, symptoms, and prescribed treatment regimen."
)

_LIFESTYLE_RULE = (
    "\nLifestyle & Dietary Advice: Provide evidence-based, non-prescription guidance for the patient's documented "
    "conditions: dietary recommendations, physical activity, monitoring tips, trigger avoidance. "
    "Do NOT suggest new medicines. Frame as general wellness support."
)

SYSTEM_PROMPT = _BASE_PROMPT + _SIDE_EFFECT_RULE + _LAB_RULE + _LIFESTYLE_RULE

_STOP_WORDS = {
    "a", "an", "and", "are", "about", "at", "for", "from", "how", "i", "is", "it", "me", "my",
    "of", "on", "or", "patient", "please", "show", "tell", "the", "this", "to", "was", "what",
    "when", "with", "you", "did", "do", "does", "can", "any", "which",
}

_RECORD_TERMS = {
    "lab", "labs", "test", "tests", "result", "results", "medicine", "medicines", "medication",
    "medications", "drug", "drugs", "meds", "prescription", "prescriptions", "prescribed", "diagnosis",
    "diagnoses", "condition", "conditions", "visit", "visits", "encounter", "encounters", "followup",
    "follow", "appointment", "vital", "vitals", "blood", "pressure", "pulse", "weight", "temperature",
    "record", "document", "documents", "symptom", "symptoms", "a1c", "hba1c", "glucose", "pain", "fever",
    "rash", "fatigue", "headache", "nausea", "vomiting", "diarrhea", "dizziness", "swelling", "cough",
    "breathing", "urine", "appetite", "sleep", "journey", "timeline", "history", "trajectory", "added",
    "started", "stopped", "discontinued", "restarted", "treatment", "therapy", "clinic", "clinical",
    "doctor", "plan", "complaint", "impression", "note", "notes", "progress", "progression", "trend",
    "trends", "change", "changes", "shift", "shifts", "side", "effect", "effects", "adverse", "indicate",
    "indicates", "indication", "mean", "meaning", "signify", "triglycerides", "lipid", "cholesterol",
    "bp", "spo2", "oxygen", "saturation", "height", "bmi", "temp", "hr", "rate",
}

_VITALS_TERMS = {
    "vital", "vitals", "bp", "blood", "pressure", "pulse", "spo2", "oxygen", "saturation",
    "weight", "height", "bmi", "temp", "temperature", "hr", "rate", "heart",
}

_SYNONYMS = {
    "bp": "blood pressure",
    "sugar": "glucose",
    "a1c": "hba1c",
    "medicine": "medication",
    "medicines": "medication",
    "drug": "medication",
    "drugs": "medication",
    "meds": "medication",
    "followup": "follow up",
    "encounter": "visit",
    "encounters": "visit",
}


def _normalize_query_text(text):
    """Normalize common medical chat tokens like visit3 -> visit 3, encounter2 -> encounter 2."""
    clean = re.sub(r"(?i)\b(visit|encounter|v)\s*#?\s*(\d+)\b", r"\1 \2", str(text))
    return clean


def _tokens(text):
    normalized = _normalize_query_text(text)
    return [token for token in re.findall(r"[a-z0-9]+", normalized.casefold()) if token not in _STOP_WORDS]


def _extract_visit_constraints(question):
    """Extract visit numbers and temporal relation from question (e.g. after visit 3 -> ('after', 3))."""
    text = _normalize_query_text(question).casefold()
    m_after = re.search(r"\b(?:after|since|following|post)\s+(?:visit|encounter|v)?\s*#?\s*(\d+)\b", text)
    if m_after:
        return "after", int(m_after.group(1))
    m_in = re.search(r"\b(?:in|at|during)\s+(?:visit|encounter|v)\s*#?\s*(\d+)\b", text)
    if m_in:
        return "in", int(m_in.group(1))
    m_before = re.search(r"\b(?:before|prior to)\s+(?:visit|encounter|v)\s*#?\s*(\d+)\b", text)
    if m_before:
        return "before", int(m_before.group(1))
    m_count = re.search(r"\b(?:after|in)\s+(\d+)\s+(?:visits?|encounters?)\b", text)
    if m_count:
        return "after", int(m_count.group(1))
    return None, None


def _date(visit):
    return str(visit.get("visit_date") or "Date not recorded")[:10]


def _source(visit, category, content, source_id=None, visit_index=None):
    idx = visit_index or visit.get("visit_index")
    return {
        "date": _date(visit),
        "visit_id": str(visit.get("visit_id") or "Unknown"),
        "visit_index": idx,
        "category": category,
        "source_id": str(source_id or visit.get("visit_id") or "Unknown"),
        "content": str(content).strip(),
    }


def _medication_names(patient_data):
    names = set()
    for visit in patient_data.get("visits", []) or []:
        for item in visit.get("db_medications", []) or []:
            if item.get("name"):
                names.add(str(item["name"]).casefold())
        names.update(str(name).casefold() for name in visit.get("document_medications", []) or [])
    return names


def _patient_diagnoses(patient_data):
    """Extract diagnosis keywords from patient record for dynamic routing."""
    diag_words = set()
    for visit in patient_data.get("visits", []) or []:
        for dx in visit.get("diagnoses", []) or []:
            for word in _tokens(str(dx.get("description") or dx.get("name") or dx)):
                if len(word) > 3:
                    diag_words.add(word)
        for doc in visit.get("documents", []) or []:
            diag = doc.get("diagnosis") or doc.get("impression") or ""
            for word in _tokens(str(diag)):
                if len(word) > 3:
                    diag_words.add(word)
    return diag_words


_LIFESTYLE_TERMS = {
    "lifestyle", "diet", "dietary", "food", "foods", "exercise", "activity", "activities",
    "routine", "manage", "management", "maintain", "maintaining", "control", "controlling",
    "prevent", "prevention", "avoid", "avoidance", "trigger", "triggers", "suggestion",
    "suggestions", "suggest", "idea", "ideas", "advice", "tips", "tip", "recommendation",
    "recommendations", "help", "improve", "improvement", "wellbeing", "wellness",
    "self", "care", "support", "weight", "stress", "sleep", "hydration", "smoking",
    "alcohol", "physical",
}

_CONDITION_TERMS = {
    "diabetes", "diabetic", "t2dm", "type2", "hypertension", "hypertensive", "htn",
    "eczema", "atopic", "dermatitis", "dyslipidemia", "hyperlipidemia", "obesity",
    "asthma", "arthritis", "cholesterol", "lipid", "cardiac", "heart", "kidney", "renal",
    "thyroid", "depression", "anxiety", "insomnia", "gerd", "reflux", "anemia",
}


def route_question(question, patient_data):
    """Classify with whole words so substrings such as latest/test never match."""
    tokens = set(_tokens(question))
    med_names = {word for name in _medication_names(patient_data) for word in _tokens(name)}
    if tokens & _RECORD_TERMS or tokens & med_names:
        return "patient_record"
    if tokens & {"side", "effect", "effects", "interaction", "interactions"} and med_names:
        return "general_medical_about_patient_meds"
    # Lifestyle/dietary advice for patient's conditions
    if tokens & _LIFESTYLE_TERMS:
        if tokens & _CONDITION_TERMS:
            return "patient_record"
        # Also check against patient's actual documented diagnosis keywords
        diag_words = _patient_diagnoses(patient_data)
        if tokens & diag_words:
            return "patient_record"
    return "out_of_scope"



def _latest_visit(patient_data):
    visits = patient_data.get("visits", []) or []
    return visits[-1] if visits else None


def _answer_simple(patient_data, question, meta=None):
    """Deterministic fast paths for exact benchmark/offline queries without hijacking nuanced questions."""
    q = set(_tokens(question))
    visits = patient_data.get("visits", []) or []
    if not visits:
        return None
    latest = _latest_visit(patient_data)

    # Detect if user is asking for interpretation / explanation / meaning
    is_interpret_query = bool(q & {
        "indicate", "indicates", "indication", "mean", "means", "meaning", "signify", "signifies",
        "suggest", "suggests", "imply", "implies", "explain", "why", "how", "interpretation",
        "interpret", "cause", "causes", "reason", "reasons", "risk", "risks", "effect", "effects",
        "impact", "concern", "worry", "side", "reaction", "reactions",
    })
    rel, v_num = _extract_visit_constraints(question)
    is_change_query = bool(q & {"change", "changed", "changes", "started", "stopped", "restarted", "discontinued", "added", "new", "prescribed", "after", "since", "before", "between", "difference"})

    # 1. Lab Lookup (deterministic — abnormal, normal, latest, all) — NOT for interpretation
    wants_lab = bool(q & {"lab", "labs", "test", "tests", "result", "results", "a1c", "hba1c", "glucose", "finding", "findings", "report"})
    abnormal = bool(q & {"abnormal", "abnormal", "elevated", "decreased"}) or (("high" in q or "low" in q) and wants_lab)
    wants_normal = bool(q & {"normal"}) and wants_lab and not abnormal
    wants_all_labs = wants_lab and not is_interpret_query and not abnormal and not wants_normal and not ("latest" in q)

    if wants_lab and not is_interpret_query:
        # Abnormal labs
        if abnormal or ("latest" in q and abnormal):
            for visit in reversed(visits):
                labs = [lab for lab in (visit.get("lab_results") or []) if any(w in str(lab.get("status") or "").casefold() for w in ("abnormal", "high", "low", "elevated", "decreased"))]
                if labs:
                    text = "; ".join(f"{lab.get('test_name')}: {lab.get('value')} [{lab.get('status', '')}]" for lab in labs)
                    ev = [_source(visit, "labs", text, visit_index=len(visits))]
                    if meta is not None: meta["rule"] = "lab_lookup"
                    return f"### Answer\n{text}\n\n### Evidence\n- {_date(visit)} · Visit {visit.get('visit_id')}", ev
            if meta is not None: meta["rule"] = "lab_lookup"
            return "### Answer\nNo abnormal laboratory result documented.\n\n### Evidence\n- No abnormal lab evidence.", []

        # Latest labs
        if "latest" in q or "last" in q or "recent" in q:
            for visit in reversed(visits):
                labs = visit.get("lab_results") or []
                if labs:
                    text = "; ".join(f"{lab.get('test_name')}: {lab.get('value')} [{lab.get('status', '')}]" for lab in labs)
                    ev = [_source(visit, "labs", text, visit_index=len(visits))]
                    if meta is not None: meta["rule"] = "lab_lookup"
                    return f"### Answer\n{text}\n\n### Evidence\n- {_date(visit)} · Visit {visit.get('visit_id')}", ev
            if meta is not None: meta["rule"] = "lab_lookup"
            return "### Answer\nNo laboratory results documented.\n\n### Evidence\n- No lab evidence.", []

        # Normal labs
        if wants_normal:
            for visit in reversed(visits):
                labs = [lab for lab in (visit.get("lab_results") or []) if "normal" in str(lab.get("status") or "").casefold() and "abnormal" not in str(lab.get("status") or "").casefold()]
                if labs:
                    text = "; ".join(f"{lab.get('test_name')}: {lab.get('value')} [{lab.get('status', '')}]" for lab in labs)
                    ev = [_source(visit, "labs", text, visit_index=len(visits))]
                    if meta is not None: meta["rule"] = "lab_lookup"
                    return f"### Answer\n{text}\n\n### Evidence\n- {_date(visit)} · Visit {visit.get('visit_id')}", ev

        # All labs across encounters
        if wants_all_labs and not is_change_query:
            all_lab_enc = []
            ev_list = []
            for idx, v in enumerate(visits, 1):
                labs = v.get("lab_results") or []
                if labs:
                    lines = [f"{lab.get('test_name')}: {lab.get('value')} [{lab.get('status', '')}]" for lab in labs]
                    all_lab_enc.append(f"Encounter #{idx} ({_date(v)}): " + "; ".join(lines))
                    ev_list.append(_source(v, "laboratory result", "; ".join(lines), visit_index=idx))
            if all_lab_enc:
                if meta is not None: meta["rule"] = "lab_lookup"
                return f"### Answer\nLaboratory results across encounters:\n" + "\n".join(f"- {l}" for l in all_lab_enc) + f"\n\n### Evidence\n" + "\n".join(f"- Encounter #{e.get('visit_index','?')} · {e['date']} · Visit {e['visit_id']}" for e in ev_list[:4]), ev_list

    # 2. Stopped/Discontinued medicines specifically requested
    is_stopped_query = bool(q & {"stopped", "discontinued", "stop"})
    if is_stopped_query and q & {"medication", "medications", "medicine", "medicines", "drug", "drugs"}:
        journey = compute_medication_journey(patient_data)
        changes = [(item, change) for item in journey for change in item.get("changes", []) if change.get("type") == "discontinued"]
        if rel == "after" and v_num is not None:
            changes = [(item, change) for item, change in changes if item.get("visit_index", 0) > v_num]
        evidence = [_source(next((v for v in visits if str(v.get("visit_id")) == str(item["visit_id"])), latest), "medication change", change["text"], visit_index=item.get("visit_index")) for item, change in changes]
        if not evidence:
            text = "No medicine is explicitly documented as stopped in the requested visits."
        else:
            text = "\n".join(f"- Encounter #{item.get('visit_index', '?')} ({item['date']}): {item['content']}" for item in evidence)
        if meta is not None:
            meta["rule"] = "stopped_meds"
        return f"### Answer\n{text}\n\n### Evidence\n" + ("\n".join(f"- {item['date']} · Visit {item['visit_id']}" for item in evidence) or "- No medication-change evidence found."), evidence

    # 2a. Conditions & Diagnoses Lookup (Deterministic from structured record)
    is_dx_query = bool(q & {"condition", "conditions", "diagnosis", "diagnoses", "disease", "diseases"})
    if is_dx_query and not is_interpret_query and not (q & _LIFESTYLE_TERMS) and v_num is None:
        dx_by_enc = []
        ev_list = []
        all_dx = set()
        for idx, v in enumerate(visits, 1):
            conf = v.get("db_confirmed_dx") or v.get("confirmed_diagnosis")
            prov = v.get("db_provisional_dx") or v.get("provisional_diagnosis")
            parts = []
            if conf:
                parts.append(f"Confirmed: {conf}")
                all_dx.add(str(conf))
            if prov:
                parts.append(f"Provisional: {prov}")
                all_dx.add(str(prov))
            if parts:
                dx_by_enc.append(f"Encounter #{idx} ({_date(v)}): " + "; ".join(parts))
                ev_list.append(_source(v, "patient diagnoses overview", "; ".join(parts), visit_index=idx))
        if dx_by_enc:
            ans_text = "Documented medical conditions across encounters:\n" + "\n".join(f"- {line}" for line in dx_by_enc) + f"\n\n**Primary documented condition(s):** " + ", ".join(sorted(all_dx))
            if meta is not None:
                meta["rule"] = "diagnoses_lookup"
            return f"### Answer\n{ans_text}\n\n### Evidence\n" + "\n".join(f"- Encounter #{e.get('visit_index', '?')} · {e['date']} · Visit {e['visit_id']}" for e in ev_list[:4]), ev_list

    # 2b. All Medications Given / Prescribed (Deterministic from structured record)
    is_med_query = bool(q & {"medication", "medications", "medicine", "medicines", "drug", "drugs", "meds", "prescriptions", "prescription"})
    is_se_query = bool(q & {"side", "effect", "effects", "adverse", "reaction", "warning", "toxicity"})
    if is_med_query and not is_se_query and not is_stopped_query and not is_interpret_query and not is_change_query and v_num is None:
        if not (q & {"current", "active"}):
            enc_meds = []
            ev_list = []
            all_known = set()
            for idx, v in enumerate(visits, 1):
                rx = [m for m in (v.get("db_medications") or []) if m.get("name")]
                if rx:
                    m_names = [f"{m.get('name')} {m.get('dosage','')}".strip() for m in rx]
                    enc_meds.append(f"Encounter #{idx} ({_date(v)}): " + "; ".join(m_names))
                    ev_list.append(_source(v, "medications", "; ".join(m_names), visit_index=idx))
                    all_known.update(m.get("name") for m in rx)
            if enc_meds:
                ans_text = "Documented medications across encounters:\n" + "\n".join(f"- {line}" for line in enc_meds) + f"\n\n**All distinct prescribed medications ({len(all_known)}):** " + ", ".join(sorted(all_known))
                if meta is not None:
                    meta["rule"] = "all_meds"
                return f"### Answer\n{ans_text}\n\n### Evidence\n" + "\n".join(f"- Encounter #{e.get('visit_index', '?')} · {e['date']} · Visit {e['visit_id']}" for e in ev_list[:4]), ev_list

    # 3. Strictly Active/Current medication roster (ONLY when specifically asking for current/active list, without temporal or change queries)
    if (q & {"current", "active"}) and not is_change_query and v_num is None:
        journey = compute_medication_journey(patient_data)
        active = journey[-1].get("all_active_meds", []) if journey else []
        if not active and latest:
            active = [item.get("name") for item in latest.get("db_medications", []) if item.get("name")]
        text = ", ".join(active) if active else "No active medication is documented in the latest encounter."
        evidence = [_source(latest, "medications", text, visit_index=len(visits))] if latest else []
        if meta is not None:
            meta["rule"] = "active_meds"
        return f"### Answer\n{text}\n\n### Evidence\n- {_date(latest)} · Visit {latest.get('visit_id')}", evidence

    # 3a. Visit / history summary fast-path (no LLM needed)
    _S = {'summarize', 'summarise', 'summary', 'overview', 'happen', 'happened', 'recap', 'details', 'detail', 'brief'}
    _T = {'lab', 'labs', 'test', 'tests', 'result', 'results', 'medication', 'medications', 'medicine', 'medicines', 'drug', 'drugs', 'vital', 'vitals', 'bp', 'side', 'effect', 'effects', 'followup', 'follow'}
    _W = {'history', 'record', 'medical', 'journey', 'timeline', 'course'}
    _n_match = re.search(r"\b(?:visit|encounter|v)\s*#?\s*(\d+)\b", _normalize_query_text(question).casefold())
    _is_summary_q = (
        bool(q & _S)
        and not bool(q & _T)
        and not is_interpret_query
        and (_n_match or bool(q & _W) or "latest" in q or "last" in q or len(q - _S) <= 1)
    )
    if _is_summary_q:
        journey = compute_medication_journey(patient_data)
        journey_by_vid2 = {str(enc.get("visit_id")): enc for enc in journey}

        def _visit_digest(visit, idx, jbv):
            lines = []
            cc = visit.get("db_chief_complaint") or visit.get("chief_complaint")
            if cc:
                lines.append(f"- Chief complaint: {cc}")
            conf = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")
            prov = visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
            dx_parts = []
            if conf:
                dx_parts.append(f"Confirmed: {conf}")
            if prov:
                dx_parts.append(f"Provisional: {prov}")
            if dx_parts:
                lines.append("- Diagnosis: " + "; ".join(dx_parts))
            rx_list = [m for m in (visit.get("db_medications") or []) if m.get("name")]
            if rx_list:
                rx_strs = [" ".join(str(m.get(k, "")).strip() for k in ("name", "dosage", "frequency") if m.get(k)) for m in rx_list]
                lines.append("- Medications: " + "; ".join(rx_strs))
            enc_j = jbv.get(str(visit.get("visit_id")))
            if enc_j:
                chg = [c.get("text", "") for c in enc_j.get("changes", []) if c.get("type") in ("started", "dose_changed", "discontinued", "restarted")]
                if chg:
                    lines.append("- Changes: " + "; ".join(chg))
                mismatch_count = sum(1 for c in enc_j.get("changes", []) if c.get("type") in ("newly_documented", "mismatch"))
                if mismatch_count:
                    lines.append(f"- {mismatch_count} record mismatch note(s)")
            labs = visit.get("lab_results") or []
            abn = [f"{l.get('test_name')}: {l.get('value')} [{l.get('status')}]" for l in labs if any(w in str(l.get("status", "")).casefold() for w in ("abnormal", "high", "low"))]
            other_lab_count = len(labs) - len(abn)
            if abn:
                lines.append("- Abnormal labs: " + "; ".join(abn[:3]) + (f" (+{len(abn)-3} more)" if len(abn) > 3 else ""))
            if other_lab_count > 0:
                lines.append(f"- {other_lab_count} other lab(s) not flagged")
            rows = [r for r in (visit.get("db_vitals_rows") or []) if isinstance(r, dict) and r.get("label")]
            if rows:
                lines.append("- Vitals: " + ", ".join(f"{r['label']}: {r.get('value', '')}" for r in rows[:4]))
            followups = visit.get("followups") or []
            fu_counts = {}
            for fu in followups:
                st = str(fu.get("status", "unknown"))
                fu_counts[st] = fu_counts.get(st, 0) + 1
            if fu_counts:
                lines.append("- Follow-up: " + ", ".join(f"{s}: {c}" for s, c in fu_counts.items()))
            docs = visit.get("documents") or []
            if docs:
                lines.append(f"- {len(docs)} clinical document(s)")
            return "\n".join(lines)

        # Single visit?
        if _n_match or "latest" in q or "last" in q:
            if _n_match:
                n_req = int(_n_match.group(1))
                if n_req > len(visits):
                    if meta is not None:
                        meta["rule"] = "visit_summary"
                    return f"### Answer\nEncounter {n_req} is not in the record ({len(visits)} documented).\n\n### Evidence\n- No evidence", []
                target_visit = visits[n_req - 1]
                t_idx = n_req
            else:
                target_visit = visits[-1]
                t_idx = len(visits)
            digest = _visit_digest(target_visit, t_idx, journey_by_vid2)
            ev_line = f"Encounter #{t_idx} · {_date(target_visit)} · Visit {target_visit.get('visit_id')}"
            evidence = [_source(target_visit, "encounter summary", digest, f"encounter:{t_idx}", visit_index=t_idx)]
            if meta is not None:
                meta["rule"] = "visit_summary"
            return f"### Answer\n{digest}\n\n### Evidence\n- {ev_line}", evidence

        # Whole history
        first_d = _date(visits[0])
        last_d = _date(visits[-1])
        show_visits = visits if len(visits) <= 6 else visits[-6:]
        show_offset = len(visits) - len(show_visits)
        parts = []
        if show_offset:
            parts.append(f"(Showing latest {len(show_visits)} of {len(visits)} encounters)")
        for si, sv in enumerate(show_visits, start=show_offset + 1):
            parts.append(f"\n**Encounter #{si} · {_date(sv)}**\n" + _visit_digest(sv, si, journey_by_vid2))
        active = journey[-1].get("all_active_meds", []) if journey else []
        if active:
            parts.append("\n**Active medicines:** " + ", ".join(active))
        answer_text = f"{len(visits)} encounter(s), {first_d} to {last_d}\n" + "\n".join(parts)
        all_evidence = [_source(v, "encounter summary", _visit_digest(v, i, journey_by_vid2), f"encounter:{i}", visit_index=i) for i, v in enumerate(visits, 1)]
        if meta is not None:
            meta["rule"] = "history_summary"
        return f"### Answer\n{answer_text}\n\n### Evidence\n- {len(visits)} encounter(s) from {first_d} to {last_d}", all_evidence

    # 4. Vitals lookup (latest OR all encounters)
    if (
        q & _VITALS_TERMS
        and not is_interpret_query
        and not q & {"trend", "trends", "compare", "over", "progress", "tips", "advice", "manage"}
    ):
        if q & {"latest", "last", "current", "recent"} or not q & {"all", "every", "history", "across", "each"}:
            for v in reversed(visits):
                rows = [r for r in (v.get("db_vitals_rows") or []) if isinstance(r, dict) and r.get("label")]
                if rows:
                    text = "; ".join(f"{r['label']}: {r.get('value', '')}" for r in rows)
                    evidence = [_source(v, "vitals", text, visit_index=len(visits))]
                    if meta is not None: meta["rule"] = "vitals_latest"
                    return f"### Answer\n{text}\n\n### Evidence\n- {_date(v)} · Visit {v.get('visit_id')}", evidence
        else:
            all_vitals = []
            ev_list = []
            for idx, v in enumerate(visits, 1):
                rows = [r for r in (v.get("db_vitals_rows") or []) if isinstance(r, dict) and r.get("label")]
                if rows:
                    row_str = "; ".join(f"{r['label']}: {r.get('value', '')}" for r in rows)
                    all_vitals.append(f"Encounter #{idx} ({_date(v)}): {row_str}")
                    ev_list.append(_source(v, "vitals", row_str, visit_index=idx))
            if all_vitals:
                if meta is not None: meta["rule"] = "vitals_all"
                return f"### Answer\nVitals across encounters:\n" + "\n".join(f"- {l}" for l in all_vitals) + f"\n\n### Evidence\n" + "\n".join(f"- Encounter #{e.get('visit_index','?')} · {e['date']}" for e in ev_list[:4]), ev_list

    # 5. Record aggregate counts
    if q & {"count", "number"} and ("many" in q or "how" in q or "visits" in q or "encounters" in q):
        evidence = [_source(latest, "record count", f"Encounters: {len(visits)}; medications: {len(patient_data.get('all_medications', []))}", visit_index=len(visits))]
        if meta is not None:
            meta["rule"] = "counts"
        return f"### Answer\nThe record has {len(visits)} encounter(s) and {len(patient_data.get('all_medications', []))} distinct medication name(s).\n\n### Evidence\n- Record aggregate", evidence

    return None


def build_patient_index(patient_data):
    """Build comprehensive, encounter-aware local evidence once per patient."""
    entries = []
    visits = patient_data.get("visits", []) or []
    journey = compute_medication_journey(patient_data)
    journey_by_vid = {str(enc.get("visit_id")): enc for enc in journey}

    # 1. Full Longitudinal Medication Journey Overview (cross-encounter timeline)
    if journey:
        journey_lines = []
        for enc in journey:
            idx = enc.get("visit_index", 1)
            date = enc.get("visit_date", "Date not recorded")
            shifts = [c.get("text", "") for c in enc.get("changes", [])]
            shift_text = "; ".join(shifts) if shifts else "No medication shifts documented"
            active_str = ", ".join(enc.get("all_active_meds", []))
            journey_lines.append(f"Encounter #{idx} ({date}): Shifts: {shift_text} | Active: {active_str}")
        entries.append({
            "date": "Longitudinal",
            "visit_id": "All Encounters",
            "visit_index": 0,
            "category": "longitudinal medication journey",
            "source_id": "medication_journey",
            "content": "Chronological Medication Shifts Timeline across all encounters:\n" + "\n".join(journey_lines),
        })

    # 2. Per-Encounter Detailed Indexing
    for idx, visit in enumerate(visits, start=1):
        v_date = _date(visit)
        v_id = str(visit.get("visit_id") or f"V{idx}")
        enc_prefix = f"Encounter #{idx} ({v_date}, Visit {v_id})"

        # A. Comprehensive Encounter Summary Card
        enc_summary = [enc_prefix]
        cc = visit.get("db_chief_complaint") or visit.get("chief_complaint")
        if cc:
            enc_summary.append(f"Chief Complaint: {cc}")
        conf = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")
        prov = visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
        if conf or prov:
            dx_strs = []
            if conf:
                dx_strs.append(f"Confirmed Diagnosis: {conf}")
            if prov:
                dx_strs.append(f"Provisional Diagnosis: {prov}")
            enc_summary.append("; ".join(dx_strs))

        rx_list = visit.get("db_medications", []) or []
        if rx_list:
            rx_names = ", ".join(f"{m.get('name', '')} {m.get('dosage', '')} {m.get('frequency', '')}".strip() for m in rx_list if m.get("name"))
            enc_summary.append(f"Prescriptions: {rx_names}")

        enc_journey = journey_by_vid.get(v_id)
        if enc_journey and enc_journey.get("changes"):
            changes_text = "; ".join(c.get("text", "") for c in enc_journey["changes"])
            enc_summary.append(f"Medication Changes: {changes_text}")

        labs = visit.get("lab_results", []) or []
        if labs:
            abnormal_labs = [f"{l.get('test_name')}: {l.get('value')} [{l.get('status')}]" for l in labs if any(w in str(l.get("status", "")).casefold() for w in ("abnormal", "high", "low"))]
            if abnormal_labs:
                enc_summary.append(f"Abnormal Labs: {'; '.join(abnormal_labs)}")

        entries.append(_source(visit, "encounter summary", " | ".join(enc_summary), f"encounter:{idx}", visit_index=idx))

        # B. Granular Medications entry (Category: "medications" - required by tests)
        med_items = [
            " ".join(str(item.get(key, "")).strip() for key in ("name", "dosage", "frequency") if item.get(key))
            for item in rx_list
        ]
        if med_items:
            entries.append(_source(visit, "medications", f"{enc_prefix}: Prescriptions: {'; '.join(med_items)}", "medications", visit_index=idx))

        # C. Granular Medication Shifts
        if enc_journey:
            for ch in enc_journey.get("changes", []):
                entries.append(_source(visit, "medication change", f"{enc_prefix} Medication Shift [{ch.get('type')}]: {ch.get('text')}", ch.get("medication"), visit_index=idx))

        # D. Granular Lab results (Category: "laboratory result" - required by tests)
        for lab in labs:
            content = " ".join(str(lab.get(key, "")).strip() for key in ("test_name", "value", "reference", "status") if lab.get(key))
            if content:
                entries.append(_source(visit, "laboratory result", f"{enc_prefix} Lab: {content}", lab.get("test_name"), visit_index=idx))

        # E. Vitals
        rows = [r for r in (visit.get("db_vitals_rows") or []) if isinstance(r, dict) and r.get("label")]
        if rows:
            entries.append(_source(visit, "vitals", f"{enc_prefix} Vitals (record): " + ", ".join(f"{r['label']}: {r.get('value', '')}" for r in rows), "vitals", visit_index=idx))
        elif isinstance(visit.get("db_vitals_json"), dict) and visit.get("db_vitals_json"):
            entries.append(_source(visit, "vitals", f"{enc_prefix} Vitals (record): " + ", ".join(f"{k}: {v}" for k, v in visit["db_vitals_json"].items()), "vitals", visit_index=idx))

        # F. Follow-ups
        for followup in visit.get("followups", []) or []:
            content = " ".join(str(followup.get(key, "")).strip() for key in ("scheduled_date", "status") if followup.get(key))
            if content:
                entries.append(_source(visit, "follow-up", f"{enc_prefix} Follow-up: {content}", "follow-up", visit_index=idx))

        # G. Documents
        for doc in visit.get("documents", []) or []:
            doc_label = doc.get("document_label", "Clinical Document")
            cs = doc.get("clinical_summary", {}) or {}
            cs_parts = []
            if cs.get("clinical_impression"):
                cs_parts.append(f"Impression: {cs['clinical_impression']}")
            if cs.get("symptoms"):
                cs_parts.append(f"Symptoms: {', '.join(cs['symptoms'])}")
            if cs_parts:
                entries.append(_source(visit, "document findings", f"{enc_prefix} Document '{doc_label}' Findings: {' | '.join(cs_parts)}", f"doc:{doc.get('doc_id')}", visit_index=idx))

            doc_vitals = [r for r in (cs.get("vitals") or []) if isinstance(r, dict) and r.get("label")]
            if doc_vitals:
                entries.append(_source(visit, "vitals", f"{enc_prefix} Vitals (document '{doc_label}'): " + ", ".join(f"{r['label']}: {r.get('value', '')}" for r in doc_vitals), f"doc:{doc.get('doc_id')}", visit_index=idx))

            words = " ".join(str(line).strip() for line in (doc.get("extracted_text", []) or []) if str(line).strip()).split()
            for offset in range(0, len(words), 60):
                chunk = " ".join(words[offset:offset + 60])
                if chunk:
                    entries.append(_source(visit, "document", f"{enc_prefix} Doc text: {chunk}", f"doc:{doc.get('doc_id', 'unknown')}:{offset // 60 + 1}", visit_index=idx))

    # 3. Comprehensive Side Effects Reference for Patient Medications
    possible_se = patient_data.get("possible_side_effects") or {}
    if not possible_se:
        try:
            from analysis import get_possible_side_effects
            all_meds = patient_data.get("all_medications", []) or list(_medication_names(patient_data))
            possible_se = get_possible_side_effects(sorted(all_meds), use_ai=False)
        except Exception:
            possible_se = {}

    for medicine, effects in possible_se.items():
        if effects and "isn't in our local reference list" not in str(effects):
            entries.append({
                "date": "Reference",
                "visit_id": "Patient medication list",
                "visit_index": 0,
                "category": "medication side effects reference",
                "source_id": str(medicine),
                "content": (
                    f"Medication Side Effects Reference for {medicine}: "
                    f"Common documented side effects, adverse reactions, and warnings: {effects}. "
                    f"Prescribed for this patient in their clinical regimen."
                ),
            })

    # 4. Patient Diagnoses & Clinical Overview (for lab interpretation & lifestyle advice)
    all_diagnoses = []
    for visit in visits:
        idx = visits.index(visit) + 1
        conf = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")
        prov = visit.get("db_provisional_dx") or visit.get("provisional_diagnosis")
        if conf:
            all_diagnoses.append(f"Encounter #{idx}: Confirmed: {conf}")
        if prov:
            all_diagnoses.append(f"Encounter #{idx}: Provisional: {prov}")
    if all_diagnoses:
        entries.append({
            "date": "Longitudinal",
            "visit_id": "All Encounters",
            "visit_index": 0,
            "category": "patient diagnoses overview",
            "source_id": "diagnoses_overview",
            "content": (
                "Patient's documented conditions and diagnoses across all encounters:\n"
                + "\n".join(all_diagnoses)
                + "\nThis provides clinical context for lab interpretation, lifestyle advice, and treatment decisions."
            ),
        })

    # 5. Neo4j graph if configured
    if is_neo4j_configured():
        try:
            entries.extend(query_patient_graph(str(patient_data.get("patient_id", ""))))
        except Exception as exc:
            logger.warning("Neo4j chat retrieval failed: %s", exc)

    return [entry for entry in entries if entry["content"]]


def _bm25(query, entries, limit=10, target_visit_rel=None, target_visit_num=None):
    terms = _tokens(query)
    if not terms:
        return []
    docs = [Counter(_tokens(entry["content"])) for entry in entries]
    total = len(docs) or 1
    average = sum(sum(doc.values()) for doc in docs) / total or 1
    scores = []
    is_side_effect_q = any(w in terms for w in ("side", "effect", "effects", "adverse", "reaction", "reactions", "toxicity", "risk", "warning"))
    is_lab_q = any(w in terms for w in ("lab", "labs", "test", "hba1c", "triglycerides", "lipid", "cholesterol", "glucose", "pcv", "tlc", "indicate", "indication", "mean", "means"))
    is_vitals_q = any(w in terms for w in _VITALS_TERMS)
    is_lifestyle_q = any(w in terms for w in ("lifestyle", "diet", "dietary", "food", "exercise", "maintain", "manage", "suggest", "advice", "tips", "idea", "prevent", "avoid", "trigger", "wellness", "self", "care"))

    for entry, doc in zip(entries, docs):
        score = 0.0

        for term in terms:
            document_frequency = sum(1 for candidate in docs if term in candidate)
            if document_frequency:
                inverse_frequency = math.log((total - document_frequency + 0.5) / (document_frequency + 0.5) + 1)
                count = doc[term]
                score += inverse_frequency * (count * 2.2 / (count + 1.2 * (1 - 0.75 + 0.75 * sum(doc.values()) / average)))

        # Temporal constraint boost
        v_idx = entry.get("visit_index")
        if target_visit_rel and target_visit_num is not None and v_idx is not None and v_idx > 0:
            if target_visit_rel == "after" and v_idx > target_visit_num:
                score += 5.0
            elif target_visit_rel == "in" and v_idx == target_visit_num:
                score += 6.0
            elif target_visit_rel == "before" and v_idx < target_visit_num:
                score += 5.0

        # Category relevance boost
        cat = entry.get("category", "")
        if cat in ("encounter summary", "medication change", "longitudinal medication journey"):
            score += 2.0
        if is_side_effect_q and cat == "medication side effects reference":
            src = str(entry.get("source_id", "")).lower()
            if any(t in src or src in t for t in terms if len(t) >= 4):
                score += 15.0
            else:
                score -= 10.0
        for t in terms:
            if len(t) >= 4 and t in entry.get("content", "").lower():
                score += 3.0
        if is_lab_q and cat in ("laboratory result", "encounter summary", "patient diagnoses overview"):
            score += 4.0
        if is_lifestyle_q and cat in ("patient diagnoses overview", "encounter summary", "longitudinal medication journey"):
            score += 5.0
        if is_vitals_q and cat == "vitals":
            score += 4.0

        if score > 0:
            scores.append((score, entry))
    return [entry for _, entry in sorted(scores, key=lambda item: item[0], reverse=True)[:limit]]


def _retrieval_query(question, history):
    normalized = " ".join(_SYNONYMS.get(token, token) for token in _tokens(question))
    prior = next((item.get("content", "") for item in reversed(history or []) if item.get("role") == "user"), "")
    return f"{normalized} {' '.join(_tokens(prior)[:12])}".strip()


def _answer_patient_question(patient_data, patient_id, question, history=None, api_key=None, retrieval_index=None, source_hash=None, meta=None):
    meta = meta if meta is not None else {}
    meta.setdefault("prompt_tokens", 0)
    meta.setdefault("completion_tokens", 0)
    meta.setdefault("total_tokens", 0)
    meta.setdefault("rule", None)
    meta.setdefault("model", None)
    meta.setdefault("error", None)

    if not patient_data or str(patient_data.get("patient_id", "")).strip() != str(patient_id).strip():
        meta["path"] = "unavailable"
        return "### Answer\nThe selected patient record is unavailable.\n\n### Evidence\n- No patient-scoped evidence available.", []
    if route_question(question, patient_data) == "out_of_scope":
        meta["path"] = "out_of_scope"
        return "### Answer\nI can answer questions about this patient's record and documented medicines only.\n\n### Evidence\n- No record lookup was performed.", []

    simple = _answer_simple(patient_data, question, meta=meta)
    if simple:
        meta["path"] = "deterministic"
        return simple

    index = retrieval_index if retrieval_index is not None else build_patient_index(patient_data)
    rel, v_num = _extract_visit_constraints(question)
    q_tokens = _tokens(question)
    is_side_effect_q = any(w in q_tokens for w in ("side", "effect", "effects", "adverse", "reaction", "reactions", "toxicity", "warning"))
    is_lab_q = any(w in q_tokens for w in ("lab", "labs", "test", "hba1c", "triglycerides", "lipid", "cholesterol", "glucose", "pcv", "tlc", "indicate", "indication", "abnormal", "mean", "means"))
    is_lifestyle_q = any(w in q_tokens for w in ("lifestyle", "diet", "dietary", "food", "exercise", "maintain", "manage", "suggest", "advice", "tips", "idea", "prevent", "avoid", "trigger", "wellness", "self", "care"))

    evidence = _bm25(_retrieval_query(question, history), index, limit=10, target_visit_rel=rel, target_visit_num=v_num)

    # 1. Temporal injection
    if rel and v_num is not None:
        for entry in index:
            v_idx = entry.get("visit_index")
            if v_idx and v_idx > 0:
                is_match = (
                    (rel == "after" and v_idx > v_num)
                    or (rel == "in" and v_idx == v_num)
                    or (rel == "before" and v_idx < v_num)
                )
                if is_match and entry.get("category") in ("medication change", "encounter summary"):
                    if entry not in evidence:
                        evidence.append(entry)

    # 2. Side effect injection: fuzzy-match drug name (handles OCR typos e.g. Domperidome→Domperidone)
    if is_side_effect_q:
        q_low = question.lower()
        q_long_tokens = [t for t in q_tokens if len(t) >= 4]  # skip tiny words
        def _drug_match(name_str):
            n = name_str.lower()
            if not n: return False
            if n in q_low or q_low in n: return True
            # Fuzzy: any 5-char prefix overlap
            return any(t[:5] == n[:5] or n[:5] in t or t[:5] in n for t in q_long_tokens if len(t) >= 5 and len(n) >= 5)
        for entry in index:
            if entry.get("category") == "medication side effects reference":
                if _drug_match(entry.get("source_id", "")):
                    if entry not in evidence:
                        evidence.insert(0, entry)
                # Also inject when LLM can use all patient's medication names as context
                elif entry not in evidence and len(evidence) < 4:
                    evidence.append(entry)  # give LLM patient-specific med context
            elif entry.get("category") in ("medication change", "medications"):
                med_name = str(entry.get("source_id", "")).lower()
                if med_name and _drug_match(med_name):
                    if entry not in evidence:
                        evidence.append(entry)

    # 3. Lab injection: include the specific lab entry + documents with clinical impression about it
    if is_lab_q:
        q_long = [t for t in q_tokens if len(t) >= 4]
        for entry in index:
            cat = entry.get("category", "")
            content_low = entry.get("content", "").lower()
            if cat in ("laboratory result", "encounter summary", "patient diagnoses overview"):
                # Match if any query lab-name token appears in content
                if any(t in content_low for t in q_long) or any(w in content_low for w in ("abnormal", "hba1c", "triglycerides", "glucose", "pcv", "diabetes", "t2dm", "hypert")):
                    if entry not in evidence:
                        evidence.append(entry)
            elif cat in ("document", "document findings"):
                # Clinical documents often have impressions about what elevated/low values mean
                if any(t in content_low for t in q_long):
                    if entry not in evidence:
                        evidence.append(entry)

    # 4. Lifestyle/dietary advice injection: always include diagnoses overview + encounter summaries
    if is_lifestyle_q:
        for entry in index:
            if entry.get("category") in ("patient diagnoses overview", "encounter summary"):
                if entry not in evidence:
                    evidence.append(entry)

    if not evidence:
        meta["path"] = "no_evidence"
        return "### Answer\nThe requested information is not documented in the available patient record.\n\n### Evidence\n- No relevant patient-record evidence found.", []

    # Drop duplicate content
    _seen_content = set()
    _deduped = []
    for _e in evidence:
        _key = _e.get("content", "")[:120]
        if _key not in _seen_content:
            _seen_content.add(_key)
            _deduped.append(_e)
    evidence = _deduped

    # If an encounter summary is present for a visit, drop that visit's
    # medications/medication-change entries (unless user is asking about changes/shifts)
    _med_change_tokens = {"change", "changed", "started", "stopped", "added", "discontinued",
                          "restarted", "dose", "shift", "shifts", "journey"}
    _keep_med_detail = is_side_effect_q or bool(set(q_tokens) & _med_change_tokens)
    if not _keep_med_detail:
        _enc_summary_vids = {e.get("visit_id") for e in evidence if e.get("category") == "encounter summary"}
        evidence = [e for e in evidence if not (
            (e.get("category") in ("medications", "medication change") and e.get("visit_id") in _enc_summary_vids)
            or (bool(_enc_summary_vids) and e.get("category") == "longitudinal medication journey")
        )]

    packed, used = [], 0
    for item in evidence:
        enc_lbl = f"Encounter #{item.get('visit_index')}" if item.get("visit_index") else "Reference/Longitudinal"
        text = f"{item['date']} | {enc_lbl} | Visit {item['visit_id']} | [{item['category']}]\n{item['content']}"
        if len(packed) >= 8 or used + len(text.split()) > 700:
            break
        packed.append(text)
        used += len(text.split())

    normalized_question = " ".join(_tokens(question))
    vitals_sig = hashlib.blake2b(json.dumps([v.get("db_vitals_json") for v in patient_data.get("visits", [])], sort_keys=True, default=str).encode(), digest_size=8).hexdigest()
    cache_key = "chat:v12:" + hashlib.blake2b(f"{source_hash or ''}|{vitals_sig}|{normalized_question}".encode(), digest_size=16).hexdigest()
    _packed_evidence = [item for item in evidence
                         if any(item.get("content", "")[:120] in t for t in packed)]
    cached = get_llm_cache(cache_key)
    if cached:
        meta["path"] = "llm_cache"
        meta["prompt_tokens"] = 0
        meta["completion_tokens"] = 0
        meta["total_tokens"] = 0
        return json.loads(cached), _packed_evidence

    _sys = _BASE_PROMPT
    if is_side_effect_q:
        _sys += _SIDE_EFFECT_RULE
    if is_lab_q:
        _sys += _LAB_RULE
    if is_lifestyle_q:
        _sys += _LIFESTYLE_RULE
    messages = [{"role": "system", "content": _sys}]
    for item in (history or [])[-2:]:
        if item.get("role") in {"user", "assistant"}:
            messages.append({"role": item["role"], "content": str(item.get("content") or "")[:400]})
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nPatient Record Evidence:\n" + "\n\n".join(packed),
    })

    try:
        response = llm_complete(messages, "chat", api_key=api_key, model="google/gemini-2.5-flash", temperature=0.1, max_tokens=450, cache=False)
        meta["path"] = "llm"
        usage = response.get("usage") or {}
        meta.update(
            model=response.get("model"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        answer = response["content"].strip()
        if "### Answer" not in answer:
            answer = "### Answer\n" + answer
        if "### Evidence" not in answer:
            ev_lines = []
            for item in evidence[:6]:
                enc_tag = f"Encounter #{item['visit_index']} · " if item.get("visit_index") else ""
                ev_lines.append(f"- {enc_tag}{item['date']} · Visit {item['visit_id']} ({item.get('category', 'Evidence')})")
            answer += "\n\n### Evidence\n" + "\n".join(ev_lines)
        try:
            set_llm_cache(cache_key, json.dumps(answer, ensure_ascii=False))
        except Exception as cache_exc:
            logger.warning("Failed to write LLM cache: %s", cache_exc)
    except Exception as exc:
        meta["path"] = "fallback"
        meta["error"] = str(exc)[:100]
        logger.warning("Patient chat generation failed: %s", exc)
        # Deterministic clinical fallback synthesis
        if is_side_effect_q:
            se_entry = next((e for e in evidence if e.get("category") == "medication side effects reference"), None)
            med_add_entry = next((e for e in evidence if e.get("category") == "medication change"), None)
            med_name_detected = se_entry.get("source_id", "this medication") if se_entry else "the medication"
            effects_str = se_entry.get("content", "") if se_entry else "Common side effects should be checked with a pharmacist or package insert."
            patient_context = f"Prescribed in {med_add_entry.get('content', 'patient encounters')}; check the patient's clinical notes for any reported reaction." if med_add_entry else "Part of patient regimen; check the patient's clinical notes for any reported reaction."
            answer = (
                f"### Answer\n"
                f"**Common Known Side Effects for {med_name_detected}:**\n"
                f"- {effects_str}\n\n"
                f"**Patient-Specific Context:**\n"
                f"- {patient_context}\n"
                f"- Monitor for symptoms and contact the clinician if adverse effects arise."
            )
        elif is_lab_q and any(w in question.lower() for w in ("indicate", "mean", "interpretation")) and any(e.get("category") in ("laboratory result", "patient diagnoses overview") for e in evidence):
            lab_items = [item for item in evidence if item.get("category") in ("laboratory result", "patient diagnoses overview")][:5]
            ans_lines = [f"- **{item['date']} (Encounter #{item.get('visit_index', '?')})**: {item['content'][:250]}" for item in lab_items]
            answer = "### Answer\nThe documented laboratory findings are listed below; interpretation should be done by the treating clinician:\n" + "\n".join(ans_lines)
        elif rel and v_num is not None and any(w in q_tokens for w in ("medicine", "medicines", "medication", "medications", "added", "started", "new", "rx")):
            changes = [item for item in evidence if item.get("category") == "medication change"]
            if changes:
                ans_lines = [f"- **Encounter #{ch.get('visit_index', '?')} ({ch['date']})**: {ch['content']}" for ch in changes]
                answer = f"### Answer\nThe following medication changes were documented {rel} Visit {v_num}:\n" + "\n".join(ans_lines)
            else:
                answer = f"### Answer\nNo new medications or changes are documented {rel} Visit {v_num}."
        else:
            ans_lines = [f"- **{item['date']} ({item.get('category', 'Record')})**: {item['content'][:250]}" for item in evidence[:5]]
            answer = "### Answer\nThe relevant patient-record evidence is summarized below:\n" + "\n".join(ans_lines)

        ev_lines = []
        for item in evidence[:6]:
            enc_tag = f"Encounter #{item['visit_index']} · " if item.get("visit_index") else ""
            ev_lines.append(f"- {enc_tag}{item['date']} · Visit {item['visit_id']}")
        answer += "\n\n### Evidence\n" + "\n".join(ev_lines)

    return answer, (_packed_evidence if "_packed_evidence" in dir() else evidence)


def answer_patient_question(
    patient_data, patient_id, question, history=None,
    api_key=None, retrieval_index=None, source_hash=None, meta=None
):
    meta = meta if meta is not None else {}
    t0 = time.perf_counter()
    evidence = []
    try:
        answer, evidence = _answer_patient_question(
            patient_data, patient_id, question,
            history=history, api_key=api_key,
            retrieval_index=retrieval_index, source_hash=source_hash,
            meta=meta,
        )
        return answer, evidence
    finally:
        try:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            log_question = str(question)[:120] if CHAT_LOG_QUESTIONS else hashlib.blake2b(str(question).encode("utf-8")).hexdigest()
            retrieved_map = dict(Counter(item.get("category", "") for item in (evidence or [])))
            sources = [f"{item.get('category')}|{item.get('visit_id')}|{item.get('source_id')}" for item in (evidence or [])[:8]]
            log_entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "patient_id": str(patient_id),
                "question": log_question,
                "path": meta.get("path"),
                "rule": meta.get("rule"),
                "model": meta.get("model"),
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "total_tokens": meta.get("total_tokens", 0),
                "retrieved": retrieved_map,
                "sources": sources,
                "ms": ms,
                "error": meta.get("error"),
            }
            audit_logger.info(json.dumps(log_entry, ensure_ascii=False))
        except Exception:
            pass
