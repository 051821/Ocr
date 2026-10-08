"""Patient-scoped hybrid retrieval and Q&A over normalized clinical records and document chunks."""
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from config import OPENROUTER_API_KEY
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

from medication_journey import compute_medication_journey
from neo4j_client import is_neo4j_configured, query_patient_graph

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
CHAT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "inclusionai/ling-3.0-flash-sante:free",
]


SYSTEM_PROMPT = """You are a patient-record question answering assistant for a clinician.

Answer the user's question using ONLY evidence retrieved for the selected patient.

Do not invent facts.
Do not use information from other patients.
Do not diagnose conditions that are not documented.
Do not infer a diagnosis solely from an isolated laboratory value.
Do not provide medication recommendations or treatment plans.
Do not provide generic lifestyle advice unless the user explicitly asks for general patient education.

When evidence is insufficient, say so clearly.
Distinguish documented facts from interpretation.
Answer directly and concisely.
Provide source evidence including date, visit ID, document ID, or other available source identifiers."""


def _date(visit):
    return (visit.get("visit_date") or "Date not recorded")[:10]


def chunk_patient_documents(patient_data, chunk_size=3):
    """Chunk document text into indexed chunks with full metadata for retrieval."""
    chunks = []
    pid = str(patient_data.get("patient_id", ""))
    for v_idx, visit in enumerate(patient_data.get("visits", []), 1):
        v_id = str(visit.get("visit_id") or f"V{v_idx}")
        v_date = _date(visit)
        for doc in visit.get("documents", []) or []:
            doc_id = str(doc.get("doc_id", "doc_unknown"))
            doc_label = str(doc.get("document_label", "Clinical Document"))
            lines = doc.get("extracted_text", []) or []
            if not lines and doc.get("clinical_summary"):
                cs = doc["clinical_summary"]
                for k in ("clinical_impression", "clinical_notes", "chief_complaint", "symptoms"):
                    val = cs.get(k)
                    if val:
                        lines.append(f"{k.replace('_', ' ').title()}: {val}")

            for i in range(0, len(lines), chunk_size):
                segment = lines[i:i + chunk_size]
                text = " ".join(str(s).strip() for s in segment if str(s).strip())
                if text:
                    chunks.append({
                        "patient_id": pid,
                        "document_id": doc_id,
                        "visit_id": v_id,
                        "date": v_date,
                        "document_type": doc_label,
                        "chunk_id": f"{doc_id}_c{i // chunk_size + 1}",
                        "text": text,
                    })
    return chunks


def retrieve_patient_evidence(patient_data, patient_id, question, limit=20):
    """Retrieve structured records, medication changes, and document chunks strictly scoped to patient_id."""
    if not patient_data or str(patient_data.get("patient_id", "")).strip() != str(patient_id).strip():
        # STRICT ISOLATION: Never retrieve data if patient_id doesn't match
        return []

    visits = patient_data.get("visits", [])

    q = question.lower()
    words = set(re.findall(r"[a-z0-9]+", q)) - {
        "what", "when", "did", "the", "this", "patient", "patients", "about", "between",
        "and", "was", "are", "is", "across", "over", "time", "latest", "previous", "visit",
        "visits", "documented", "condition", "conditions", "uploaded", "documents", "document",
        "say", "said", "for", "from", "with", "show", "tell", "me"
    }

    chosen = []

    def add(date, visit_id, category, source_id, content):
        if content:
            chosen.append({
                "date": date,
                "visit_id": visit_id,
                "category": category,
                "source_id": source_id,
                "content": str(content).strip()
            })

    wants_labs = any(w in q for w in ("lab", "test", "a1c", "hba1c", "result", "investigation", "blood", "cholesterol", "triglyceride", "calcium", "hemoglobin"))
    wants_medications = any(w in q for w in ("medication", "medicine", "drug", "prescription", "dose", "dosage", "started", "taking"))
    wants_diagnosis = any(w in q for w in ("diagnos", "condition", "disease", "diabetes", "hypertension", "illness"))
    abnormal_only = "abnormal" in q or "high" in q or "low" in q

    # 1. Neo4j Graph Retrieval if available
    if is_neo4j_configured():
        try:
            graph_records = query_patient_graph(str(patient_id), query_type="all", limit=100)
            for rec in graph_records:
                category = rec.get("category", "Graph record")
                category_lower = category.lower()
                if wants_labs and category != "Lab":
                    continue
                if wants_medications and category not in ("Medication", "Visit", "DocumentChunk"):
                    continue
                if wants_diagnosis and category not in ("Diagnosis", "Visit", "DocumentChunk"):
                    continue
                if abnormal_only and category == "Lab" and not any(
                    flag in rec.get("content", "").lower()
                    for flag in ("abnormal", "high", "low", "elevated", "decreased")
                ):
                    continue
                add(rec.get("date", "Date not recorded"), rec.get("visit_id", "Unknown"),
                    f"Neo4j {category_lower}", rec.get("source_id", category), rec.get("content", ""))
        except Exception as exc:
            logger.warning("Neo4j retrieval error, falling back to normalized record: %s", exc)

    # 2. Medication Changes & Regimen questions
    if wants_medications or any(w in q for w in ("changed", "change")):
        med_shifts = compute_medication_journey(patient_data)
        for enc in med_shifts:
            v_date = enc["visit_date"]
            v_id = enc["visit_id"]
            for ch in enc.get("changes", []):
                add(v_date, v_id, "medication change", v_id, ch["text"])
        # Also include any prescription details
        for visit in visits:
            rx_list = visit.get("db_medications") or visit.get("medications") or visit.get("rx_items", [])
            if rx_list:
                rx_str = "; ".join(
                    f"{m.get('name') or m.get('medication_name', '')} {m.get('dosage','')} {m.get('frequency','')}".strip()
                    for m in rx_list if m.get("name") or m.get("medication_name")
                )
                add(_date(visit), visit.get("visit_id"), "prescriptions", visit.get("visit_id"), f"Active Rx: {rx_str}")

    # Vitals, symptoms, and follow-up details may be stored on Visit nodes or
    # in the normalized encounter payload rather than dedicated graph nodes.
    if any(w in q for w in ("vital", "blood pressure", "pulse", "heart rate", "temperature", "oxygen", "weight", "height", "symptom", "follow up", "follow-up", "appointment")):
        for visit in visits:
            v_id = visit.get("visit_id", "Unknown")
            vital_data = visit.get("vitals") or visit.get("db_vitals")
            if vital_data:
                add(_date(visit), v_id, "visit vitals", v_id, f"Vitals: {vital_data}")
            symptoms = visit.get("symptoms") or visit.get("chief_complaint")
            if symptoms:
                add(_date(visit), v_id, "visit symptoms", v_id, f"Symptoms/complaint: {symptoms}")
            followups = visit.get("followups") or []
            if followups:
                add(_date(visit), v_id, "follow-up", v_id, "; ".join(str(x) for x in followups))

    # 3. Diagnosis & Condition Timeline questions (e.g. "When did diabetes first appear?")
    if any(w in q for w in ("diagnos", "condition", "disease", "diabetes", "appear", "first", "eczema", "hypertension", "illness")):
        for visit in visits:
            v_date = _date(visit)
            v_id = visit.get("visit_id")
            dxs = []
            for k in ("db_confirmed_dx", "confirmed_diagnosis", "db_provisional_dx", "provisional_diagnosis"):
                val = visit.get(k)
                if val and str(val).strip() and str(val).lower() not in ("—", "-", "none"):
                    dxs.append(f"{k.replace('db_', '').replace('_', ' ').title()}: {val}")
            for doc in visit.get("documents", []) or []:
                cs = doc.get("clinical_summary", {}) or {}
                for k in ("clinical_impression", "confirmed_diagnosis", "provisional_diagnosis"):
                    val = cs.get(k)
                    if val and str(val).strip():
                        dxs.append(f"Doc {doc.get('doc_id')} {k.replace('_', ' ').title()}: {val}")
            if dxs:
                add(v_date, v_id, "diagnoses", v_id, "; ".join(dict.fromkeys(dxs)))

    # 4. Latest Document question (e.g. "What did the latest document say?")
    if "latest document" in q or "most recent document" in q or ("document" in q and "latest" in q):
        found_doc = False
        for visit in reversed(visits):
            docs = visit.get("documents", [])
            if docs:
                latest_doc = docs[-1]
                doc_id = latest_doc.get("doc_id", "doc_latest")
                label = latest_doc.get("document_label", "Document")
                cs = latest_doc.get("clinical_summary", {}) or {}
                parts = []
                if cs.get("clinical_impression"):
                    parts.append(f"Impression: {cs['clinical_impression']}")
                if cs.get("chief_complaint"):
                    parts.append(f"Complaint: {cs['chief_complaint']}")
                if cs.get("symptoms"):
                    parts.append(f"Symptoms: {', '.join(cs['symptoms'])}")
                raw_lines = latest_doc.get("extracted_text", [])[:5]
                if raw_lines:
                    parts.append(f"Text excerpt: {'; '.join(str(x) for x in raw_lines)}")
                add(_date(visit), visit.get("visit_id"), "latest document", doc_id, f"[{label} (ID: {doc_id})]: " + " | ".join(parts))
                found_doc = True
                break

    # 5. Visit Comparison questions (e.g. "What changed between the last two visits?")
    if ("between" in q or "compare" in q) and len(visits) >= 2:
        for visit in visits[-2:]:
            v_date = _date(visit)
            v_id = visit.get("visit_id")
            summary = [
                f"Complaint: {visit.get('db_chief_complaint') or visit.get('chief_complaint') or 'None'}",
                f"Diagnosis: {visit.get('db_confirmed_dx') or visit.get('confirmed_diagnosis') or visit.get('db_provisional_dx') or 'None'}",
            ]
            rx = [m.get("name") for m in visit.get("db_medications", []) if m.get("name")]
            if rx:
                summary.append(f"Meds: {', '.join(rx)}")
            labs = [f"{l.get('test_name')}: {l.get('value')}" for l in visit.get("lab_results", [])]
            if labs:
                summary.append(f"Labs: {', '.join(labs)}")
            add(v_date, v_id, "visit comparison", v_id, "; ".join(summary))

    # 6. Evidence for Latest Documented Diagnosis
    if "evidence" in q and ("diagnosis" in q or "diagnos" in q):
        for visit in reversed(visits):
            dx = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis") or visit.get("db_provisional_dx")
            if dx:
                v_date = _date(visit)
                v_id = visit.get("visit_id")
                evidence_items = [f"Documented Diagnosis: {dx}"]
                labs = [f"{l.get('test_name')}: {l.get('value')} ({l.get('status','')})" for l in visit.get("lab_results", [])]
                if labs:
                    evidence_items.append("Labs: " + "; ".join(labs))
                for doc in visit.get("documents", []):
                    cs = doc.get("clinical_summary", {}) or {}
                    if cs.get("clinical_impression"):
                        evidence_items.append(f"Doc {doc.get('doc_id')} Impression: {cs['clinical_impression']}")
                add(v_date, v_id, "diagnostic evidence", v_id, "; ".join(evidence_items))
                break

    # 7. Labs questions
    if wants_labs and not any(item["category"] == "Neo4j lab" for item in chosen):
        for visit in visits:
            labs = visit.get("lab_results", [])
            if labs:
                relevant_labs = [l for l in labs if not abnormal_only or any(
                    flag in str(l.get("status") or "").lower()
                    for flag in ("abnormal", "high", "low", "elevated", "decreased")
                )]
                if relevant_labs:
                    add(_date(visit), visit.get("visit_id"), "labs", visit.get("visit_id"),
                        "; ".join(f"{l.get('test_name')}: {l.get('value')} (Ref: {l.get('reference', 'N/A')}) [{l.get('status', 'Evaluated')}]" for l in relevant_labs))

    # 8. Document Chunk Semantic Retrieval for Narrative Questions
    doc_chunks = chunk_patient_documents(patient_data)
    if doc_chunks and (words or "document" in q or "uploaded" in q or "note" in q):
        scored_chunks = []
        for ch in doc_chunks:
            chunk_words = set(re.findall(r"[a-z0-9]+", ch["text"].lower()))
            score = len(words & chunk_words) if words else 1
            if score > 0 or "document" in q:
                scored_chunks.append((score, ch))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        for _, ch in scored_chunks[:4]:
            add(ch["date"], ch["visit_id"], "document chunk", ch["chunk_id"], f"[{ch['document_type']} (Chunk {ch['chunk_id']})]: {ch['text']}")

    # Fallback to general visit complaints if nothing chosen yet
    if not chosen:
        for visit in visits:
            cc = visit.get("db_chief_complaint") or visit.get("chief_complaint")
            dx = visit.get("db_confirmed_dx") or visit.get("confirmed_diagnosis")
            if cc or dx:
                add(_date(visit), visit.get("visit_id"), "encounter summary", visit.get("visit_id"),
                    f"Chief complaint: {cc or 'N/A'}; Diagnosis: {dx or 'N/A'}")

    # Deduplicate exact source_id + content
    seen = set()
    dedup = []
    for item in chosen:
        key = (item["source_id"], item["category"], item["content"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)

    # Put the most question-relevant graph and local records first before the
    # context size cap, instead of allowing arbitrary Neo4j row order to win.
    query_terms = set(re.findall(r"[a-z0-9]+", q))
    def relevance(item):
        content_terms = set(re.findall(r"[a-z0-9]+", (item["category"] + " " + item["content"]).lower()))
        score = len(query_terms & content_terms)
        if wants_labs and "lab" in item["category"].lower():
            score += 20
        if wants_medications and any(k in item["category"].lower() for k in ("medication", "prescription")):
            score += 20
        if wants_diagnosis and "diagnos" in item["category"].lower():
            score += 20
        if abnormal_only and "abnormal" in item["content"].lower():
            score += 5
        return score
    dedup.sort(key=relevance, reverse=True)
    return dedup[:limit]


def answer_patient_question(patient_data, patient_id, question, history=None, api_key=None):
    """Answer a clinician's question using ONLY retrieved evidence for the selected patient.

    Uses exactly one LLM call.
    """
    evidence = retrieve_patient_evidence(patient_data, patient_id, question)
    logger.info(
        "patient_chat patient_id=%s question=%s retrieved=%d",
        patient_id, question[:150], len(evidence)
    )

    if not evidence:
        return (
            "### Answer\nThe available record for this patient does not contain evidence to answer this question.\n\n"
            "### Evidence\n- No relevant evidence found in the selected patient's record.\n\n"
            "### Limitations\nInsufficient documentation available for this specific inquiry.",
            []
        )

    # Format evidence citations
    citations = []
    for e in evidence[:5]:
        src_label = f"Document {e['source_id']}" if "chunk" in e["category"] or "document" in e["category"] else f"Visit {e['visit_id']}"
        citations.append(f"- {e['date']} / {src_label} — {e['content'][:300]}")

    if not api_key:
        return (
            "### Answer\nThe relevant evidence was retrieved from the patient record, but the language model is not configured (missing API key).\n\n"
            "### Evidence\n" + "\n".join(citations) + "\n\n"
            "### Limitations\nLanguage model execution skipped due to missing API key.",
            evidence
        )

    context_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Include small bounded conversation history
    for item in (history or [])[-4:]:
        if item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": str(item.get("content", ""))[:1500]})

    user_prompt = (
        f"Selected Patient ID: {patient_id}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Evidence (Strictly Patient {patient_id}):\n{context_json}\n\n"
        "Instructions: Return your response with the following markdown headers:\n"
        "### Answer\n(Direct, concise answer based ONLY on the evidence above)\n\n"
        "### Evidence\n(Concise source bullets with date and encounter/document ID)\n\n"
        "### Limitations\n(Only if evidence is incomplete or conflicting; otherwise omit this section)"
    )
    messages.append({"role": "user", "content": user_prompt})

    # Exactly ONE LLM call
    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.15, "max_tokens": 1000},
        timeout=60
    )
    response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"].strip()

    # Ensure response has ### Answer and ### Evidence structure
    if "### Answer" not in raw_content:
        raw_content = f"### Answer\n{raw_content}"

    if "### Evidence" not in raw_content:
        raw_content = raw_content + "\n\n### Evidence\n" + "\n".join(citations)

    return raw_content, evidence
