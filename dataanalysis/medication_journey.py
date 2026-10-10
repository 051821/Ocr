"""Medication Journey and Longitudinal Change Tracking.

Deterministic analysis of how medications change across patient encounters.
Detects:
- started / added
- continued
- dose/frequency changed
- discontinued (only when explicitly documented)
- newly documented (from clinical documents)
- structured prescription vs document mismatches
"""
import re

try:
    from analysis import normalize_med_name
    from analysis import _is_lab_term
except ImportError:
    normalize_med_name = lambda value: str(value or "").strip()
    _is_lab_term = lambda value: False


def _normalize_name(name):
    if not name:
        return ""
    clean = re.sub(r"(?i)^(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp(?:rup)?)\.?\s+", "", str(name).strip())
    clean = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", "", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip(" .,:;-_")
    return normalize_med_name(clean).casefold().strip()


def _display_name(name):
    clean = re.sub(r"(?i)^(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp(?:rup)?)\.?\s+", "", str(name or "").strip())
    clean = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", "", clean, flags=re.I)
    return normalize_med_name(re.sub(r"\s+", " ", clean).strip(" .,:;-_"))


def _extract_explicit_discontinuations(visit, known_prescriptions):
    """Detect if clinical instructions or notes explicitly state a drug was stopped.

    Rule: Never infer discontinuation merely because a drug is absent.
    """
    stopped = []
    text_corpus = []

    # Check database prescription instructions
    for inst in visit.get("db_instructions", []) or []:
        text_corpus.append(str(inst))

    # Check document clinical notes/summaries
    for doc in visit.get("documents", []) or []:
        cs = doc.get("clinical_summary", {}) or {}
        for k in ("clinical_notes", "clinical_impression", "instructions"):
            val = cs.get(k)
            if val:
                text_corpus.append(str(val))

    known_names = {key: value.get("name", key) for key, value in known_prescriptions.items()}
    known_pattern = "|".join(re.escape(name) for name in sorted(known_names.values(), key=len, reverse=True))
    if not known_pattern:
        return []
    stop_pattern = re.compile(
        rf"\b(?:discontinue|discontinued|stop|stopped|cease|ceased|withhold|withdrawn|hold)\s+(?:the\s+)?({known_pattern})\b",
        re.IGNORECASE,
    )
    for text in text_corpus:
        matches = stop_pattern.findall(text)
        for match in matches:
            m_clean = _normalize_name(match)
            if m_clean in known_names:
                stopped.append(match.strip())

    return list(dict.fromkeys(stopped))


def compute_medication_journey(patient_data):
    """Compute chronological medication changes across visits.

    Returns a list of encounter change dicts:
    [
        {
            "visit_id": str,
            "visit_date": str,
            "visit_index": int,
            "changes": [
                {"type": "started"|"continued"|"dose_changed"|"discontinued"|"newly_documented"|"mismatch",
                 "medication": str, "details": str}
            ],
            "all_active_meds": [...]
        }
    ]
    """
    if not patient_data:
        return []
    cached = patient_data.get("_medication_journey_cache")
    if cached is not None:
        return cached
    if not patient_data.get("visits"):
        return []

    visits = patient_data.get("visits", [])
    # Sort visits by visit_date if available
    sorted_visits = sorted(
        visits,
        key=lambda v: str(v.get("visit_date") or "")
    )

    history = []
    # Track known medications: med_name_lower -> {"dosage": ..., "frequency": ..., "last_seen_date": ...}
    known_prescriptions = {}
    known_doc_meds = set()

    for idx, v in enumerate(sorted_visits, 1):
        v_date = (v.get("visit_date") or f"Visit {idx}")[:10]
        v_id = v.get("visit_id", f"V{idx}")

        rx_list = v.get("db_medications", []) or []
        doc_meds_raw = []
        for doc in v.get("documents", []) or []:
            cs = doc.get("clinical_summary", {}) or {}
            for dm in cs.get("medications", []) or []:
                if isinstance(dm, dict) and dm.get("name"):
                    doc_meds_raw.append(dm)
                elif isinstance(dm, str):
                    doc_meds_raw.append({"name": dm})
            for dm_str in doc.get("medications_found", []) or []:
                doc_meds_raw.append({"name": dm_str})

        # Also check visit level document_medications if present
        for dm_str in v.get("document_medications", []) or []:
            doc_meds_raw.append({"name": str(dm_str)})

        # Deduplicate current visit prescriptions by lowercase name
        current_rx = {}
        for rx in rx_list:
            display_name = _display_name(rx.get("name"))
            med_key = _normalize_name(rx.get("name"))
            if med_key and not _is_lab_term(display_name):
                current_rx[med_key] = {
                    "name": display_name,
                    "dosage": str(rx.get("dosage") or "").strip(),
                    "frequency": str(rx.get("frequency") or "").strip(),
                    "duration": str(rx.get("duration") or "").strip(),
                }

        # Deduplicate current visit doc meds
        current_doc_meds = {}
        for dm in doc_meds_raw:
            display_name = _display_name(dm.get("name"))
            med_key = _normalize_name(dm.get("name"))
            if med_key and not _is_lab_term(display_name):
                current_doc_meds[med_key] = {
                    "name": display_name,
                    "strength": str(dm.get("strength") or dm.get("dosage") or "").strip(),
                    "frequency": str(dm.get("frequency") or "").strip(),
                }

        encounter_changes = []

        # 1. Evaluate prescriptions against history
        for med_key, rx_info in current_rx.items():
            name = rx_info["name"]
            dosage = rx_info["dosage"]
            frequency = rx_info["frequency"]

            dose_str = f"{dosage} {frequency}".strip()
            spec = f"({dose_str})" if dose_str else ""

            if med_key not in known_prescriptions:
                # Started
                action_word = "Started" if idx == 1 else "Added"
                detail = f"{name} {spec}".strip()
                encounter_changes.append({
                    "type": "started",
                    "medication": name,
                    "text": f"{action_word}: {detail}"
                })
            elif known_prescriptions[med_key].get("discontinued"):
                encounter_changes.append({
                    "type": "restarted",
                    "medication": name,
                    "text": f"Restarted: {name} {spec}".strip(),
                })
            else:
                prev = known_prescriptions[med_key]
                prev_dosage = prev.get("dosage", "")
                prev_freq = prev.get("frequency", "")

                # Check if dose or frequency changed
                if (dosage and prev_dosage and dosage.lower() != prev_dosage.lower()) or \
                   (frequency and prev_freq and frequency.lower() != prev_freq.lower()):
                    prev_spec = f"{prev_dosage} {prev_freq}".strip()
                    encounter_changes.append({
                        "type": "dose_changed",
                        "medication": name,
                        "text": f"{name} dose/frequency changed (from {prev_spec or 'prior dose'} to {dose_str})"
                    })
                else:
                    encounter_changes.append({
                        "type": "continued",
                        "medication": name,
                        "text": f"{name} continued {spec}".strip()
                    })

            # Update known prescriptions
            known_prescriptions[med_key] = {
                "name": name,
                "dosage": dosage,
                "frequency": frequency,
                "last_seen_date": v_date,
                "discontinued": False,
            }

        # 2. Check for explicit discontinuations documented in record
        explicit_stops = _extract_explicit_discontinuations(v, known_prescriptions)
        for stop_name in explicit_stops:
            stop_key = _normalize_name(stop_name)
            if stop_key not in known_prescriptions:
                continue
            encounter_changes.append({
                "type": "discontinued",
                "medication": stop_name,
                "text": f"Discontinued: {stop_name} (explicitly noted in record)"
            })
            if stop_key in known_prescriptions:
                known_prescriptions[stop_key]["discontinued"] = True

        # 3. Newly documented meds from clinical documents
        for doc_key, doc_info in current_doc_meds.items():
            dname = doc_info["name"]
            if doc_key not in known_doc_meds and doc_key not in current_rx:
                d_spec = f"({doc_info['strength']})".strip() if doc_info.get("strength") else ""
                encounter_changes.append({
                    "type": "newly_documented",
                    "medication": dname,
                    "text": f"Newly documented in clinical document: {dname} {d_spec}".strip()
                })
            known_doc_meds.add(doc_key)

        # 4. Structured prescription vs document mismatch
        for doc_key, doc_info in current_doc_meds.items():
            dname = doc_info["name"]
            if doc_key not in current_rx:
                encounter_changes.append({
                    "type": "mismatch",
                    "medication": dname,
                    "text": f"Mismatch: {dname} noted in document but not in structured prescription"
                })

        all_active_meds = sorted(
            value["name"] for value in known_prescriptions.values()
            if not value.get("discontinued")
        )
        history.append({
            "visit_id": v_id,
            "visit_date": v_date,
            "visit_index": idx,
            "changes": encounter_changes,
            "active_rx_count": len(all_active_meds),
            "all_active_meds": all_active_meds,
        })

    return history


def format_medication_journey_text(journey_data):
    """Format the medication journey as clean text lines for display."""
    if not journey_data:
        return "No medication records found."

    lines = []
    for enc in journey_data:
        lines.append(f"### {enc['visit_date']} (Encounter {enc['visit_index']})")
        if not enc["changes"]:
            lines.append("No medication changes or prescriptions documented.")
        else:
            for ch in enc["changes"]:
                lines.append(f"- {ch['text']}")
        lines.append("")
    return "\n".join(lines)
