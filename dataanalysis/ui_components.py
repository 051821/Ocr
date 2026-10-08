"""Doctor-friendly visual presentation components for Clinical Timeline and AI Review.

Provides high-readability cards, abnormal lab callouts, clean document summaries,
and executive longitudinal review layouts tailored for clinical review.
"""
import html
import re
import streamlit as st


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
def parse_ai_review(markdown: str) -> dict:
    """Parse generated AI markdown into structured sections for visual rendering."""
    sections = {
        "overall_pattern": "",
        "key_insights": [],
        "review_points": [],
        "considerations": [],
        "evidence": [],
        "declaration": "",
    }
    if not markdown:
        return sections

    current_section = None
    lines = markdown.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("## "):
            sec_name = stripped[3:].strip().lower()
            if "overall pattern" in sec_name:
                current_section = "overall_pattern"
            elif "key insights" in sec_name:
                current_section = "key_insights"
            elif "review points" in sec_name:
                current_section = "review_points"
            elif "general patient considerations" in sec_name or "considerations" in sec_name:
                current_section = "considerations"
            elif "evidence" in sec_name:
                current_section = "evidence"
            else:
                current_section = None
            continue
        elif stripped.startswith("# "):
            continue

        if "Retrospective record-based observations only" in stripped:
            sections["declaration"] = stripped
            continue

        if current_section == "overall_pattern":
            sections["overall_pattern"] = (sections["overall_pattern"] + " " + stripped).strip()
        elif current_section in ("key_insights", "review_points", "considerations", "evidence"):
            if stripped.startswith(("- ", "* ", "• ")):
                bullet_text = re.sub(r"^[-*•]\s*", "", stripped).strip()
                sections[current_section].append(bullet_text)
            elif not sections[current_section] and stripped:
                sections[current_section].append(stripped)

    return sections


def render_doctor_friendly_ai_review(raw_markdown: str):
    """Render the AI Clinical Review in an executive, doctor-friendly layout."""
    data = parse_ai_review(raw_markdown)
    if not data["overall_pattern"] and not data["key_insights"]:
        # Fallback to standard markdown if parsing yielded nothing
        st.markdown(raw_markdown)
        return

    # 1. Executive Longitudinal Trajectory (Hero Card)
    if data["overall_pattern"]:
        st.markdown(f"""
        <div class="ai-hero-card">
            <div class="ai-card-title">
                <span style="font-size: 1.15rem;">🩺</span> Longitudinal Clinical Trajectory
            </div>
            <div class="ai-hero-text">
                {html.escape(data["overall_pattern"])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Key Longitudinal Insights
    if data["key_insights"]:
        st.markdown('<div class="ai-section-hdr">🔍 Key Longitudinal Insights</div>', unsafe_allow_html=True)
        for insight in data["key_insights"]:
            title_match = re.match(r"^\*\*(.*?)\*\*[:\-]?\s*(.*)$", insight)
            if title_match:
                lead_title = title_match.group(1).strip()
                body_text = title_match.group(2).strip()
            else:
                lead_title = ""
                body_text = insight

            # Auto-tag icons based on clinical themes
            low = insight.lower()
            if any(k in low for k in ("hba1c", "glucose", "diabetes", "glycemic", "triglyceride", "lipid", "cholesterol", "metabolic")):
                icon, tag, border_color = "🩸", "Cardiometabolic & Glycemic Finding", "#EF4444"
            elif any(k in low for k in ("pcv", "hemoglobin", "rbc", "hematol", "blood")):
                icon, tag, border_color = "🧪", "Hematological Indicator", "#8B5CF6"
            elif any(k in low for k in ("medication", "prescription", "discrepan", "cetirizine", "omnacortil", "reconcil")):
                icon, tag, border_color = "💊", "Medication Reconciliation", "#F59E0B"
            elif any(k in low for k in ("follow-up", "missed", "adherence", "appoint")):
                icon, tag, border_color = "📅", "Care Continuity", "#0EA5E9"
            elif any(k in low for k in ("dermatol", "rash", "eczema", "zoster", "itch")):
                icon, tag, border_color = "🩹", "Dermatological Finding", "#10B981"
            else:
                icon, tag, border_color = "📌", "Longitudinal Pattern", "#3B82F6"

            header_html = (
                f'<div style="font-weight:700; font-size:0.95rem; margin-bottom:4px; color:{border_color};">'
                f'{html.escape(lead_title)}</div>'
            ) if lead_title else ""

            st.markdown(f"""
            <div class="ai-insight-card" style="border-left-color: {border_color};">
                <div class="ai-tag" style="color: {border_color};">
                    <span>{icon}</span> {tag}
                </div>
                {header_html}
                <div class="ai-insight-text">{_md_to_html(body_text)}</div>
            </div>
            """, unsafe_allow_html=True)

    # 3. Clinical Review & Verification Points
    if data["review_points"]:
        st.markdown('<div class="ai-section-hdr">⚠️ Items Warranting Clinical Verification</div>', unsafe_allow_html=True)
        items_html = "".join(f"<li>{_md_to_html(pt)}</li>" for pt in data["review_points"])
        st.markdown(f"""
        <div class="ai-review-box">
            <div class="ai-review-title">Clinical Audit Checkpoints</div>
            <ul class="ai-review-list">{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)


    # 4. General Patient Considerations (Patient Discussion Guide)
    if data["considerations"]:
        st.markdown('<div class="ai-section-hdr">💬 Patient Education & Discussion Considerations</div>', unsafe_allow_html=True)
        st.caption("Condition-relevant points for clinician-patient dialogue (practical diet, lifestyle, and monitoring guidance).")
        col1, col2 = st.columns(2)
        for i, item in enumerate(data["considerations"]):
            target_col = col1 if (i % 2 == 0) else col2
            title_match = re.match(r"^\*\*(.*?)\*\*[:\-]?\s*(.*)$", item)
            if title_match:
                lead_title = title_match.group(1).strip()
                body_text = title_match.group(2).strip()
            else:
                lead_title = ""
                body_text = item

            low = item.lower()
            if any(k in low for k in ("diet", "food", "meal", "nutrition", "carbohydrate", "sugar", "sweetened", "glycemic")):
                c_icon = "🥗"
                default_title = "Dietary Carbohydrate & Glycemic Nutrition"
            elif any(k in low for k in ("fat", "lipid", "triglyceride", "cholesterol", "oil", "fried", "heart")):
                c_icon = "🥑"
                default_title = "Lipid & Heart-Healthy Nutrition"
            elif any(k in low for k in ("sodium", "salt", "hypertension", "blood pressure")):
                c_icon = "🧂"
                default_title = "Dietary Sodium & Blood Pressure"
            elif any(k in low for k in ("activity", "exercise", "walk")):
                c_icon = "🏃"
                default_title = "Daily Physical Activity"
            elif any(k in low for k in ("skin", "eczema", "emollient", "moisturizer", "barrier")):
                c_icon = "🧴"
                default_title = "Skin Barrier & Emollient Care"
            elif any(k in low for k in ("symptom", "thirst", "vision", "fever", "seek", "warning", "attention", "hypoglycemia", "hyperglycemia")):
                c_icon = "🩺"
                default_title = "Symptom Awareness & Warning Signs"
            elif any(k in low for k in ("follow-up", "monitoring", "appoint", "check", "adherence")):
                c_icon = "📅"
                default_title = "Care Continuity & Follow-Up"
            else:
                c_icon = "💡"
                default_title = "Condition Self-Care Consideration"

            card_title = lead_title if lead_title else default_title

            with target_col:
                st.markdown(f"""
                <div class="ai-consideration-card">
                    <div style="width: 100%;">
                        <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                            <span style="font-size: 1.15rem;">{c_icon}</span>
                            <span style="font-weight: 700; font-size: 0.86rem; color: #10B981;">{html.escape(card_title)}</span>
                        </div>
                        <div style="font-size: 0.88rem; line-height: 1.48; color: inherit;">{html.escape(body_text)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 5. Evidence References
    if data["evidence"]:
        st.markdown('<div class="ai-section-hdr">📎 Source Traceability References</div>', unsafe_allow_html=True)
        chips_html = "".join(f'<span class="evidence-chip">📌 {html.escape(e)}</span>' for e in data["evidence"])
        st.markdown(f'<div style="margin-bottom: 1rem;">{chips_html}</div>', unsafe_allow_html=True)

    # Closing Notice
    dec_text = data["declaration"] or "Retrospective record-based observations only; clinician review is required for interpretation and decision-making."
    st.markdown(f'<div class="ai-declaration-footer">⚖️ {html.escape(dec_text)}</div>', unsafe_allow_html=True)

    # Clean Raw Markdown Expander for EHR note export
    with st.expander("📋 View / Copy Raw Markdown for EHR Export", expanded=False):
        st.code(raw_markdown, language="markdown")


# ── Clinical Timeline Visual Encounter Renderer ─────────────────────────────
def render_clinical_encounter(idx: int, visit: dict, med_shifts_for_visit: list):
    """Render a single clinical encounter compactly with high visual doctor-friendliness."""
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

        has_findings = bool(impression or symptoms or (cc and cc != visit.get("chief_complaint")))
        if has_findings:
            active_docs.append({
                "label": label,
                "short_id": short_id,
                "impression": impression,
                "symptoms": symptoms,
                "cc": cc,
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
    header_title = f"🗓️ Encounter #{idx} — {v_date} (Visit ID: {v_id[:8]}...{v_id[-4:]})"

    with st.expander(header_title, expanded=True):
        # 1. Chief Complaint & Diagnoses
        cc = visit.get("db_chief_complaint") or visit.get("chief_complaint") or "Routine encounter / Not documented"
        st.markdown(f"""
        <div style="margin-bottom: 0.6rem;">
            <span style="font-weight: 700; color: #0284C7; font-size: 0.90rem; text-transform: uppercase; letter-spacing: 0.05em;">Chief Complaint:</span>
            <span style="font-weight: 600; font-size: 0.95rem; margin-left: 6px;">{html.escape(cc)}</span>
        </div>
        """, unsafe_allow_html=True)

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
                st.markdown("<p style='font-size:0.85rem; color:#64748B; font-style:italic;'>No documents attached to this encounter (Direct database entry).</p>", unsafe_allow_html=True)
            elif not active_docs:
                st.markdown(f"<div class='doc-clean-card'><span style='color:#64748B; font-size:0.86rem;'>📄 {len(docs)} verified clinical documents processed (no narrative notes).</span></div>", unsafe_allow_html=True)
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
                        impr_block = (
                            '<div style="margin-top:7px;">'
                            '<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
                            'letter-spacing:0.05em; color:#94A3B8; margin-bottom:3px;">Clinical Impression</div>'
                            f'<div style="font-size:0.875rem; line-height:1.55; color:inherit;">'
                            f'{html.escape(ad["impression"])}</div>'
                            '</div>'
                        )

                    symptom_block = (
                        f'<div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:3px;">{symptom_chips}</div>'
                        if symptom_chips else ""
                    )

                    st.markdown(f"""
                    <div class="doc-clean-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="display:flex; align-items:center; gap:6px;">
                                <span style="font-size:1.1rem;">{doc_icon}</span>
                                <span class="doc-label-badge">{html.escape(ad['label'])}</span>
                            </div>
                            <span class="doc-id-badge">ID: {html.escape(ad['short_id'])}</span>
                        </div>
                        {impr_block}
                        {symptom_block}
                    </div>
                    """, unsafe_allow_html=True)

                if routine_doc_count > 0:
                    st.markdown(
                        f"<div style='font-size:0.78rem; color:#64748B; margin-top:4px;'>"
                        f"+ {routine_doc_count} routine documents (no narrative notes).</div>",
                        unsafe_allow_html=True,
                    )

        with col_labs:
            st.markdown('<div class="encounter-subheading">🧪 Laboratory Investigations</div>', unsafe_allow_html=True)
            if not raw_labs:
                st.markdown("<p style='font-size:0.85rem; color:#64748B; font-style:italic;'>No laboratory investigations recorded for this visit.</p>", unsafe_allow_html=True)
            else:
                if abnormal_labs:
                    st.markdown(f"<div style='font-size:0.80rem; font-weight:700; color:#EF4444; margin-bottom:4px;'>⚠️ ABNORMAL FINDINGS ({len(abnormal_labs)}):</div>", unsafe_allow_html=True)
                    for ab in abnormal_labs:
                        ref_str = f" <span style='font-size:0.75rem; color:#94A3B8;'>(Ref: {html.escape(ab['ref'])})</span>" if ab['ref'] else ""
                        st.markdown(f"""
                        <div class="lab-abnormal-card">
                            <span style="font-weight:650; color:#F87171;">{html.escape(ab['name'])}:</span>
                            <span style="font-weight:700; margin: 0 4px;">{html.escape(ab['val'])}</span>
                            <span class="lab-status-tag">{html.escape(ab['status'])}</span>
                            {ref_str}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size:0.82rem; color:#10B981; font-weight:600; margin-bottom:4px;'>✅ All reported laboratory parameters within normal range.</div>", unsafe_allow_html=True)

                if normal_labs:
                    with st.expander(f"✅ View {len(normal_labs)} Normal Parameters", expanded=False):
                        st.caption(", ".join(normal_labs))

        st.markdown("<hr style='margin: 0.6rem 0; opacity: 0.15;' />", unsafe_allow_html=True)

        # 3. Bottom Row: Medication Shifts & Follow-up Action
        col_meds, col_fu = st.columns([1.2, 0.8])
        with col_meds:
            st.markdown('<div class="encounter-subheading">💊 Medication Shifts in this Encounter</div>', unsafe_allow_html=True)
            if med_shifts_for_visit:
                for ch in med_shifts_for_visit:
                    ch_type = ch.get("type", "")
                    ch_text = ch.get("text", "")
                    if ch_type == "started":
                        badge = '<span class="diff-badge-added">Started</span>'
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
                    st.markdown(f"<div style='font-size:0.86rem; color:#64748B;'>Active Prescriptions: {html.escape(med_names)}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size:0.85rem; color:#64748B; font-style:italic;'>No medication adjustments or prescriptions documented.</div>", unsafe_allow_html=True)

        with col_fu:
            st.markdown('<div class="encounter-subheading">📅 Encounter Disposition / Follow-up</div>', unsafe_allow_html=True)
            if fu_badges:
                st.markdown(" ".join(fu_badges), unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.85rem; color:#64748B; font-style:italic;'>No subsequent follow-up scheduled.</div>", unsafe_allow_html=True)
