"""AI Analysis & Summary and Lab Results tabs, rendered exactly as the static Patient Insight Review page.

The data comes from insight_view.build_view (API: GET /patients/{id}/insight-view); the markup and behaviour are the
static page's own renderers (patient_dashboard.html), mounted in st.html: status tiles that highlight their points, one
Patient Brief merging the AI and rule-based briefs (each point marked AI + Rule / AI · checked / Rule-based), #n
citations that open the visit, Key-finding links to their concerns, concerns with full evidence, raw data and rule, the
visit grid, Clinical Course, and the lab table with its out-of-range filter. Styles: ui/insight_tabs.css (scoped .cdx).
"""
import base64
import html
import json
import os
import uuid

import streamlit as st

from area_tiles import AREA_CSS, AREA_JS
from medicine_grid import MED_GRID_CSS, MED_GRID_JS

_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_DIR, "ui", "insight_tabs.css"), encoding="utf-8") as _f:
    _CSS = _f.read()


def _json(value) -> str:
    """JSON safe inside a <script> element."""
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


def _script(code: str) -> str:
    """A <script> that survives st.html's DOMPurify pass.

    DOMPurify (SAFE_FOR_XML) deletes any element whose text holds '<' followed by a letter, '/' or '!', which every
    renderer here does (HTML template strings). The code is shipped base64-encoded (no '<') and decoded as UTF-8 in
    the browser."""
    b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return ("<script>(function(){var b=atob('" + b64 + "'),u=new Uint8Array(b.length);"
            "for(var i=0;i!==b.length;i++)u[i]=b.charCodeAt(i);"
            "(new Function(new TextDecoder('utf-8').decode(u)))();})();</script>")


# ── Shared helpers (verbatim from the static page) ───────────────────────────
_COMMON_JS = r"""
const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
const REF_YEAR = String(p.latest_visit || "").slice(0, 4);
const day = d => { const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(d || "")); return m ? `${+m[3]} ${MONTHS[+m[2] - 1]}${m[1] !== REF_YEAR ? " " + m[1] : ""}` : String(d ?? ""); };
const esc = s => String(s ?? "").replace(/\b\d{4}-\d{2}-\d{2}\b/g, day)
  .replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
// Highlighted concerns: the pipeline's order (snapshot_view.decorate), grouped by topic, most serious group first.
function concernList(p) {
  const order = p.snap_concerns || p.flags.map((f, i) => i).filter(i => p.flags[i].level !== "info");
  const group = {};
  (p.snap_concern_groups || []).forEach(g => g.concerns.forEach(i => group[i] = g.label));
  return order.map(i => ({ ...p.flags[i], _i: i, _group: group[i] || "" })).map((f, n) => ({ ...f, _n: n + 1 }));
}
const human = k => k.replace(/_/g, " ").replace(/^\w/, c => c.toUpperCase());
// Prescriber specialization tags (no names); same colour per specialization as medicine_grid.spec_colour.
const SPEC_COLOURS = ["#2563eb", "#d97706", "#7c3aed", "#0f766e", "#be185d", "#4d7c0f", "#b45309", "#0369a1"];
const specColour = s => (p.spec_colours || {})[s] || SPEC_COLOURS[[...String(s)].reduce((a, c) => a + c.codePointAt(0), 0) % SPEC_COLOURS.length];
const specTags = specs => (specs || []).filter(Boolean).map(s =>
  `<span class="rx-spec" style="--c:${specColour(s)}" title="Prescribed by: ${esc(s)}">${esc(s)}</span>`).join("");
function statusBadge(s) {
  s = String(s || "");
  if (/critical/i.test(s)) return `<span class="badge b-high">${esc(s)}</span>`;
  if (/unverified/i.test(s)) return `<span class="badge b-medium">${esc(s)}</span>`;
  if (/high|low|abnormal|reactive/i.test(s) && !/non/i.test(s)) return `<span class="badge b-medium">${esc(s)}</span>`;
  if (/normal/i.test(s)) return `<span class="badge b-ok">${esc(s)}</span>`;
  return `<span class="badge b-kind">${esc(s || "—")}</span>`;
}
// Evidence-pack values: tables for lists of objects, key/value grids for objects.
function renderValue(v) {
  if (v == null || v === "") return `<span class="muted">—</span>`;
  if (Array.isArray(v)) {
    if (!v.length) return `<span class="muted">none</span>`;
    if (v.every(x => x && typeof x === "object" && !Array.isArray(x))) {
      const keys = [...new Set(v.flatMap(Object.keys))].filter(k => !["key", "canonical", "direction", "test_as_written", "number"].includes(k));
      return `<div class="tbl-wrap"><table><tr>${keys.map(k => `<th>${esc(human(k))}</th>`).join("")}</tr>${v.map(row =>
        `<tr>${keys.map(k => `<td>${cell(row[k])}</td>`).join("")}</tr>`).join("")}</table></div>`;
    }
    return `<ul class="points">${v.map(x => `<li>${typeof x === "object" ? renderValue(x) : esc(x)}</li>`).join("")}</ul>`;
  }
  if (typeof v === "object") {
    return `<div class="kv">${Object.entries(v).filter(([k]) => !["key", "canonical", "direction", "test_as_written"].includes(k))
      .map(([k, x]) => `<div class="k">${esc(human(k))}</div><div>${typeof x === "object" ? renderValue(x) : esc(x)}</div>`).join("")}</div>`;
  }
  return esc(v);
}
function cell(x) {
  if (x == null || x === "") return "";
  if (Array.isArray(x)) return esc(x.join("; "));
  if (typeof x === "object") return renderValue(x);
  if (typeof x === "boolean") return x ? "yes" : "no";
  return esc(x);
}
function visitTable(rows, detailLabel = "Detail") {
  if (!rows || !rows.length) return `<div class="muted">No matching rows.</div>`;
  return `<div class="tbl-wrap"><table><tr><th>Visit</th><th>Date</th><th>Chief complaint</th><th>${esc(detailLabel)}</th></tr>${rows.map(r =>
    `<tr><td>#${r.enc}</td><td>${esc(r.date)}</td><td>${esc(r.complaint)}</td><td>${esc(r.detail)}</td></tr>`).join("")}</table></div>`;
}
function labTable(rows) {
  if (!rows || !rows.length) return `<div class="muted">No lab rows.</div>`;
  return `<div class="tbl-wrap"><table><tr><th>Test (as read)</th><th>Value</th><th>Reference</th><th>Status</th><th>Source</th><th>Note</th></tr>${rows.map(l =>
    `<tr${/high|low|abnormal|critical|unverified/i.test(l.status || "") ? ' class="flagged"' : ""}><td>${esc(l.test)}</td><td><b>${esc(l.value)}</b></td><td>${esc(l.reference)}</td><td>${statusBadge(l.status)}</td><td>${esc(l.source)}</td><td class="small muted">${esc(l.note)}</td></tr>`).join("")}</table></div>`;
}
function rawBlock(raw) {
  if (!raw || !Object.keys(raw).length) return `<div class="muted">The evidence line above lists the source values directly.</div>`;
  let h = "";
  if (raw.visits) h += visitTable(raw.visits);
  if (raw.labs) h += labTable(raw.labs);
  if (raw.reports) h += renderValue(raw.reports);
  if (raw.note) h += `<div class="small muted" style="margin-top:6px">${esc(raw.note)}</div>`;
  return h;
}
const ruleBox = (rule, open = false) => rule ? `<details class="rulebox"${open ? " open" : ""}><summary>Rule: ${esc(rule.name)}</summary><div class="dbody"><div class="rule-text">${esc(rule.text)}</div><div class="rule-src">${esc(rule.source)}</div></div></details>` : "";
const rawBox = (inner, label = "Raw data used") => `<details class="raw"><summary>${esc(label)}</summary><div class="dbody">${inner}</div></details>`;
const lvl = f => f.category === "data quality" ? "data_quality" : f.level;
"""

# ── Tab: AI Analysis & Summary (below the tiles and the Patient Brief heading) ──────────────────────────────
_AI_JS = r"""
function cited(text, p) {
  return esc(text).replace(/#(\d+)/g, (m, n) => {
    const v = p.visits.find(x => x.enc === +n);
    const tip = v ? `#${n} · ${v.date} · ${v.complaint || ""}${v.provisional || v.confirmed ? " · " + (v.provisional || v.confirmed) : ""}` : `#${n}`;
    return `<span class="cite" data-enc="${n}" title="${esc(tip)}">#${n}</span>`;
  });
}
// A short leading label ('Type 2 diabetes mellitus:', 'Hypertension:') in bold, as ui_components._point_html.
function pointHtml(text, p) {
  const m = /^([^:]{2,60}?):\s+([\s\S]+)$/.exec(text);
  if (m && m[1].trim().split(/\s+/).length <= 7 && !m[1].includes("**") && !/#\d/.test(m[1]))
    return `<b>${esc(m[1])}:</b> ${cited(m[2], p)}`;
  return cited(text, p);
}
function visitBox(v) {
  return `<div class="enc-pop"><b>#${v.enc} · ${esc(v.date)}</b> <button class="fbtn" data-enc="0" style="float:right">close</button>
    <div class="kv small" style="margin-top:6px"><div class="k">Complaint</div><div>${esc(v.complaint) || "—"}</div><div class="k">Diagnosis</div><div>${esc(v.confirmed || v.provisional) || "—"}</div>
    <div class="k">Vitals</div><div>${v.vitals.map(r => `<span class="chip">${esc(r.label)} ${esc(r.value)}</span>`).join("") || "—"}</div>
    ${(v.specialties || []).length ? `<div class="k">Prescribed by</div><div>${specTags(v.specialties)}</div>` : ""}
    <div class="k">Medicines</div><div>${esc(v.meds.map(m => `${m.name} ${m.dose || ""}${m.spec && (v.specialties || []).length > 1 ? " (Rx: " + m.spec + ")" : ""}`).join(", ")) || "—"}</div>
    ${v.followups.length ? `<div class="k">Follow-up</div><div>${v.followups.map(f => `${esc(f.date)} ${esc(f.status)}`).join("; ")}</div>` : ""}</div></div>`;
}
const THEMES = { "course": ["🗓️", "#104eb2", false], "conditions": ["🩺", "#029c89", false], "key findings": ["⚠️", "#d32f2f", true],
                 "treatment": ["💊", "#c77700", false], "gaps to check": ["🔎", "#7c3aed", false],
                 "gaps": ["🔎", "#7c3aed", false] };
function kfLinks(b, k, p) {
  const idx = (b.links || [])[k] || [];
  if (!idx.length) return "";
  const num = Object.fromEntries(concernList(p).map(f => [f._i, f._n]));
  const ns = idx.filter(i => num[i] !== undefined);
  return ns.length ? ` <span class="kf-links">→ concern ${ns.map(i => `<a class="kf-link" data-concern="${i}" title="Open concern ${num[i]} in Highlighted concerns">${num[i]}</a>`).join(" ")}</span>` : "";
}
const areasAttr = a => (a && a.length ? ` data-areas="${esc(a.join(" "))}"` : "");
const SRC = { both: ["src-both", "AI + Rule", "Made by the AI and by the record rules"],
              ai: ["src-ai", "AI · checked", "AI-written, checked against the evidence"],
              rule: ["src-rule", "Rule-based", "From the record rules only"] };
function themeEvidence(t) {
  const parts = Object.entries(t.evidence || {}).map(([k, v]) => `<span class="section-subhead">${esc(human(k))}</span>${renderValue(v)}`).join("");
  const flags = (t.flags || []).length ? `<span class="section-subhead">Related rule flags</span>${t.flags.map(f => `<span class="chip">${esc(f)}</span>`).join("")}` : "";
  return parts + flags;
}
function mergedCards(items, p) {
  return `<div class="brief-grid">${items.map((b, i) => {
    const [icon, accent, wide] = THEMES[b.theme.toLowerCase()] || ["📌", "#3B82F6", true];
    const why = `<details class="evid"><summary>Evidence and rule</summary><div class="dbody">${themeEvidence(b)}${b.rule ? ruleBox(b.rule) : ""}</div></details>`;
    return `<div class="brief-card${wide ? " brief-wide" : ""}" style="--accent:${accent};animation-delay:${i * 60}ms">
      <div class="brief-head"><span>${icon}</span>${esc(b.theme)}</div>
      <ul class="brief-points">${b.points.map((x, k) => { const [cls, label, tip] = SRC[(b.src || [])[k]] || SRC.ai;
        return `<li${areasAttr((b.areas || [])[k])}>${pointHtml(x, p)}${kfLinks(b, k, p)} <span class="src ${cls}" title="${tip}">${label}</span></li>`; }).join("")}</ul>${why}</div>`;
  }).join("")}</div>`;
}
function encPop(v) { return v ? visitBox(v) : ""; }


// ── Two tiers (summary_tiers.py): headline, detailed brief with links to concerns, quick view ──
function headlineHtml(p) {
  const h = (p.summary || {}).headline;
  if (!h) return "";
  const [cls, label, tip] = SRC[h.src] || SRC.rule;
  return `<div class="tier-headline">${esc(h.text)} <span class="src ${cls}" title="${tip}">${label}</span></div>`;
}
// A brief point that states a concern links to it (the concern is written once, under Highlighted concerns).
function seeLinks(ids, p) {
  const num = Object.fromEntries(concernList(p).map(f => [f._i, f._n]));
  return ids.map(id => {
    const i = p.flags.findIndex(f => f.id === id);
    if (i < 0) return "";
    const n = num[i];
    return `<a class="kf-link" data-concern="${i}" title="Open this concern">${n !== undefined ? n : "note"}</a> ${esc(p.flags[i].title)}`;
  }).filter(Boolean).join("; ");
}
function detailedCards(p) {
  const det = (p.summary || {}).detailed;
  if (!det || !det.brief || !det.brief.length) return "";
  const key = t => t.toLowerCase().replace("gaps to check", "gaps");
  const merged = Object.fromEntries((p.merged || []).map(b => [key(b.theme), b]));
  return `<div class="brief-grid">${det.brief.map((b, i) => {
    const [icon, accent, wide] = THEMES[key(b.theme)] || ["📌", "#3B82F6", true];
    const m = merged[key(b.theme)] || {};
    const why = m.rule || m.evidence ? `<details class="evid"><summary>Evidence and rule</summary><div class="dbody">${themeEvidence(m)}${m.rule ? ruleBox(m.rule) : ""}</div></details>` : "";
    return `<div class="brief-card${wide ? " brief-wide" : ""}" style="--accent:${accent};animation-delay:${i * 60}ms">
      <div class="brief-head"><span>${icon}</span>${esc(b.theme)}</div>
      <ul class="brief-points">${b.points.map(x => { const [cls, label, tip] = SRC[x.src] || SRC.rule;
        const body = x.see ? `<span class="see">→ ${seeLinks(x.see, p)}</span>` : pointHtml(x.text, p);
        return `<li${areasAttr(x.areas)}>${body} <span class="src ${cls}" title="${tip}">${label}</span></li>`; }).join("")}</ul>${why}</div>`;
  }).join("")}</div>`;
}
function courseHead(p, c) {
  const pts = (((p.summary || {}).detailed || {}).course || {}).brief;
  if (!pts) return `<div class="course-head">${esc(c.headline || "")}</div>`;
  return pts.length ? `<div class="course-head">${pts.map(x => { const [cls, label, tip] = SRC[x.src] || SRC.rule;
    return `${cited(x.text, p)} <span class="src ${cls}" title="${tip}">${label}</span>`; }).join("<br>")}</div>` : "";
}
function tabAI(p) {
  const useAI = p.ai.length > 0;
  const concerns = concernList(p), info = p.flags.filter(f => lvl(f) === "info"), dq = p.flags.filter(f => lvl(f) === "data_quality");
  const c = p.course || {};
  const dates = p.snap_dates || [];
  const sel = state.enc;
  const selVisit = sel ? p.visits.find(v => v.enc === sel) : null;
  const corr = p.corrections ? `<div class="small" style="margin-bottom:6px"><b>This review:</b> ${p.corrections.total} validator correction(s): ${esc(p.corrections.text)}</div>` : "";
  const clog = (p.citation_log || []).length ? `<details><summary>Citation corrections (${p.citation_log.length})</summary><div class="dbody">${p.citation_log.map(c => `<div class="small">${esc(c.point)}: ${c.removed.length ? "removed " + esc(c.removed.join(", ")) : ""}${c.removed.length && c.added.length ? "; " : ""}${c.added.length ? "cited " + esc(c.added.join(", ")) : ""}</div>`).join("")}</div></details>` : "";
  return `
  <div class="ai-bar">
    <span class="small muted">${useAI ? `🤖 AI brief merged with the rule-based brief${STATUS ? " • " + esc(STATUS) : ""} • ${p.encounters} visits • AI generated ${esc(p.ai_generated)} UTC • ${esc(p.ai_model)}` : "🧮 Rule-based brief from the record (no AI tokens). Generate the AI brief for an AI-written analysis."}</span>
    <span class="src-legend"><span class="src src-both">AI + Rule</span> both agree <span class="src src-ai">AI · checked</span> AI only <span class="src src-rule">Rule-based</span> rules only</span>
  </div>
  ${headlineHtml(p)}
  ${detailedCards(p) || mergedCards(p.merged && p.merged.length ? p.merged : p.brief.map(b => ({ ...b, src: b.points.map(() => "rule") })), p)}
  ${selVisit && state.encFrom === "brief" ? visitBox(selVisit) : ""}
  ${useAI ? `<details class="rulebox"><summary>Checks every AI point went through</summary><div class="dbody">${corr}${clog}<ul class="points">${DATA.ai_checks.map(x => `<li>${esc(x)}</li>`).join("")}</ul><div class="rule-src">dataanalysis/review_rules.py · sanitize_review()</div></div></details>` : ""}
  ${p.ai_abnormal.length ? `<details><summary>🧪 Lab details: exact values and reference ranges (${p.ai_abnormal.length})</summary><div class="dbody"><ul class="points">${p.ai_abnormal.map((x, k) => `<li${areasAttr((p.ai_abnormal_areas || [])[k])}>${esc(x.replace(/\*\*/g, ""))}</li>`).join("")}</ul></div></details>` : ""}
  ${p.ai_evidence.length ? `<details><summary>📎 Source visits</summary><div class="dbody">${p.ai_evidence.map(e => `<span class="evidence-chip">📌 ${esc(e)}</span>`).join("")}</div></details>` : ""}
  <div class="ai-declaration-footer">⚖️ Retrospective record-based observations only; clinician review is required for interpretation and decision-making.</div>
  ${p.ai_markdown ? `<details><summary>📋 View / Copy Raw Markdown for EHR Export</summary><div class="dbody"><pre class="md">${esc(p.ai_markdown)}</pre></div></details>` : ""}

  <div class="snap-card">
    ${concerns.length ? `<div class="snap-sec">⚠️ Highlighted concerns <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">— grouped by topic, most serious first · open a line for visits, full evidence, raw data and rule</span></div>` : `<div class="snap-sec">✅ No high-risk pattern detected by the record rules</div>`}
    ${concerns.map((f, k) => `${f._group && (k === 0 || concerns[k - 1]._group !== f._group) ? `<div class="snap-group">${esc(f._group)} <span>(${concerns.filter(c => c._group === f._group).length})</span></div>` : ""}<details class="snap-flag snap-${f.record_error ? "error" : esc(f.level)}" data-concern-row="${f._i}"${areasAttr(f.area ? [f.area] : [])}><summary><span class="snap-more">raw data · rule ▾</span><span class="c-num">${f._n}</span>${f.time ? `<span class="badge t-${f.time === "past" ? "past" : "latest"}">${f.time === "past" ? "Past" + (f.as_of ? " · " + esc(f.as_of) : "") : (f.as_of ? "Latest visit · " + esc(f.as_of) : (f.undated ? "Undated report" : "Latest visit"))}</span> ` : ""}${esc(f.summary || f.title)}</summary><div class="dbody"><div class="snap-full"><b>${esc(f.title)}</b><br><span class="k">Full evidence:</span> ${esc(f.evidence)}${f.significance ? `<br><span class="k">Why it matters:</span> ${esc(f.significance)}` : ""}</div>${rawBox(rawBlock(f.raw))}${ruleBox(f.rule)}</div></details>`).join("")}
    ${info.length ? `<div class="snap-sec">Also noted</div>${info.map(f => `<span class="snap-info" data-concern-row="${p.flags.indexOf(f)}"><b>${esc(f.title)}</b>: ${esc(f.evidence)}</span>`).join("")}` : ""}
    ${dq.length ? `<div class="snap-sec">Data quality (record problems, not findings)</div>${dq.map(f => `<span class="snap-info" data-concern-row="${p.flags.indexOf(f)}" style="border-left:3px dotted #8a94a6;background:#f6f7f9"><b>${esc(f.title)}</b>: ${esc(f.evidence)}</span>`).join("")}` : ""}
    ${p.guideline.length ? `<details class="snap-guide-box"><summary>Guideline context (ICMR / national) · ${p.guideline.length} row(s)</summary>${p.guideline.map(r => `<div class="snap-guide"><b>${esc(r.topic)}</b><span class="snap-status">${esc(r.status)}</span>${r.verified ? '<span class="badge b-ok">✓ Verified</span>' : '<span class="badge b-draft">Draft · not verified</span>'}<span>${esc(r.finding)} <span class="snap-src">— ${esc(r.source)}</span></span></div>`).join("")}</details>` : ""}
    ${p.snap_timeline.length ? `<div class="snap-sec">Timeline across visits <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:400">— click any column to open that visit</span></div>
    <div class="snap-tl">
      <div class="snap-row"><span class="snap-label"><b>Visit →</b></span>${dates.map((d, i) => `<span class="snap-colhead" data-enc="${i + 1}"><b>#${i + 1}</b>${esc(day(d))}</span>`).join("")}</div>
      ${p.snap_timeline.map(r => `<div class="snap-row"><span class="snap-label" title="${esc(r.label)}">${esc(r.label)}</span>${dates.map((d, i) => `<span class="snap-cell ${r.encs.includes(i + 1) ? (r.kind === "alert" ? "alert" : "on") : ""}${sel === i + 1 ? " sel" : ""}" data-enc="${i + 1}" title="#${i + 1} · ${esc(d)}"></span>`).join("")}</div>`).join("")}
    </div>
    <div class="snap-legend"><span><span class="snap-key" style="background:#029c89"></span>condition recorded at that visit</span><span><span class="snap-key" style="background:#d32f2f"></span>alert (raised BP, missed follow-up, abnormal report)</span><span><span class="snap-key" style="background:#f1f4f9;border:1px solid #d5ddeb"></span>not recorded</span></div>
    ${p.med_grid_html || ""}
    ${selVisit && state.encFrom !== "brief" ? visitBox(selVisit) : ""}` : ""}
  </div>
  <div class="small muted">Rule-based from the record (no AI tokens). Values marked 'undated report' come from documents not linked to a visit.</div>

  <div class="section-title">Clinical Course</div>
  <div class="course-card">
    ${courseHead(p, c)}
    <div class="course-grid">
      <div class="course-col c-ok"><h4>Consistent</h4><ul>${(c.consistent || []).map(x => `<li>${esc(x)}</li>`).join("") || "<li>None documented</li>"}</ul></div>
      <div class="course-col c-dx"><h4>Diagnosis shifts</h4><ul>${(c.diagnosis_shifts || []).map(x => `<li>${esc(x)}</li>`).join("") || "<li>None documented</li>"}</ul></div>
      <div class="course-col c-med"><h4>Medicine shifts</h4><ul>${(c.medicine_shifts || []).map(x => `<li>${esc(x)}</li>`).join("") || "<li>None documented</li>"}</ul></div>
    </div>
    <div class="course-sec">Timeline</div>
    ${(c.timeline || []).map(t => `<div class="course-tl-row"><div class="course-month">${esc(t.month)}<span>${esc(t.encounters)}</span></div><div>${esc(t.text)}</div></div>`).join("")}
  </div>
  ${p.reports.length ? `<details class="raw"><summary>Show documents not linked to a visit (${p.reports.length})</summary><div class="dbody">${p.reports.map(r => `<div class="card"><div class="card-head"><span class="card-title">${esc(r.label)}</span>${r.likely_visit ? `<span class="badge b-info">likely ${esc(r.likely_visit)}</span>` : ""}</div>${r.impression ? `<div class="small">${esc(r.impression)}</div>` : ""}${r.labs.length ? labTable(r.labs) : ""}</div>`).join("")}</div></details>` : ""}
  <div class="ending">Retrospective record-based observations only; clinician review is required for interpretation and decision-making.</div>`;
}

function render() {
  ROOT.innerHTML = tabAI(p);
  ROOT.querySelectorAll("[data-concern]").forEach(b => b.onclick = () => {
    const row = ROOT.querySelector(`[data-concern-row="${b.dataset.concern}"]`);
    if (!row) return;
    row.open = true; row.scrollIntoView({ behavior: "smooth", block: "center" });
    row.classList.add("flash"); setTimeout(() => row.classList.remove("flash"), 1600); });
  ROOT.querySelectorAll("[data-enc]").forEach(b => b.onclick = () => {
    const n = +b.dataset.enc; state.enc = state.enc === n ? 0 : n;
    state.encFrom = b.classList.contains("cite") ? "brief" : "grid"; render(); });
}
const state = { enc: 0, encFrom: "" };
render();
"""

# ── AI Analysis & Summary, Quick tier (summary.quick from summary_tiers.py) ───────────────────────────────────────
_AI_HELPERS_JS = _AI_JS.split("function tabAI(p) {")[0]
_QUICK_JS = r"""
const LEVEL = { high: ["b-high", "High"], verify: ["lvl-badge-verify", "Verify"], medium: ["b-medium", "Medium"] };
const HIDDEN = { now_overflow: "more current concern(s)", earlier_overflow: "more earlier concern(s)",
  past_concerns: "other past concern(s)", info: "note(s)", record_quality: "record-quality note(s)",
  labs: "abnormal lab value(s)", course: "course item(s)", tests_overflow: "more test(s) to check",
  treatment_overflow: "more medicine(s)" };
const TEST_STATE = { "missing": ["Not in record", "t-miss"], "undated": ["Undated report only", "t-warn"],
  "overdue": ["Overdue", "t-warn"], "advised, no result": ["Advised, no result", "t-miss"],
  "no repeat": ["Not repeated", "t-warn"], "due": ["Due", "t-warn"] };
const MAX_STOPPED = 4;
// Encounter numbers in '#2–4, #6' style text -> [2, 3, 4, 6].
function encNumbers(text) {
  const out = [];
  for (const m of String(text).matchAll(/#?(\d+)(?:\s*[–-]\s*#?(\d+))?/g)) {
    const a = +m[1], b = m[2] ? +m[2] : a;
    for (let n = a; n <= b && n - a < 200; n++) out.push(n);
  }
  return out;
}
// [1, 2, 3, 4, 6, 10] -> '#1–4, 6, 10', one clickable span per run.
function citeRuns(ns) {
  const runs = [];
  [...new Set(ns)].sort((a, b) => a - b).forEach(n => {
    const r = runs[runs.length - 1];
    if (r && n === r[1] + 1) r[1] = n; else runs.push([n, n]);
  });
  return runs.map((r, k) => `<span class="q-cite" data-enc="${r[0]}" title="Open visit #${r[0]}">${k ? "" : "#"}${r[0]}${r[1] > r[0] ? "–" + r[1] : ""}</span>`).join(", ");
}
function quickLine(c, latest) {
  const [lcls, ltext] = LEVEL[c.level] || ["chip", c.level];
  let text = String(c.text || ""), ns = c.cites || [];
  const m = /\s*\(((?:#\d+(?:\s*[–-]\s*#?\d+)?[\s,]*)+)\)\s*\.?\s*$/.exec(text);  // citations written in the line
  if (m) { ns = encNumbers(m[1]); text = text.slice(0, m.index).replace(/[\s.;,]+$/, "") + "."; }
  const when = c.undated ? "undated report" : (c.as_of && c.as_of !== latest ? day(c.as_of) : "");
  const ai = c.src !== "rule" ? ` <span class="src src-both" title="${SRC.both[2]}">AI</span>` : "";
  return `<li class="q-row q-${esc(c.level)}"${areasAttr([c.area])}>
    <span class="badge ${lcls} q-badge">${ltext}</span>
    <span class="q-text">${esc(text)}${ai}</span>
    <span class="q-meta">${ns.length ? citeRuns(ns) : ""}${ns.length && when ? "<br>" : ""}${esc(when)}</span></li>`;
}
function quickTreatment(t) {
  const groups = (t.by_condition || []).map(g => {
    const specs = [...new Set(g.medicines.map(m => m.specialization).filter(Boolean))];
    // One specialty for the group: shown once on the right; several: next to each medicine.
    const each = specs.length > 1;
    const meds = g.medicines.map(m => `<span class="q-med" title="${esc(m.class || "")}"><b>${esc(m.name)}</b>${m.regimen ? ` <small>${esc(m.regimen)}</small>` : ""}${m.status !== "current" ? ` <span class="q-tag t-warn" title="Course still running; not on the latest prescription">not renewed</span>` : ""}${each ? " " + specTags([m.specialization]) : ""}</span>`).join("");
    return `<div class="q-rx"><div class="q-cond">${esc(g.condition)}</div><div class="q-meds">${meds}</div><div class="q-spec">${each ? "" : specTags(specs)}</div></div>`;
  }).join("");
  const st = t.recently_stopped || [];
  const shown = st.slice(0, MAX_STOPPED).map(m => `${esc(m.name)} <small>${esc(day(m.last))}</small>`).join(" · ");
  const more = st.length > MAX_STOPPED ? ` <span class="muted" title="${esc(st.slice(MAX_STOPPED).map(m => m.name).join(", "))}">+${st.length - MAX_STOPPED} more</span>` : "";
  const stopped = st.length ? `<div class="q-stopped"><span class="q-label">Stopped recently</span> ${shown}${more}</div>` : "";
  return groups || stopped ? groups + stopped : `<div class="small muted">No medicines on the latest prescription.</div>`;
}
function quickTest(t) {
  const [label, cls] = TEST_STATE[t.state] || [t.state, "t-warn"];
  const detail = t.state === "overdue" || t.state === "advised, no result" || t.state === "no repeat" ? t.detail : "";
  const basis = t.basis === "guideline" ? (t.verified ? ["t-ok", "Guideline · verified"] : ["t-draft", "Guideline · draft"]) : ["t-draft", "Record rule"];
  return `<li class="q-test"><b>${esc(t.test)}</b>
    <span class="q-tag ${cls}">${esc(label)}</span>${detail ? ` <small class="muted">${esc(detail)}</small>` : ""}
    ${t.reasons.length ? `<div class="q-for">for ${esc(t.reasons.join(", "))}${t.interval ? ` · recheck ${esc(t.interval)}` : ""} <span class="q-tag ${basis[0]}" title="${esc(t.source || "")}">${basis[1]}</span></div>` : `<div class="q-for"><span class="q-tag ${basis[0]}" title="${esc(t.source || "")}">${basis[1]}</span></div>`}</li>`;
}
function card(title, sub, count, body, cls) {
  return `<section class="q-card ${cls || ""}"><div class="q-head"><span class="q-title">${esc(title)}</span>${count !== "" ? `<span class="q-count">${count}</span>` : ""}<span class="q-sub">${esc(sub)}</span></div>${body}</section>`;
}
function tabQuick(p) {
  const s = p.summary || {}, q = s.quick || {};
  const sel = state.enc ? p.visits.find(v => v.enc === state.enc) : null;
  const hidden = Object.entries(q.hidden_counts || {}).filter(([k, n]) => n > 0 && HIDDEN[k]).map(([k, n]) => `<span class="q-chip">${n} ${esc(HIDDEN[k])}</span>`);
  const ai = q.ai ? `<div class="small muted">AI wording: ${q.ai.accepted} line(s) accepted${q.ai.rejected.length ? `, ${q.ai.rejected.length} kept as rule text (${esc(q.ai.rejected.map(r => r.reason).join("; "))})` : ""}.</div>` : "";
  const now = (q.now || []).length ? `<ul class="q-list">${q.now.map(c => quickLine(c, q.as_of)).join("")}</ul>` : `<div class="q-empty">No current concerns at the latest visit.</div>`;
  const earlier = (q.earlier || []).length ? card("Earlier", "past concerns still relevant to the latest prescription or an active condition", q.earlier.length,
    `<ul class="q-list">${q.earlier.map(c => quickLine(c, "")).join("")}</ul>`) : "";
  const tests = (q.tests_to_check || []).length ? card("Tests to check", "missing, undated or overdue", q.tests_to_check.length,
    `<ul class="q-list">${q.tests_to_check.map(quickTest).join("")}</ul>`) : "";
  return `
  ${headlineHtml(p)}
  ${card("Now", q.as_of ? "as of the latest visit · " + day(q.as_of) : "as of the latest visit", (q.now || []).length, now, "q-now")}
  ${sel ? visitBox(sel) : ""}
  ${earlier}
  <div class="q-cols">
    ${card("Treatment", "latest prescription, by condition", "", quickTreatment(q.treatment || {}))}
    ${tests}
  </div>
  <div class="q-hidden"><span class="q-label">Only in Detailed</span> ${hidden.length ? hidden.join("") : "nothing else; Detailed holds the full brief, course and reference"}</div>
  ${ai}
  <div class="ending">Retrospective record-based observations only; clinician review is required for interpretation and decision-making.</div>`;
}
function render() {
  ROOT.innerHTML = tabQuick(p);
  ROOT.querySelectorAll("[data-enc]").forEach(b => b.onclick = () => {
    const n = +b.dataset.enc; state.enc = state.enc === n ? 0 : n; render(); });
}
const state = { enc: 0, encFrom: "" };
render();
"""

_VIEW_HELP = (
    '<details class="rulebox"><summary>Quick and Detailed: what each shows</summary><div class="dbody small">'
    "<b>Quick</b> is one bedside screen, relative to the latest visit: the headline; <i>Now</i> (current concerns, most "
    "serious first, one line each with its date and #visit citations); <i>Earlier</i> (past concerns that still bear on "
    "the latest prescription or an active condition); <i>Treatment</i> (the latest prescription grouped by the condition "
    "each medicine treats, courses still running but not renewed, medicines stopped recently); <i>Tests to check</i> "
    "(missing, undated or overdue monitoring tests, one per test family, marked guideline-verified, guideline-draft or "
    "record rule). Lists are capped; the last line counts everything that is only in Detailed, so nothing is dropped.<br>"
    "<b>Detailed</b> has everything, each fact once: the brief (Key findings, Conditions, Treatment, Gaps; a point that "
    "states a concern links to it), every concern by topic and the notes, the clinical course (with its grids and "
    "timeline), and the reference (lab values with ranges, guideline rows, record-quality notes, unlinked documents, "
    "AI-cited visits, AI checks, EHR markdown).<br>"
    "Source tags: <i>AI + Rule</i> raised by the record rules and worded by the AI (checked against the evidence), "
    "<i>AI · checked</i> AI only, <i>Rule-based</i> record rules only. When the AI is unavailable or a line fails its "
    "checks, the rule text is shown. Every concern and abnormal result is checked to be in Detailed and either shown "
    "or counted in Quick.<br>"
    "The view that opens comes from <code>?view=quick</code> / <code>?view=detailed</code> in the address, else "
    "Quick. A headline that starts with 'Latest visit …, N "
    "months ago' means the record is old: 'now' means as of that visit."
    '<div class="rule-src">dataanalysis/summary_tiers.py · build_summary() · dataanalysis/quick_review.py</div>'
    "</div></details>")


def render_view_help() -> None:
    st.html(f'<style>{_CSS}</style><div class="cdx">{_VIEW_HELP}</div>')


def render_quick_body(view: dict) -> None:
    """Quick tier: headline, Now, Earlier, Treatment, Tests to check, and what only Detailed holds."""
    _mount(view, _AI_HELPERS_JS + _QUICK_JS)


# ── Tab: Lab Results ─────────────────────────────────────────────────────────
_LABS_JS = r"""
const state = { labFilter: "all" };
function tabLabs(p) {
  const rows = p.labs.filter(l => state.labFilter === "all" || /high|low|abnormal|critical|unverified/i.test(l.status || ""));
  const c = p.lab_counts || {};
  return `<div class="section-title">Lab results</div>
  <div class="section-desc">${c.results ?? p.labs.length} evaluated result(s) · ${c.out_of_range ?? 0} out of range (adverse) · ${c.unverified ?? 0} unverified · ${c.not_evaluated ?? 0} not evaluated</div>
  <div class="filters"><button class="fbtn${state.labFilter === "all" ? " active" : ""}" data-lab="all">All (${p.labs.length})</button><button class="fbtn${state.labFilter === "abn" ? " active" : ""}" data-lab="abn">Out of range only</button></div>
  <div class="card" style="padding:0">${labTable(rows)}</div>`;
}
function render() {
  ROOT.innerHTML = tabLabs(p);
  ROOT.querySelectorAll("[data-lab]").forEach(b => b.onclick = () => { state.labFilter = b.dataset.lab; render(); });
}
render();
"""


def _mount(view: dict, body_js: str, status: str = "") -> None:
    """One .cdx container and the script that renders into it (the static page's renderer for that tab)."""
    root_id = f"cdx-{uuid.uuid4().hex[:12]}"
    script = ("(function () {\n"
              f"const ROOT = document.getElementById({_json(root_id)});\n"
              "if (!ROOT) return;\n"
              f"const DATA = {_json(view)};\n"
              "const p = DATA.patient;\n"
              f"const STATUS = {_json(status)};\n"
              + _COMMON_JS + body_js + "\n})();")
    st.html(f"<style>{_CSS}{AREA_CSS}{MED_GRID_CSS}</style>"
            f'<div class="cdx" id="{root_id}"></div>'
            f"<script>{AREA_JS}{MED_GRID_JS}</script>{_script(script)}",
            unsafe_allow_javascript=True)


_TILES_RULE = (
    '<details class="rulebox"><summary>How the tiles are set</summary><div class="dbody small">One tile per area that '
    "applies to this patient (no BP readings and no hypertension → no BP tile). Status = the highest flag level in that "
    "area: high → red, medium → amber, none → green; no result in the record → grey \"Not tested\". The value is the "
    "latest result (HbA1c / glucose, BP, creatinine, liver enzyme, BMI), the missed follow-up count, the missing test, or "
    "the top medicine flag. Click a tile to highlight the brief points, lab details and concerns tagged with that area; "
    'click it again to clear.<div class="rule-src">dataanalysis/snapshot_view.py · status_tiles(), tag_review() · '
    "dataanalysis/area_tiles.py</div></div></details>")


def render_ai_header(view: dict) -> None:
    """Status tiles (click to highlight) and how they are set; then the Patient Brief heading."""
    patient = view["patient"]
    tiles = patient.get("tiles_html") or ""
    st.html(f"<style>{_CSS}{AREA_CSS}</style>"
            f'<div class="cdx">'
            f"{tiles}{_TILES_RULE if tiles else ''}</div><script>{AREA_JS}</script>",
            unsafe_allow_javascript=True)


def render_section_title(title: str) -> None:
    st.html(f'<style>{_CSS}</style><div class="cdx"><div class="section-title" style="margin:6px 0 0">'
            f"{html.escape(title)}</div></div>")


def render_ai_body(view: dict, status: str = "") -> None:
    """Patient Brief (merged AI + rule), checks, lab details, concerns, visit grid, Clinical Course, unlinked documents."""
    _mount(view, _AI_JS, status=status)


def render_labs_tab(view: dict) -> None:
    """Every lab result (deduplicated across copies), status badges, out-of-range filter."""
    _mount(view, _LABS_JS)
