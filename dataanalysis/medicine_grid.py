"""Medicine-class grid under the visit timeline, shared by the Streamlit AI tab and the static patient dashboard.

Rows come from snapshot['medicine_timeline'] (snapshot_view.medicine_timeline); columns are the visits, using the
same snap-row / snap-label / snap-cell / snap-colhead classes as the visit timeline so the columns line up.
Hovering a coloured cell shows the medicines of that class on that date.
"""
import datetime
import html
import re

MED_GRID_CSS = """
.snap-cell.med-on { background:#5b8def; border-color:#5b8def; cursor:help; }
.snap-cell.med-dup { background:#e65100; border-color:#e65100; cursor:help; }
.med-tip { position:fixed; z-index:10000; max-width:320px; background:#0f2a5f; color:#fff; font-size:0.78rem; line-height:1.4;
           padding:7px 10px; border-radius:8px; box-shadow:0 4px 16px rgba(0,0,0,0.25); pointer-events:none; white-space:pre-line;
           display:none; }
.med-grid-note { font-size:0.74rem; color:#5b6474; margin:2px 0 6px; }
.spec-tag { display:block; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:2px;
            font-size:0.54rem; font-weight:700; color:#fff; border-radius:999px; padding:0 1px; text-align:center;
            letter-spacing:-0.01em; }
"""

MED_GRID_JS = """
(function () {
  if (window.__medGridTip) return;
  window.__medGridTip = true;
  var tip = null;
  function el() { if (!tip) { tip = document.createElement("div"); tip.className = "med-tip"; document.body.appendChild(tip); } return tip; }
  document.addEventListener("mouseover", function (ev) {
    var c = ev.target.closest && ev.target.closest("[data-med-tip]");
    if (!c) return;
    var t = el(); t.textContent = c.getAttribute("data-med-tip"); t.style.display = "block";
    var r = c.getBoundingClientRect(), w = t.offsetWidth, h = t.offsetHeight;
    t.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
    t.style.top = (r.top - h - 8 < 8 ? r.bottom + 8 : r.top - h - 8) + "px";
  });
  document.addEventListener("mouseout", function (ev) {
    if (tip && ev.target.closest && ev.target.closest("[data-med-tip]")) tip.style.display = "none";
  });
  window.addEventListener("scroll", function () { if (tip) tip.style.display = "none"; }, true);
})();
"""


def _day(d):
    try:
        return datetime.date.fromisoformat(str(d)[:10]).strftime("%d %b").lstrip("0")
    except ValueError:
        return str(d or "date unknown")


# Colours for specialization tags: per patient, in order of first appearance, so no two of a patient's specialties
# share a colour; nothing per specialization is hard-coded.
_SPEC_COLOURS = ("#2563eb", "#d97706", "#7c3aed", "#0f766e", "#be185d", "#4d7c0f", "#0369a1", "#b45309")


def spec_palette(specialties):
    """{specialization: colour} for one patient (specialties: one list per visit)."""
    names = list(dict.fromkeys(s for specs in specialties or [] for s in specs or [] if s))
    return {s: _SPEC_COLOURS[k % len(_SPEC_COLOURS)] for k, s in enumerate(names)}


def spec_colour(name, palette=None):
    return (palette or {}).get(name) or _SPEC_COLOURS[sum(map(ord, str(name or ""))) % len(_SPEC_COLOURS)]


def spec_short(name):
    """A label that fits a visit column: the initials of a several-word name ('General Medicine' -> 'GM'), else its
    first five letters ('Orthopaedics' -> 'Ortho'). Derived from the name; nothing per specialization is listed."""
    words = [w for w in re.findall(r"[A-Za-z]+", str(name or "")) if w.casefold() not in ("and", "of", "the")]
    if len(words) > 1:
        return "".join(w[0].upper() for w in words)[:4]
    return words[0][:5] if words else ""


def spec_tags(specs, palette=None):
    """Small coloured tags with the short specialization label (the full name in the hover title)."""
    return "".join(f'<span class="spec-tag" style="background:{spec_colour(s, palette)}" title="Prescribed by: {html.escape(s)}">'
                   f"{html.escape(spec_short(s))}</span>" for s in specs or [] if s)


def spec_key(specialties, palette=None):
    """Key under the grid: each short label with its full specialization."""
    names = list(dict.fromkeys(s for specs in specialties or [] for s in specs or [] if s))
    return "".join(f'<span><span class="spec-tag" style="background:{spec_colour(s, palette)};display:inline-block;padding:0 5px">'
                   f"{html.escape(spec_short(s))}</span> {html.escape(s)}</span>" for s in names)


def medicine_grid_html(rows, dates, title="Medicines across visits", specialties=None):
    """The grid: one row per medicine class, a coloured cell where it was prescribed (orange: two of the class that
    day), the medicines in a hover tip with the specialization of the doctor who prescribed each. Under each visit
    date, the specialization(s) of that visit's prescribers (specialties: one list per visit; never a name).
    '' when the patient has no repeated medicine class."""
    if not rows:
        return ""
    specialties = specialties or []
    palette = spec_palette(specialties)
    head = ('<div class="snap-row"><span class="snap-label"><b>Visit →</b>'
            + ("<br><small>Prescribed by</small>" if any(specialties) else "") + "</span>"
            + "".join(f'<span class="snap-colhead"><b>#{i}</b>{html.escape(_day(d))}'
                      f'{spec_tags(specialties[i - 1] if i <= len(specialties) else [], palette)}</span>'
                      for i, d in enumerate(dates, 1))
            + "</div>")
    body = ""
    for r in rows:
        cells = {c["encounter"]: c for c in r["cells"]}
        same_day = {}
        for c in r["cells"]:
            same_day.setdefault(c["date"], []).append(c)
        spans = ""
        for i, d in enumerate(dates, 1):
            c = cells.get(i)
            if not c:
                spans += '<span class="snap-cell"></span>'
                continue
            specs = c.get("specializations") or []
            tip = f"#{i} · {_day(d)} · {r['label']}\n" + "\n".join(
                f"• {m}" + (f"  (Rx: {specs[k]})" if k < len(specs) and specs[k] else "")
                for k, m in enumerate(c["medicines"]))
            if c["duplicate"]:
                others = [o for o in same_day.get(c["date"], []) if o is not c]
                tip += "\n⚠ Same class twice on this date" + (
                    ": also " + "; ".join(f"{', '.join(o['medicines'])} (#{o['encounter']})" for o in others) if others else "")
            spans += (f'<span class="snap-cell {"med-dup" if c["duplicate"] else "med-on"}" '
                      f'data-med-tip="{html.escape(tip)}"></span>')
        body += (f'<div class="snap-row"><span class="snap-label" title="{html.escape(r["label"])}">'
                 f'{html.escape(r["label"])}</span>{spans}</div>')
    legend = ('<div class="snap-legend"><span><span class="snap-key" style="background:#5b8def"></span>prescribed at that visit</span>'
              '<span><span class="snap-key" style="background:#e65100"></span>two medicines of the class on the same date</span>'
              '<span><span class="snap-key" style="background:#f1f4f9;border:1px solid #d5ddeb"></span>not prescribed</span>'
              + (f"<span><b>Prescribed by:</b></span>{spec_key(specialties, palette)}" if any(specialties) else "") + "</div>")
    return (f'<div class="snap-sec">{html.escape(title)} <span style="text-transform:none;letter-spacing:0;font-weight:400">'
            '— NSAIDs and repeated medicine classes · hover a coloured cell to see the medicines</span></div>'
            f'<div class="snap-tl">{head}{body}</div>{legend}')
