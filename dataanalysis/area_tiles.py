"""Status tiles + tile-area highlighting, shared by the Streamlit AI tab and the static patient dashboard.

One component for both so they cannot drift: the tiles' markup, their CSS and the click script.
Anything carrying data-areas="glucose medicine" (brief points, lab details, concerns) is highlighted when the
tile of one of those areas is clicked; clicking the same tile again clears it.
"""
import html

AREA_CSS = """
.area-tiles { display:grid; grid-template-columns:repeat(auto-fill, minmax(150px, 1fr)); gap:10px; margin:4px 0 14px; }
.area-tile { text-align:left; font:inherit; cursor:pointer; border:1px solid; border-radius:10px; padding:8px 10px;
             background:none; color:#1f2937; }
.area-tile:hover { box-shadow:0 2px 8px rgba(16,78,178,0.15); }
.area-tile .t-label { font-size:0.7rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase;
                      display:flex; align-items:center; gap:5px; }
.area-tile .t-dot { width:8px; height:8px; border-radius:50%; flex:none; }
.area-tile .t-n { margin-left:auto; font-weight:600; letter-spacing:0; text-transform:none; }
.area-tile .t-value { font-size:0.86rem; font-weight:650; margin-top:3px; line-height:1.3; color:#1f2937; }
.area-tile.red { background:#fdecec; border-color:#f3b4b4; } .area-tile.red .t-dot { background:#d32f2f; } .area-tile.red .t-label { color:#a61b1b; }
.area-tile.amber { background:#fff4e2; border-color:#f6cf94; } .area-tile.amber .t-dot { background:#ff9800; } .area-tile.amber .t-label { color:#a35a00; }
.area-tile.green { background:#e7f6ec; border-color:#b9e2c6; } .area-tile.green .t-dot { background:#1b873f; } .area-tile.green .t-label { color:#1b6b3a; }
.area-tile.grey { background:#f3f5f8; border-color:#d5ddeb; } .area-tile.grey .t-dot { background:#9aa3b2; } .area-tile.grey .t-label { color:#5b6474; }
.area-tile.on { outline:2px solid #104eb2; outline-offset:1px; box-shadow:0 2px 10px rgba(16,78,178,0.25); }
.area-tiles-note { font-size:0.75rem; color:#5b6474; margin:-8px 0 12px; }
.area-hit { background:#fff4c2 !important; box-shadow:0 0 0 2px #f2c94c; border-radius:6px; transition:background 0.2s; }
"""

AREA_JS = """
(function () {
  if (window.__areaTiles) return;
  window.__areaTiles = true;
  var NOTE = "Click a tile to highlight its points and lab details below; click again to clear.";
  function note(text) { document.querySelectorAll(".area-tiles-note").forEach(function (n) { n.textContent = text; }); }
  function clear(doc) {
    note(NOTE);
    doc.querySelectorAll(".area-hit").forEach(function (e) { e.classList.remove("area-hit"); });
    doc.querySelectorAll(".area-tile.on").forEach(function (e) { e.classList.remove("on"); });
  }
  document.addEventListener("click", function (ev) {
    var tile = ev.target.closest && ev.target.closest("[data-area-tile]");
    if (!tile) return;
    ev.preventDefault();
    var area = tile.getAttribute("data-area-tile"), wasOn = tile.classList.contains("on");
    clear(document);
    if (wasOn) return;
    tile.classList.add("on");
    var hits = Array.prototype.slice.call(document.querySelectorAll("[data-areas]")).filter(function (e) {
      return e.getAttribute("data-areas").split(" ").indexOf(area) >= 0; });
    hits.forEach(function (e) {
      e.classList.add("area-hit");
      for (var d = e.parentElement; d; d = d.parentElement) if (d.tagName === "DETAILS") d.open = true;
    });
    var label = (tile.querySelector(".t-label") || tile).childNodes[1].textContent;
    if (hits.length) hits[0].scrollIntoView({ behavior: "smooth", block: "center" });
    note(hits.length ? hits.length + " item(s) about " + label + " highlighted; click the tile again to clear."
                     : "No brief point or lab detail is about " + label + " for this patient.");
  });
})();
"""


def areas_attr(areas):
    """' data-areas="glucose medicine"' for an element tagged with these tile areas ('' when none)."""
    areas = [a for a in (areas or []) if a]
    return f' data-areas="{html.escape(" ".join(areas))}"' if areas else ""


def tiles_html(tiles):
    """The patient's own tiles (only those generated for them, in the existing order) as clickable buttons."""
    if not tiles:
        return ""
    buttons = "".join(
        f'<button type="button" class="area-tile {html.escape(t.get("status", "grey"))}" data-area-tile="{html.escape(t["area"])}"'
        f' title="Highlight the points and lab details about {html.escape(t["label"])}; click again to clear">'
        f'<div class="t-label"><span class="t-dot"></span>{html.escape(t["label"])}'
        + (f'<span class="t-n">{t["concerns"]}</span>' if t.get("concerns") else "")
        + f'</div><div class="t-value">{html.escape(str(t.get("value") or ""))}</div></button>'
        for t in tiles)
    return (f'<div class="area-tiles">{buttons}</div>'
            '<div class="area-tiles-note">Click a tile to highlight its points and lab details below; click again to clear.</div>')
