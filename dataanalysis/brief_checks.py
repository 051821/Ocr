"""Automatic checks on a built patient view (insight_view.build_view): what a reviewer would check by hand.

Each check reads only the view (the brief shown to the doctor and the record it came from) and returns issues as
{"check", "severity", "theme", "point", "detail"}. Errors are wrong as shown; warnings are possibly missing content.

  cite_out_of_range    a #n the patient does not have
  garbled_text         a citation or phrase left broken ('(–2)', '(Visit )', 'noted at')
  stale_latest_rx      'not on the latest prescription' for a medicine written on the latest day
  contradicts_record   'no <test> in the record' while the record holds that test's result
  not_covered          a high-level concern, an abnormal result or a long-term condition the brief never states

Used by build_view (the view carries 'brief_checks'), scripts/audit_sample.py (failure rate over a random sample) and
the golden regression tests (tests/test_golden.py).
"""
import re

import lab_validation

ERROR, WARNING = "error", "warning"

_GARBLED = re.compile(r"\(\s*[–-]\s*\d|\(\s*(?:visit|encounter)?\s*\)|\(#\)|\(\s*,|,\s*\)|"
                      r"\b(?:at|on|in|from)\s*(?:[.;)]|$)", re.I)
_NO_IN_RECORD = re.compile(r"(?i)\b(?:no|without)\s+((?:[\w\-]+(?:\s*/\s*|\s*,\s*|\s+or\s+|\s+)){1,8}?)(?:results?\s+)?in\s+the\s+record")
_NOT_LATEST = "not on the latest prescription"
_STOP = {"with", "from", "that", "this", "range", "result", "results", "level", "count", "total", "serum", "blood",
         "abnormal", "high", "low", "the", "and", "for", "record", "visit", "visits", "latest", "flag"}


def _norm(text):
    return str(text or "").casefold().replace("haemo", "hemo").replace("anaem", "anem").replace("oedema", "edema")


def _words(text):
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", _norm(text)) if w not in _STOP}


def _stated(words, text_words):
    """Any word of the item is stated (allowing 'diabetes' / 'diabetic': same first 5 letters)."""
    stems = {w[:5] for w in text_words}
    return any(w in text_words or (len(w) >= 5 and w[:5] in stems) for w in words)


def _points(view):
    for theme in view.get("merged") or []:
        for k, point in enumerate(theme.get("points") or []):
            yield theme["theme"], k, point


def _issue(check, severity, theme, point, detail):
    return {"check": check, "severity": severity, "theme": theme, "point": point, "detail": detail}


def check_view(view):
    """Issues found in one patient view (the dict build_view returns under 'patient')."""
    from clinical_insights import _expand_ranges
    from clinical_snapshot import _first_word
    from patient_chat import conditions_matching

    issues = []
    n_enc = int(view.get("encounters") or len(view.get("visits") or []))
    visits = view.get("visits") or []
    latest_day = visits[-1]["date"] if visits else ""
    latest_meds = {_first_word(m.get("name")) for v in visits if v["date"] == latest_day for m in v.get("meds") or []} - {""}
    labs = [l for l in view.get("labs") or [] if not str(l.get("status") or "").startswith("Unverified")]
    lab_families = {str(lab_validation.identify(l.get("test")) or "").split("_")[0] for l in labs} - {""}

    for theme, _, point in _points(view):
        # Citation range
        bad = sorted({int(n) for n in re.findall(r"#(\d+)", _expand_ranges(point)) if not 1 <= int(n) <= n_enc})
        if bad:
            issues.append(_issue("cite_out_of_range", ERROR, theme, point, f"cites {bad}; patient has {n_enc} encounters"))
        # Broken fragments
        m = _GARBLED.search(point.rstrip(" ."))
        if m and not re.search(r"(?i)\b(?:at|on|in|from)\s+#", point[m.start():]):
            issues.append(_issue("garbled_text", ERROR, theme, point, f"broken fragment '{m.group(0).strip()}'"))
        # A medicine written on the latest day called 'not on the latest prescription'
        low = point.casefold()
        if _NOT_LATEST in low:
            for med in latest_meds:
                listed = re.search(rf"{_NOT_LATEST}:[^;]*\b{re.escape(med)}", low) or \
                    re.search(rf"\b{re.escape(med)}\w*[^;:]*?\b{_NOT_LATEST}", low)
                if listed:
                    issues.append(_issue("stale_latest_rx", ERROR, theme, point,
                                         f"'{med}' is on the latest prescription day ({latest_day})"))
        # 'No <test> in the record' while the record has that test
        for m in _NO_IN_RECORD.finditer(point):
            for token in re.split(r"\s*/\s*|\s*,\s*|\s+or\s+", m.group(1)):
                family = str(lab_validation.identify(token.strip()) or "").split("_")[0]
                if family and family in lab_families:
                    issues.append(_issue("contradicts_record", ERROR, theme, point,
                                         f"the record holds a '{token.strip()}' result"))

    # Coverage: what the brief must state
    text = " ".join(p for _, _, p in _points(view))
    text_words = _words(text)
    linked = {i for theme in view.get("merged") or [] for ln in theme.get("links") or [] for i in ln}
    for i, f in enumerate(view.get("flags") or []):
        if f.get("level") == "high" and i not in linked and not _stated(_words(f.get("title")), text_words):
            issues.append(_issue("not_covered", WARNING, "", "", f"high-level concern not stated: {f.get('title')}"))
    for g in view.get("abnormal_labs") or []:
        if not (_stated(_words(g.get("test")), text_words) or _stated(_words(g.get("test_as_written")), text_words)):
            issues.append(_issue("not_covered", WARNING, "", "", f"abnormal result not stated: {g.get('test')} {g.get('value')}"))
    brief_conditions = set(conditions_matching(text))
    for name in view.get("long_term") or []:
        if not (set(conditions_matching(name)) & brief_conditions or _stated(_words(name), text_words)):
            issues.append(_issue("not_covered", WARNING, "", "", f"long-term condition not stated: {name}"))
    return issues


def summarise(issues):
    """{'errors': n, 'warnings': n, 'by_check': {check: n}}"""
    by = {}
    for i in issues:
        by[i["check"]] = by.get(i["check"], 0) + 1
    return {"errors": sum(i["severity"] == ERROR for i in issues),
            "warnings": sum(i["severity"] == WARNING for i in issues), "by_check": by}
