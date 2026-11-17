"""Shared clinical review sanitization, validation, and deterministic sections.

The model writes one concise 4-5 bullet Clinical Insights brief (course, conditions, key findings, treatment,
gaps) from the patient's evidence pack (clinical_insights). The overview line, the lab details and the evidence list are always rendered from the
record itself, so exact values, units and reference ranges never depend on the model, and every model line
is checked against the evidence.
"""
import logging
import re

from answer_verification import verify_answer
from core.tracing import traced

logger = logging.getLogger(__name__)


# Sections the model writes, in order.
REVIEW_SECTIONS = ("Clinical Insights",)
OUTPUT_SECTIONS = ("Overview", "Clinical Insights", "Abnormal Results", "Evidence")
REQUIRED_ENDING = "Retrospective record-based observations only; clinician review is required for interpretation and decision-making."
MAX_INSIGHTS = 5
MAX_INSIGHT_WORDS = 190

# Headings the model may echo back (the title, sections rendered from the record, or sections of an older
# review an update starts from). Their content is dropped; the record-based versions replace it.
_MODEL_EXTRA_HEADINGS = {"ai clinical review", "overview", "abnormal results", "evidence", "review points",
                         "general patient considerations", "overall pattern", "key insights",
                         "risks and medication considerations", "follow-up needs"}


def _plain(line):
    """Line text without surrounding Markdown emphasis."""
    return line.strip().strip("*_").strip()


def _heading_name(line):
    return _plain(re.sub(r"^#+\s*", "", line))


_IMPERATIVE = re.compile(
    r"\b(?:prescribe|start|stop|increase|decrease|switch|adjust|administer|take|hold)\b"
    r".{0,80}\b(?:medication|medicine|drug|dose|tablet|capsule|\d+\s*(?:mg|mcg|ml|iu))\b",
    re.I,
)


def _trim_incomplete_lines(lines, max_words=400):
    kept, count = [], 0
    for line in lines:
        words = len(line.split())
        if count + words > max_words:
            break
        if line.startswith("- ") and line.rstrip().endswith((",", ":")):
            break
        if line and not line.startswith(("#", "- ")) and not re.search(r"[.!?)]$", line):
            break
        kept.append(line)
        count += words
    return kept


_THEME_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*\*\*(?:course|conditions|key findings|treatment|gaps to check)"
                           r"\s*:?\s*\*\*", re.I)


def with_insights_heading(markdown):
    """Add the '## Clinical Insights' heading when the model wrote the theme bullets without it.

    Models often drop the heading but write the brief correctly; a missing heading is a formatting slip,
    not a reason to discard the brief or pay for a retry.
    """
    lines = str(markdown or "").splitlines()
    if any(line.lstrip().startswith("#") and _heading_name(line).casefold() == "clinical insights" for line in lines):
        return str(markdown or "")
    first = next((i for i, line in enumerate(lines) if _THEME_BULLET.match(line)), None)
    if first is None:
        return str(markdown or "")
    # Theme bullets after another heading (an echoed "## Overview") still belong to the brief.
    return "\n".join(lines[:first] + ["## Clinical Insights"] + lines[first:])


def _model_insights(markdown):
    """The model's Clinical Insights bullets, without echoed sections, tables or prescribing sentences."""
    lines, current = [], None
    for raw in with_insights_heading(markdown).splitlines():
        line = raw.strip()
        if not line or line.startswith("```") or _plain(line) == REQUIRED_ENDING:
            continue
        if line.startswith("#"):
            current = _heading_name(line).casefold()
            continue
        if current != "clinical insights" or line.startswith("|") or re.fullmatch(r"[-|: ]+", line):
            continue
        line = re.sub(r"^(?:[*•]|\d+[.)])\s+", "- ", line)
        if not line.startswith("- "):
            line = "- " + line  # "**Theme:** text" written without a bullet is still one item
        if re.match(r"(?i)^-\s*(?:\*\*[^*]+\*\*:?\s*)?(?:none|nil|n/?a|not applicable)\b", line):
            continue  # an empty theme ("Abnormal blood tests: None in the record.") says nothing
        if _IMPERATIVE.search(line):
            safe = [part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip() and not _IMPERATIVE.search(part)]
            line = " ".join(safe)
            if not line:
                continue
        lines.append(line)
    return lines


def _record_names(patient_data):
    visits = patient_data.get("visits") or []
    meds = {str(m.get("name") or "") for v in visits for m in v.get("db_medications") or []}
    labs = {str(l.get("test_name") or "") for v in visits for l in v.get("lab_results") or []}
    labs |= {str(l.get("test_name") or "") for d in patient_data.get("unlinked_documents") or []
             for l in d.get("lab_results") or []}
    return sorted(m for m in meds if m), sorted(l for l in labs if l)


def _assemble(evidence, insights):
    from clinical_insights import (cite_points, complete_flag_points, dedupe_gaps, record_overview,
                                   render_abnormal_results, render_evidence, route_record_issues,
                                   brief_violations, deterministic_insights, finalise_brief, plain_wording, use_first_recorded_wording, use_record_terms)

    # Record terms, record errors out of Key findings, then only citations each encounter supports.
    insights = cite_points(complete_flag_points(dedupe_gaps(route_record_issues(use_record_terms(insights, evidence)),
                                                            evidence), evidence), evidence)
    insights = use_first_recorded_wording(insights, evidence)
    # The UI calls an encounter a "visit"; the model's wording follows.
    insights = [re.sub(r"\b([Ee])ncounter(s?)\b", lambda m: ("V" if m.group(1) == "E" else "v") + "isit" + m.group(2), x)
                for x in insights]
    insights = plain_wording(insights)
    insights = finalise_brief(insights, evidence)
    # Validator: a brief that still breaks the format is rejected for the rule-built brief in the same format.
    violations = brief_violations(insights, evidence)
    if violations:
        fallback = finalise_brief(plain_wording(deterministic_insights(evidence)), evidence)
        left = brief_violations(fallback, evidence)
        evidence.setdefault("_metrics", {})["format_violations"] = violations
        if len(left) < len(violations):
            insights = fallback
            evidence["_metrics"]["format_rejected"] = True
            logger.info("brief rejected for format: %s", "; ".join(violations))

    return "\n".join(["# AI Clinical Review", "## Overview", record_overview(evidence),
                      "## Clinical Insights", *insights,
                      "## Abnormal Results", *render_abnormal_results(evidence),
                      "## Evidence", *render_evidence(evidence, insights), "", REQUIRED_ENDING])


_SELF_CHECK_SYSTEM = """You check a short clinical brief against the PATIENT EVIDENCE it was written from. For every numbered
point give a verdict:
- "ok": every fact is in the evidence and worded correctly.
- "fix": something is wrong or misleading: a medicine grouped under or linked to a condition that its
  medicines[].for does not list; a medicine called current when at_latest_visit is false; a wrong count, date,
  value or encounter; a condition status that differs from conditions[].status; a cause or interpretation the
  evidence does not state ("consistent with", "suggests", "due to"); a rule-flag title copied as a sentence;
  an ORDER the dates do not show: "despite X" when X's medicines[].first is on or after the result's date or the
  result has no date, "X stopped after Y" when Y is not dated between X's first and last, "started after Y" when X
  began before Y, or "stopped because Y" (the record never states why a medicine changed).
  Give "text": the corrected point, using only evidence facts and no longer than the original.
- "drop": not supported by the evidence, generic (would fit any patient), or a repeat of another point.
Return only JSON: {"points": [{"id": "<id>", "verdict": "ok" | "fix" | "drop", "text": "<only for fix>"}]}"""


def _point_split(lines):
    """[(prefix, [points])] for '- **Theme:** a; b' bullets."""
    out = []
    for line in lines:
        prefix = re.match(r"^(\s*-\s+(?:\*\*[^*]+\*\*:?\s*)?)?", line).group(0)
        out.append((prefix, [p.strip() for p in line[len(prefix):].split("; ") if p.strip()]))
    return out


@traced(name="ai_review.self_check", process_inputs=lambda i: {"lines": i.get("lines"), "patient_id": i.get("patient_id")})
def self_check(lines, evidence, evidence_json, patient_id=None):
    """Second model pass that checks each point of the brief against the evidence and fixes or drops it.

    Guards: a fix may not bring in a number that is neither in the point nor in the evidence and may not grow
    much; when the check drops more than half of the points it is ignored. Any failure keeps the lines as they
    are, so the check can only remove or correct, never block a review.
    """
    import json
    from clinical_terms import enabled
    from core.config import CHAT_MODEL
    from core.llm_client import complete

    split = _point_split(lines)
    numbered = {f"{b}.{p}": text for b, (_, points) in enumerate(split, 1) for p, text in enumerate(points, 1)}
    if not enabled() or not numbered:
        return lines

    def parse(content):
        match = re.search(r"\{.*\}", str(content or ""), flags=re.S)
        return json.loads(match.group(0)) if match else None

    try:
        reply = complete([{"role": "system", "content": _SELF_CHECK_SYSTEM},
                          {"role": "user", "content": "PATIENT EVIDENCE:\n" + evidence_json + "\n\nPOINTS:\n"
                           + "\n".join(f"{k} {v}" for k, v in numbered.items())}],
                         "review_check", model=CHAT_MODEL, temperature=0, patient_id=patient_id,
                         validate=lambda c: isinstance((parse(c) or {}).get("points"), list))
        verdicts = {str(v.get("id")): v for v in parse(reply["content"])["points"] if isinstance(v, dict)}
    except Exception as exc:
        logger.warning("Review self-check skipped: %s", exc)
        return lines

    known_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence_json))
    decided = {}
    for key, text in numbered.items():
        verdict = verdicts.get(key) or {}
        kind, fix = str(verdict.get("verdict") or "ok").casefold(), str(verdict.get("text") or "").strip()
        if kind == "drop":
            decided[key] = None
        elif kind == "fix" and fix:
            new_numbers = set(re.findall(r"\d+(?:\.\d+)?", fix)) - set(re.findall(r"\d+(?:\.\d+)?", text)) - known_numbers
            fits = len(fix.split()) <= len(text.split()) * 1.3 + 4
            decided[key] = fix if not new_numbers and fits else text
        else:
            decided[key] = text
    if sum(v is None for v in decided.values()) * 2 > len(decided):
        logger.warning("Review self-check dropped most points; ignoring it")
        return lines
    out = []
    for b, (prefix, points) in enumerate(split, 1):
        kept = [decided[f"{b}.{p}"] for p in range(1, len(points) + 1) if decided[f"{b}.{p}"]]
        if kept:
            out.append(prefix + "; ".join(kept))
    return out


@traced(name="ai_review.sanitize_review")
def sanitize_review(markdown, patient_data):
    """Rebuild a model review around the record-based overview, lab details and evidence.

    Each insight is kept only if it suggests no medicine change, states no value, date, dose or encounter
    that the evidence pack lacks, gives no condition a diagnosis status the record does not support, and calls no
    condition active / current unless the latest visit, known history or the long-term list shows it, and orders no
    events against the dates ("despite X" before X started, "stopped after Y" before X began, any stated cause).
    If nothing survives, the rule-based insights are used instead.
    """
    from clinical_insights import (build_insight_evidence, deterministic_insights, drop_filler_points,
                                   drop_unsupported_active_claims, drop_unsupported_control_claims,
                                   drop_implausible_cause_claims, drop_unsupported_stop_claims,
                                   drop_unsupported_timing_claims, fix_new_condition_claims, fix_record_contradictions,
                                   replace_contradicted_points,
                                   drop_unsupported_medicine_links, drop_unsupported_status_claims, ensure_key_points,
                                   evidence_for_prompt)

    evidence = build_insight_evidence(patient_data)
    evidence["_metrics"] = {}
    count_points = lambda ls: sum(len([p for p in l.split("; ") if p.strip()]) for l in ls)
    model_points = count_points(_model_insights(markdown))
    medicines, labs = _record_names(patient_data)
    text = strip_care_suggestions("\n".join(_model_insights(markdown)), medicines=medicines, tests=False)
    text, _ = verify_answer(text, evidence_for_prompt(evidence), len(patient_data.get("visits") or []),
                            medicines=medicines, lab_names=labs)
    kept = drop_unsupported_status_claims([line for line in text.splitlines() if line.strip()], evidence)
    kept = fix_new_condition_claims(replace_contradicted_points(kept, evidence), evidence)
    # The model's own check of each point against the evidence (grouping, timing, speculation); the
    # deterministic checks below then run on its output as well.
    kept = self_check(kept, evidence, evidence_for_prompt(evidence), patient_data.get("patient_id"))
    kept = drop_unsupported_control_claims(drop_unsupported_medicine_links(kept, evidence), evidence)
    kept = drop_unsupported_timing_claims(drop_unsupported_active_claims(kept, evidence), evidence)
    kept = drop_implausible_cause_claims(drop_unsupported_stop_claims(kept, evidence), evidence)
    kept = drop_filler_points(fix_record_contradictions(kept, evidence), evidence)
    kept = _trim_incomplete_lines(kept, MAX_INSIGHT_WORDS)[:MAX_INSIGHTS]
    evidence["_metrics"]["validator_removed_points"] = max(0, model_points - count_points(kept))
    # Must-not-miss points are added from the record after trimming, so a length cut can never drop them.
    review = _assemble(evidence, ensure_key_points(kept, evidence) if kept else deterministic_insights(evidence))
    # Corrections the validator made, for the dashboard and the scale metrics.
    patient_data["_review_metrics"] = dict(evidence["_metrics"])
    return review


def validate_review(markdown):
    """Validate only the final review's required Markdown structure."""
    if not markdown or not markdown.startswith("# AI Clinical Review"):
        return False
    if not markdown.rstrip().endswith(REQUIRED_ENDING):
        return False
    headings = re.findall(r"^## (.+)$", markdown, flags=re.M)
    if headings != list(OUTPUT_SECTIONS):
        return False
    insights = re.search(r"## Clinical Insights\s+(.+?)(?=\n## )", markdown, flags=re.S)
    return bool(insights and insights.group(1).strip())


def validate_model_review(markdown):
    """Check the model's Clinical Insights section before the record-based sections are added.

    Lenient on formatting that sanitize_review rebuilds anyway (an optional "# AI Clinical Review" title,
    heading level/case, a bolded ending line, echoed record-based sections). Strict on content: a non-empty
    Clinical Insights section and the required ending, whose absence usually means the reply was cut off.
    """
    lines = [line.strip() for line in with_insights_heading(markdown).splitlines()
             if line.strip() and not line.strip().startswith("```")]
    if not lines or _plain(lines[-1]) != REQUIRED_ENDING:
        return False
    headings, insights, current = [], [], None
    for line in lines[:-1]:
        if line.startswith("#"):
            current = _heading_name(line).casefold()
            if current not in _MODEL_EXTRA_HEADINGS:
                headings.append(current)
        elif current == "clinical insights":
            insights.append(line)
    return headings == [name.casefold() for name in REVIEW_SECTIONS] and bool(insights)


@traced(name="ai_review.deterministic_fallback")
def deterministic_review(patient_data):
    """Patient-specific review built from the record alone, used when the model is unavailable."""
    from clinical_insights import build_insight_evidence, deterministic_insights

    evidence = build_insight_evidence(patient_data)
    return _assemble(evidence, deterministic_insights(evidence))


def sanitize_chat_answer(answer_text: str) -> str:
    """Sanitize LLM chat output by removing sentences with imperative prescribing directives.

    Preserves ### Answer and ### Evidence structure. Returns cleaned text.
    """
    if not answer_text or not isinstance(answer_text, str):
        return ""

    sections = {}
    current_sec = "body"
    sections[current_sec] = []

    for raw in answer_text.splitlines():
        line = raw.strip()
        if not line:
            sections[current_sec].append("")
            continue
        if line.startswith("### "):
            heading = line.lstrip("#").strip().casefold()
            if "answer" in heading:
                current_sec = "answer"
            elif "evidence" in heading:
                current_sec = "evidence"
            else:
                current_sec = heading
            sections.setdefault(current_sec, []).append(line)
            continue

        # Filter sentences within current line using _IMPERATIVE
        if _IMPERATIVE.search(line):
            sentences = re.split(r"(?<=[.!?])\s+", line)
            safe_sentences = [
                s.strip() for s in sentences
                if s.strip() and not _IMPERATIVE.search(s)
            ]
            if safe_sentences:
                clean_line = " ".join(safe_sentences)
                sections.setdefault(current_sec, []).append(clean_line)
        else:
            sections.setdefault(current_sec, []).append(line)

    # Reconstruct text
    result_lines = []
    for sec_name, lines in sections.items():
        result_lines.extend(lines)

    cleaned = "\n".join(result_lines).strip()
    return cleaned


# A suggestion phrase in the assistant's own voice ...
_SUGGESTION_CUE = re.compile(
    r"\b(?:consider(?:ing)?|recommend(?:ed|s|ing)?|suggest(?:ed)?|advisable|should|may need|might need|"
    r"needs? to|would benefit|warrant(?:s|ed)?|it is (?:important|advisable|prudent) to)\b",
    re.I,
)
_PASSIVE_CUES = {"recommended", "suggested", "advisable", "warranted"}
_TEST_WORDS = (r"tests?|testing|labs?|panel|screening|investigations?|work-?up|scan|x-?ray|ultrasound|ecg|"
               r"echo(?:cardiogram)?|repeat(?:ing|ed)?|re-?check(?:ing)?")
_MEDICINE_WORDS = (r"medicines?|medications?|drugs?|doses?|dosage|dosing|supplements?|tablets?|capsules?|"
                   r"prescri\w*|therap\w*|treatments?|regimens?|titrat\w*|initiat\w*")
# A change of treatment ("dose may need adjustment"); only counts when a medicine is named in the sentence,
# so lifestyle advice ("should start walking") is left alone.
_CHANGE_WORDS = re.compile(r"\b(?:adjust\w*|increas\w*|decreas\w*|reduc\w*|chang\w*|stop\w*|start\w*|switch\w*|"
                           r"add\w*|review\w*|optimi[sz]\w*)\b", re.I)
# Advice the record itself documents is reported, not suggested.
_DOCUMENTED = re.compile(r"\b(?:documented|recorded|per the record|in the record)\b|"
                         r"\b(?:note|consult\w*|report|clinician|doctor|physician|specialist)\s+"
                         r"(?:advised|recommended|suggested)\b", re.I)


def _word_alternation(base, names):
    names = [re.escape(str(n).strip()) for n in names if n and len(str(n).strip()) >= 3]
    return re.compile(r"\b(?:" + "|".join([base] + names) + r")\b", re.I)


def strip_care_suggestions(answer_text: str, medicines=(), labs=(), tests=True) -> str:
    """Remove sentences that suggest a medicine or a lab test; drop a bullet left empty.

    Runs on every chat answer regardless of feature flags: the chat reports the record and
    never proposes care. medicines/labs are this patient's own names, so "consider repeating TSH"
    is caught as well as generic wording. tests=False keeps monitoring suggestions (the review's
    Follow-up Needs) and removes only medicine suggestions.
    """
    if not answer_text or not isinstance(answer_text, str):
        return ""
    medicine = _word_alternation(_MEDICINE_WORDS, medicines)
    targets = (_TEST_WORDS + "|" if tests else "") + _MEDICINE_WORDS
    target = _word_alternation(targets, list(medicines) + (list(labs) if tests else []))

    def is_suggestion(sentence):
        cue = _SUGGESTION_CUE.search(sentence)
        if not cue or _DOCUMENTED.search(sentence):
            return False
        # Passive wording puts the target first ("a lipid panel is recommended").
        start = 0 if cue.group().casefold() in _PASSIVE_CUES else cue.start()
        if target.search(sentence, start):
            return True
        # A treatment change only counts next to a medicine ("dose may need review", "stop metformin"),
        # not "metformin should be taken with food to reduce nausea".
        return any(medicine.search(sentence[max(0, m.start() - 25):m.end() + 25])
                   for m in _CHANGE_WORDS.finditer(sentence, cue.start()))

    out = []
    for raw in answer_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not _SUGGESTION_CUE.search(line):
            out.append(raw)
            continue
        bullet = re.match(r"^(\s*(?:[-*•]|\d+[.)])\s+)", raw)
        prefix = bullet.group(1) if bullet else ""
        body = raw[len(prefix):]
        kept = [s for s in re.split(r"(?<=[.!?;])\s+", body.strip()) if s and not is_suggestion(s)]
        if kept:
            out.append(prefix + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
