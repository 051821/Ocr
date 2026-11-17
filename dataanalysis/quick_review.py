"""AI wording for the QUICK tier of the AI Analysis & Summary (the detailed tier keeps the existing review prompt).

One small LLM call per review (prompts/review_quick.md) returns JSON: {"headline": str, "lines": {flag id: str}}.
It only rewords the rule-built quick skeleton (summary_tiers.build_quick): the concerns, their order, the treatment
and the tests stay rule-built. apply() checks every AI line against the evidence pack before it replaces the rule
line; a line that fails keeps the rule text, and without AI output the whole quick tier is rule-built.

Checks per line: an id the skeleton has; at most line_words words; at least one #n citation, every #n an encounter
of the record and, when the item has its own citations, one of them; every number in the facts or the evidence pack;
no care suggestion (review_rules.strip_care_suggestions) and no unsupported record claim
(answer_verification.verify_answer); no level words. The headline: at most headline_words, numbers from the facts,
and it keeps the stale-record note when the rule headline has one.
"""
import json
import logging
import re

import summary_tiers
from summary_tiers import CONFIG

logger = logging.getLogger(__name__)
_NUMBER = re.compile(r"(?<![#\w.])\d+(?:\.\d+)?(?![\w])")
_LEVEL_WORDS = re.compile(r"(?i)\b(?:high|medium|info)[- ](?:level|priority|flag)\b|\blevel\s*:")
_STALE = re.compile(r"(?i)\bmonths? ago\b")
# Advice wording: the quick tier reports the record, it never tells the clinician what to do.
_ADVICE = re.compile(r"(?i)\b(?:consider\w*|recommend\w*|advis\w*|suggest\w*|should|must|needs? to|warrant\w*|"
                     r"please|refer\w*\s+to|rule out)\b")


# ── Prompt ──────────────────────────────────────────────────────────────────
def prompt_payload(quick, evidence, headline):
    """The QUICK EVIDENCE JSON: headline facts and one item per concern line (id, level, time, as_of, facts, cites)."""
    pt = evidence.get("patient") or {}
    items = [{"id": c["id"], "level": c["level"], "time": c["time"], "as_of": c["as_of"] or "undated",
              "facts": f"{c['title']}: {c.get('_evidence') or c['text']}", "cites": [f"#{n}" for n in c["cites"]]}
             for c in quick["now"] + quick["earlier"]]
    return {"headline_facts": {"rule_headline": headline, "age_sex": pt.get("age_sex"),
                               "visits": pt.get("clinical_days") or pt.get("encounters"),
                               "latest_visit": pt.get("latest_visit")},
            "items": items}


def _system_prompt():
    from core.prompts import load
    return load("review_quick.md")


def parse(content):
    """{'headline': str, 'lines': {id: str}} from the model's reply (a fenced block is allowed), or None."""
    text = str(content or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("lines", {}), dict):
        return None
    lines = {str(k): str(v) for k, v in (data.get("lines") or {}).items() if isinstance(v, str)}
    return {"headline": str(data.get("headline") or ""), "lines": lines}


def generate(patient_data, today=None):
    """The AI quick tier for this patient ({'headline', 'lines', 'model'}), or None when the call fails or returns
    nothing usable (the quick tier is then rule-built)."""
    from clinical_insights import build_insight_evidence
    from clinical_snapshot import build_snapshot
    from core.config import MAX_TOKENS
    from core.llm_client import complete as llm_complete

    try:
        snapshot = build_snapshot(patient_data) or {}
        evidence = build_insight_evidence(patient_data)
        quick = summary_tiers.build_quick(snapshot, evidence, today)
        _attach_evidence(quick, snapshot)
        payload = prompt_payload(quick, evidence, quick["headline"]["text"])
        messages = [{"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": "QUICK EVIDENCE (JSON):\n" + json.dumps(payload, ensure_ascii=False)}]
        result = llm_complete(messages, "review_quick", temperature=0.1, max_tokens=MAX_TOKENS["review_quick"],
                              patient_id=patient_data.get("patient_id"), validate=lambda c: parse(c) is not None)
        parsed = parse(result.get("content"))
        return {**parsed, "model": result.get("model", "")} if parsed else None
    except Exception as exc:  # the quick tier falls back to the rule text
        logger.warning("AI quick tier failed; using the rule-built quick tier: %s", exc)
        return None


def _attach_evidence(quick, snapshot):
    by_id = {f["id"]: f for f in snapshot.get("flags") or []}
    for c in quick["now"] + quick["earlier"]:
        c["_evidence"] = (by_id.get(c["id"]) or {}).get("evidence", "")


# ── Validation ──────────────────────────────────────────────────────────────
def _numbers_supported(text, allowed):
    for n in _NUMBER.findall(text):
        if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", allowed):
            return n
    return None


def check_line(text, item, evidence, evidence_text, medicines=()):
    """None when the AI line may replace the rule line; else the reason it may not."""
    from answer_verification import verify_answer
    from review_rules import strip_care_suggestions

    text = " ".join(str(text or "").split())
    if not text:
        return "empty"
    if summary_tiers.words(text) > CONFIG["line_words"]:
        return f"over {CONFIG['line_words']} words"
    cited = summary_tiers.cites(text)
    if not cited:
        return "no #n citation"
    encounters = {int(n) for n in (evidence.get("encounter_dates") or {})}
    if not set(cited) <= encounters:
        return f"cites an encounter the record does not have: {sorted(set(cited) - encounters)}"
    if item["cites"] and not set(cited) & set(item["cites"]):
        return "cites none of the encounters that show this concern"
    facts = f"{item['title']} {item.get('_evidence', '')} {item['text']} {item['as_of']} {evidence_text}"
    bad = _numbers_supported(re.sub(r"#\d+(?:\s*[–-]\s*#?\d+)?", "", text), facts)
    if bad:
        return f"value {bad} is not in the evidence"
    if _LEVEL_WORDS.search(text):
        return "names a flag level"
    names = list(medicines) + [summary_tiers._first_word(m) for m in medicines if summary_tiers._first_word(m)]
    if _ADVICE.search(text) or strip_care_suggestions(text, medicines=names, tests=True).strip() != text.strip():
        return "suggests a care change"
    from patient_chat import _CONDITION_PATTERNS, _REFERENCE_DRUGS
    kept, problems = verify_answer(text, evidence_text, len(encounters), medicines=medicines,
                                   drug_names=_REFERENCE_DRUGS, conditions=_CONDITION_PATTERNS)
    unknown = [d for d in _REFERENCE_DRUGS if len(d) >= 4 and summary_tiers._mentions(text, d)
               and not summary_tiers._mentions(evidence_text, d)]
    if unknown:
        return f"names a medicine the record does not have: {unknown[0]}"
    if problems or not kept.strip():
        return "states a record fact the evidence does not have: " + ", ".join(str(v) for _, v in problems[:2])
    return None


def check_headline(text, rule_headline, evidence_text):
    text = " ".join(str(text or "").split())
    if not text:
        return "empty", text
    if _STALE.search(rule_headline) and not _STALE.search(text):
        text = rule_headline.split(". ")[0].rstrip(".") + ". " + text  # keep the stale-record note
    if summary_tiers.words(text) > CONFIG["headline_words"]:
        return f"over {CONFIG['headline_words']} words", text
    bad = _numbers_supported(text, f"{rule_headline} {evidence_text}")
    if bad:
        return f"value {bad} is not in the evidence", text
    if _LEVEL_WORDS.search(text):
        return "names a flag level", text
    return None, text


def apply(quick, ai_quick, evidence, medicines=None):
    """Put the checked AI wording into the rule-built quick tier, in place. Accepted lines get src 'both' (raised by
    the rules, worded by the AI); the headline src 'ai'. quick['ai'] records what was accepted and why lines were not."""
    from clinical_insights import evidence_for_prompt

    evidence_text = evidence_for_prompt(evidence)
    if medicines is None:
        medicines = list(evidence.get("medicine_names") or [])
    items = {c["id"]: c for c in quick["now"] + quick["earlier"]}
    report = {"model": ai_quick.get("model", ""), "accepted": 0, "rejected": []}
    for fid, text in (ai_quick.get("lines") or {}).items():
        item = items.get(fid)
        if not item:
            report["rejected"].append({"id": fid, "reason": "unknown id"})
            continue
        reason = check_line(text, item, evidence, evidence_text, medicines)
        if reason:
            report["rejected"].append({"id": fid, "reason": reason})
            continue
        item["text"], item["src"] = " ".join(text.split()), "both"
        report["accepted"] += 1
    if ai_quick.get("headline"):
        reason, text = check_headline(ai_quick["headline"], quick["headline"]["text"], evidence_text)
        if reason:
            report["rejected"].append({"id": "headline", "reason": reason})
        else:
            quick["headline"] = {"text": text, "src": "ai"}
    for c in items.values():
        c.pop("_evidence", None)
    quick["ai"] = report
    return quick
