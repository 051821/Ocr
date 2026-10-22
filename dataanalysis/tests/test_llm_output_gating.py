"""Review validation, LLM cache gating, fallback freshness, and image stripping."""
import json

import pytest

import ai_review_service
from core import llm_client
from review_rules import REQUIRED_ENDING, validate_model_review
from ui_components import strip_remote_images


def _review(*, title=False, ending=REQUIRED_ENDING, extra=(), overall="Glucose rose from 140 to 182 mg/dL."):
    lines = ["# AI Clinical Review"] if title else []
    lines += ["## Overall Pattern", overall, "## Key Insights", "- Insight.", "## Review Points", "- Point."]
    lines += list(extra)
    lines += ["## Evidence", "- Visit 1."]
    if ending:
        lines.append(ending)
    return "\n".join(lines)


def test_model_review_without_title_is_accepted():
    assert validate_model_review(_review())


def test_model_review_accepts_title_and_bolded_ending():
    assert validate_model_review(_review(title=True, ending=f"**{REQUIRED_ENDING}**"))


def test_incremental_review_keeping_considerations_section_is_accepted():
    assert validate_model_review(_review(title=True, extra=("## General Patient Considerations", "- Keep follow-up.")))


@pytest.mark.parametrize("markdown", [
    _review(ending=None),                       # cut off before the ending
    _review(overall=""),                        # empty Overall Pattern
    _review().replace("## Review Points", "## Notes"),
    "",
])
def test_incomplete_model_review_is_rejected(markdown):
    assert not validate_model_review(markdown)


class _Resp:
    def __init__(self, content, finish_reason="stop"):
        self._body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}], "usage": {}}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


@pytest.fixture
def cache(monkeypatch):
    store = {}
    monkeypatch.setattr(llm_client, "get_llm_cache", lambda key: store.get(key, (None,))[0])
    monkeypatch.setattr(llm_client, "set_llm_cache",
                        lambda key, value, patient_id=None: store.__setitem__(key, (value, patient_id)))
    return store


def _complete(monkeypatch, content, finish_reason="stop", **kwargs):
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _Resp(content, finish_reason))
    return llm_client.complete([{"role": "user", "content": "q"}], "review", api_key="k", **kwargs)


def test_valid_output_is_cached_with_patient_id(monkeypatch, cache):
    _complete(monkeypatch, "ok", patient_id="p-1", validate=lambda c: True)
    assert [pid for _, pid in cache.values()] == ["p-1"]


def test_truncated_output_is_returned_but_not_cached(monkeypatch, cache):
    result = _complete(monkeypatch, "partial", finish_reason="length")
    assert result["truncated"] and result["content"] == "partial"
    assert cache == {}


def test_invalid_output_is_not_cached(monkeypatch, cache):
    _complete(monkeypatch, "bad", validate=lambda c: False)
    assert cache == {}


def test_raising_validator_counts_as_invalid(monkeypatch, cache):
    result = _complete(monkeypatch, "bad", validate=lambda c: 1 / 0)
    assert result["content"] == "bad" and cache == {}


def test_invalid_cached_entry_is_regenerated(monkeypatch, cache):
    _complete(monkeypatch, "stale")
    result = _complete(monkeypatch, "fresh", validate=lambda c: c == "fresh")
    assert result["content"] == "fresh" and not result["cache_hit"]


def test_fallback_review_is_never_current(monkeypatch):
    row = {"source_hash": "h", "prompt_version": ai_review_service.PROMPT_VERSION,
           "model": ai_review_service.FALLBACK_MODEL, "review_md": ""}
    monkeypatch.setattr(ai_review_service, "get_patient_row", lambda pid: row)
    monkeypatch.setattr(ai_review_service, "_valid_review", lambda md: True)
    assert not ai_review_service.is_review_current({"patient_id": "p", "_review_source_hash": "h"})


def test_failed_generation_is_labelled_fallback(monkeypatch):
    saved = {}
    monkeypatch.setattr(ai_review_service, "get_patient_row", lambda pid: None)
    monkeypatch.setattr(ai_review_service, "analyze_patient_with_ai",
                        lambda *a, **k: {"raw_markdown": "no headings", "model_name": "some-llm"})
    monkeypatch.setattr(ai_review_service, "upsert_ai_analysis",
                        lambda data, md, model, *a, **k: saved.update(model=model) or {})
    report = ai_review_service.get_or_generate_review(
        {"patient_id": "p", "visits": [], "_review_facts": [], "_review_visit_hashes": {}, "_review_source_hash": "h"})
    assert report["model_name"] == saved["model"] == ai_review_service.FALLBACK_MODEL


@pytest.mark.parametrize("markdown", [
    "See ![x](https://evil.example/?d=diabetes)",
    "See ![x][ref]\n\n[ref]: https://evil.example/",
    "See ![](https://evil.example/a(b)c)",
    'See <img src="https://evil.example/">',
])
def test_remote_images_are_stripped(markdown):
    cleaned = strip_remote_images(markdown)
    assert "(image removed)" in cleaned
    assert "![" not in cleaned and "<img" not in cleaned


def test_links_and_text_survive_image_stripping():
    text = "Glucose 182 mg/dL [source](https://example.org) - **bold**"
    assert strip_remote_images(text) == text
