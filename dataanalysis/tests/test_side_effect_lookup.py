import json

import analysis


def test_unlisted_medicine_uses_one_dynamic_batch_lookup(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis, "get_llm_cache", lambda key: None)
    monkeypatch.setattr(analysis, "set_llm_cache", lambda key, value: None)

    def fake_complete(messages, task, **kwargs):
        calls.append((messages, task, kwargs))
        return {"content": json.dumps({"Cetrizine": "Drowsiness, dry mouth, headache"})}

    monkeypatch.setattr(analysis, "llm_complete", fake_complete)
    result = analysis.get_possible_side_effects(["Cetrizine"], use_ai=True)

    assert len(calls) == 1
    assert "Drowsiness" in result["Cetrizine"]
    assert "AI-assisted general reference" in result["Cetrizine"]
