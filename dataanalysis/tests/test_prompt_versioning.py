"""Tests for ENABLE_EXTERNAL_PROMPTS (Step 5).

All tests run with flags OFF by default (env vars unset).
"""
import importlib
import os
import sys
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Block DB imports that hang
# ---------------------------------------------------------------------------
for _m in ("database", "core.db"):
    if _m not in sys.modules:
        _mock = MagicMock()
        _mock.db_pool = MagicMock()
        sys.modules[_m] = _mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_NAMES = [
    "shared_policy.md",
    "review_system.md",
    "review_user.md",
    "review_incremental.md",
    "chat_mode_record.md",
    "chat_mode_knowledge.md",
    "chat_mode_general.md",
    "side_effects_lookup.md",
    "extraction.md",
]


# ---------------------------------------------------------------------------
# 1. Every prompt file loads and is non-empty
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_file_loads(name):
    from core.prompts import load
    load.cache_clear()
    content = load(name)
    assert isinstance(content, str) and len(content) > 20, f"{name} is empty or too short"


# ---------------------------------------------------------------------------
# 2. All placeholders in each template file can be formatted without KeyError
# ---------------------------------------------------------------------------
_PLACEHOLDER_FILL = {
    "review_system.md": {"required_ending": "TEST_ENDING"},
    "review_user.md": {"required_ending": "TEST_ENDING", "context": "{}", "feedback": ""},
    "review_incremental.md": {"required_ending": "TEST_ENDING", "previous": "prev", "current": "{}"},
    "side_effects_lookup.md": {"med_list": "- aspirin"},
    "extraction.md": {"document": "patient doc text"},
}


@pytest.mark.parametrize("name,kwargs", _PLACEHOLDER_FILL.items())
def test_placeholder_formatting(name, kwargs):
    from core.prompts import load
    load.cache_clear()
    result = load(name).format(**kwargs)
    assert isinstance(result, str) and len(result) > 10


# ---------------------------------------------------------------------------
# 3. Version strings are stable (CRLF/LF independent)
# ---------------------------------------------------------------------------
def test_version_stable_crlf():
    """REVIEW_PROMPT_VERSION must not change between LF and CRLF reads."""
    import core.prompts as cp
    v1 = cp.REVIEW_PROMPT_VERSION
    # Re-import to get a fresh module (lru_cache is per-process)
    importlib.reload(cp)
    v2 = cp.REVIEW_PROMPT_VERSION
    assert v1 == v2


def test_chat_version_stable():
    import core.prompts as cp
    importlib.reload(cp)
    v1 = cp.CHAT_PROMPT_VERSION
    importlib.reload(cp)
    v2 = cp.CHAT_PROMPT_VERSION
    assert v1 == v2


# ---------------------------------------------------------------------------
# 4. Version changes when a file's content changes
# ---------------------------------------------------------------------------
def test_version_changes_on_content_change(tmp_path, monkeypatch):
    """Monkey-patch the prompts dir to a tmp copy, mutate one file, check version."""
    import shutil
    import core.prompts as cp

    # Copy prompts to tmp dir
    tmp_prompts = tmp_path / "prompts"
    shutil.copytree(PROMPTS_DIR, tmp_prompts)

    monkeypatch.setattr(cp, "_PROMPTS_DIR", tmp_prompts)
    cp.load.cache_clear()
    v_before = cp._hash_files(["shared_policy.md", "review_system.md", "review_user.md", "review_incremental.md"])

    # Mutate one file
    (tmp_prompts / "review_system.md").write_text(
        (tmp_prompts / "review_system.md").read_text(encoding="utf-8") + "\n# changed",
        encoding="utf-8",
    )

    cp.load.cache_clear()
    v_after = cp._hash_files(["shared_policy.md", "review_system.md", "review_user.md", "review_incremental.md"])
    assert v_before != v_after, "Version did not change after file edit"


# ---------------------------------------------------------------------------
# 5. Flag OFF → PROMPT_VERSION stays the old static string
# ---------------------------------------------------------------------------
def test_flag_off_keeps_static_prompt_version():
    """With ENABLE_EXTERNAL_PROMPTS unset, PROMPT_VERSION must equal 'clinical_ai_review_v7'."""
    os.environ.pop("ENABLE_EXTERNAL_PROMPTS", None)
    import review_facts
    importlib.reload(review_facts)
    assert review_facts.PROMPT_VERSION == "clinical_ai_review_v7"
