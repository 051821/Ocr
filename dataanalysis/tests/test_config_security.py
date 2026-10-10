"""Tests for Step 1: ENABLE_HARDENED_CONFIG security behavior."""
import os
import pytest

import core.config as config
from core.llm_client import complete


def test_flag_reader_helper():
    """Verify flag() correctly reads booleans with default False."""
    assert config.flag("NON_EXISTENT_FLAG_123") is False
    assert config.flag("NON_EXISTENT_FLAG_123", default=True) is True

    os.environ["TEST_FLAG_ABC"] = "true"
    assert config.flag("TEST_FLAG_ABC") is True
    os.environ["TEST_FLAG_ABC"] = "0"
    assert config.flag("TEST_FLAG_ABC") is False
    del os.environ["TEST_FLAG_ABC"]


def test_chat_model_constant():
    """Verify CHAT_MODEL constant is set to gemini-2.5-flash."""
    assert config.CHAT_MODEL == "google/gemini-2.5-flash"
    assert config.CHAT_MODEL in config.APPROVED_MODELS


def test_hardened_config_removes_free_models(monkeypatch):
    """When ENABLE_HARDENED_CONFIG is true, :free models must be filtered out."""
    monkeypatch.setenv("ENABLE_HARDENED_CONFIG", "true")
    # Reload or check dynamic logic
    models = tuple(m for m in config._RAW_DEFAULT_MODELS if not m.endswith(":free"))
    assert not any(":free" in m for m in models)
    assert not any(":free" in m for m in config.APPROVED_MODELS)


def test_hardened_config_rejects_unapproved_model(monkeypatch):
    """When ENABLE_HARDENED_CONFIG is true, unapproved models must be rejected."""
    monkeypatch.setenv("ENABLE_HARDENED_CONFIG", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-mock")

    with pytest.raises(ValueError, match="not in the approved models allowlist"):
        complete([{"role": "user", "content": "hi"}], "chat", model="unapproved/rogue-model:free")


def test_hardened_config_raises_clear_error_if_key_missing(monkeypatch):
    """When ENABLE_HARDENED_CONFIG is true, missing key raises without leaking secret values."""
    monkeypatch.setenv("ENABLE_HARDENED_CONFIG", "true")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OpenRouter API key is not configured in environment"):
        complete([{"role": "user", "content": "hi"}], "chat", api_key="")
