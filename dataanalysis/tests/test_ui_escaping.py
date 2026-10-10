"""Tests for Step 4: ENABLE_STRICT_HTML_SANITIZATION and esc() helper."""
import os
import pytest
from ui_components import esc


def test_esc_helper_with_flag_off(monkeypatch):
    """When ENABLE_STRICT_HTML_SANITIZATION is false, raw values pass through."""
    monkeypatch.setenv("ENABLE_STRICT_HTML_SANITIZATION", "false")
    raw = "<script>alert(1)</script>"
    assert esc(raw) == raw
    assert esc(None) == ""


def test_esc_helper_with_flag_on(monkeypatch):
    """When ENABLE_STRICT_HTML_SANITIZATION is true, HTML tags are escaped."""
    monkeypatch.setenv("ENABLE_STRICT_HTML_SANITIZATION", "true")
    raw = "<script>alert(1)</script>"
    escaped = esc(raw)
    assert "<script>" not in escaped
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" == escaped
    assert esc(None) == ""
    assert esc("Normal medicine 500mg") == "Normal medicine 500mg"
