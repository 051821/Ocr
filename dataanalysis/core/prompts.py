"""Prompt file loader with version hashing for ENABLE_EXTERNAL_PROMPTS.

All hashing is deterministic: CRLF normalised to LF, stripped, then
sorted by (filename, content) so file ordering never matters.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Files that belong to each feature set (order-independent — sorted when hashed).
_REVIEW_FILES = [
    "shared_policy.md",
    "review_system.md",
    "review_user.md",
    "review_incremental.md",
]
_CHAT_FILES = [
    "shared_policy.md",
    "chat_mode_record.md",
    "chat_mode_knowledge.md",
    "chat_mode_general.md",
]


def _normalise(text: str) -> str:
    """CRLF → LF, strip leading/trailing whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


@lru_cache(maxsize=32)
def load(name: str) -> str:
    """Return the normalised content of prompts/<name>."""
    path = _PROMPTS_DIR / name
    return _normalise(path.read_text(encoding="utf-8"))


def _hash_files(filenames: list[str]) -> str:
    """Stable blake2b digest over sorted (filename, content) pairs."""
    h = hashlib.blake2b(digest_size=16)
    for fname in sorted(filenames):
        content = _normalise((_PROMPTS_DIR / fname).read_text(encoding="utf-8"))
        h.update(fname.encode("utf-8"))
        h.update(b"\x00")
        h.update(content.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# These are computed once at import time (the files do not change at runtime).
REVIEW_PROMPT_VERSION: str = "review_" + _hash_files(_REVIEW_FILES)
CHAT_PROMPT_VERSION: str = "chat_" + _hash_files(_CHAT_FILES)
SIDE_EFFECTS_PROMPT_VERSION: str = _hash_files(["side_effects_lookup.md"])
