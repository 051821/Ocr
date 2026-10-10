"""Canonical local SQLite cache interface.

``ai_analysis_csv`` remains as an import shim for existing integrations.
"""
from .ai_analysis_csv import (
    FULL_RERUN_THRESHOLD,
    compute_source_hash,
    get_llm_cache,
    get_patient_row,
    get_top_questions,
    record_chat_question,
    set_llm_cache,
    upsert_ai_analysis,
)

__all__ = [
    "FULL_RERUN_THRESHOLD",
    "compute_source_hash",
    "get_llm_cache",
    "get_patient_row",
    "get_top_questions",
    "record_chat_question",
    "set_llm_cache",
    "upsert_ai_analysis",
]
