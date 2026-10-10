"""Patient-history read facade.

The implementation remains in ``analysis`` while callers transition to the
core package. The queries are read-only.
"""
from analysis import fetch_all_patient_ids, fetch_patient_history

__all__ = ["fetch_all_patient_ids", "fetch_patient_history"]
