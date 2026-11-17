"""FastAPI layer. Puts the repo root and dataanalysis/ on sys.path so their flat imports resolve."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAANALYSIS_DIR = os.path.join(PROJECT_ROOT, "dataanalysis")
for _p in (PROJECT_ROOT, DATAANALYSIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
