"""Pytest global configuration and fixtures.

Mocks external network connections and database pools for unit test isolation.
"""
import sys
from unittest.mock import MagicMock

# Prevent ThreadedConnectionPool from attempting external network handshake during test imports
if "database" not in sys.modules:
    mock_db = MagicMock()
    mock_db.db_pool = MagicMock()
    sys.modules["database"] = mock_db

if "core.db" not in sys.modules:
    mock_core_db = MagicMock()
    mock_core_db.db_pool = MagicMock()
    sys.modules["core.db"] = mock_core_db
