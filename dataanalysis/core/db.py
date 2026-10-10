"""Read-only PostgreSQL pool shared by clinical-history readers."""
from database import db_pool

__all__ = ["db_pool"]
