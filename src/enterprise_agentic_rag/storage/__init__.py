"""PostgreSQL storage layer with automatic fallback to in-memory.

When PostgreSQL is unreachable, all operations gracefully fall back
to the existing mock/in-memory implementations.
"""

from enterprise_agentic_rag.storage.database import DatabaseManager, get_db_manager
from enterprise_agentic_rag.storage.repositories import Repository

__all__ = ["DatabaseManager", "get_db_manager", "Repository"]
