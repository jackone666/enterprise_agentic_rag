"""Production API adapters with automatic mock fallback.

Each adapter reads its base URL from env vars and gracefully
falls back to the current mock implementation when unavailable.
"""

from enterprise_agentic_rag.tools.adapters.http_client import ProductionHTTPClient

__all__ = ["ProductionHTTPClient"]
