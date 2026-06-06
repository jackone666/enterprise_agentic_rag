"""Tenant context middleware — extracts tenant_id from JWT or X-Tenant-ID header."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TenantContext:
    tenant_id: str = "default"
    user_id: str = "anonymous"
    role: str = "basic"
    permissions: list[str] = field(default_factory=list)

    @property
    def metadata_filter(self) -> str:
        """Returns a filter expression for tenant-aware retrieval."""
        return f'tenant_id == "{self.tenant_id}"'


def extract_tenant_context(
    headers: dict[str, str] | None = None,
    jwt_payload: dict[str, Any] | None = None,
) -> TenantContext:
    """Extract tenant context from JWT payload or HTTP headers.

    Priority: JWT > X-Tenant-ID header > default
    """
    ctx = TenantContext()

    # 1. Try JWT payload
    if jwt_payload:
        ctx.tenant_id = jwt_payload.get("tenant_id", ctx.tenant_id)
        ctx.user_id = jwt_payload.get("sub", jwt_payload.get("user_id", ctx.user_id))
        ctx.role = jwt_payload.get("role", ctx.role)
        ctx.permissions = jwt_payload.get("permissions", [])

    # 2. Fallback to headers (dev/test mode)
    if headers:
        ctx.tenant_id = headers.get("x-tenant-id", headers.get("X-Tenant-ID", ctx.tenant_id))
        ctx.role = headers.get("x-role", headers.get("X-Role", ctx.role))
        ctx.user_id = headers.get("x-user-id", headers.get("X-User-ID", ctx.user_id))

    return ctx
