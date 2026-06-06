"""Production HTTP client — unified request layer with retry, timeout, tracing.

Never lets exceptions propagate — always returns a structured result.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class HTTPResult:
    """Uniform result from every HTTP call."""

    success: bool
    status_code: int = 0
    data: Any = None
    error: str = ""
    latency_ms: float = 0.0
    request_id: str = ""
    retries: int = 0


class ProductionHTTPClient:
    """Async HTTP client with retry, timeout, and tracing."""

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> HTTPResult:
        return await self._request("GET", path, params=params, trace_id=trace_id)

    async def post(
        self,
        path: str,
        json_body: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> HTTPResult:
        return await self._request("POST", path, json_body=json_body, trace_id=trace_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        trace_id: str = "",
    ) -> HTTPResult:
        req_id = str(uuid.uuid4())[:8]
        url = f"{self.base_url}{path}" if self.base_url else path
        headers = {**self._headers, "X-Request-ID": req_id}
        if trace_id:
            headers["X-Trace-ID"] = trace_id

        last_error = ""
        t0 = time.monotonic()

        for attempt in range(1, self.max_retries + 2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        resp = await client.get(url, params=params, headers=headers)
                    else:
                        resp = await client.post(url, json=json_body, headers=headers)

                    latency = (time.monotonic() - t0) * 1000

                    if resp.status_code < 400:
                        try:
                            body = resp.json()
                        except Exception:
                            body = resp.text
                        return HTTPResult(
                            success=True,
                            status_code=resp.status_code,
                            data=body,
                            latency_ms=round(latency, 2),
                            request_id=req_id,
                            retries=attempt - 1,
                        )
                    elif resp.status_code < 500:
                        # 4xx — don't retry
                        return HTTPResult(
                            success=False,
                            status_code=resp.status_code,
                            error=f"Client error: {resp.status_code}",
                            latency_ms=round(latency, 2),
                            request_id=req_id,
                            retries=attempt - 1,
                        )
                    else:
                        last_error = f"Server error: {resp.status_code}"
            except httpx.TimeoutException:
                last_error = f"Timeout after {self.timeout}s"
            except httpx.ConnectError:
                last_error = "Connection refused"
            except Exception as exc:
                last_error = str(exc)

            if attempt <= self.max_retries:
                await asyncio.sleep(self.backoff * attempt)

        latency = (time.monotonic() - t0) * 1000
        return HTTPResult(
            success=False,
            error=f"请求失败 (已重试 {self.max_retries} 次): {last_error}",
            latency_ms=round(latency, 2),
            request_id=req_id,
            retries=self.max_retries,
        )
