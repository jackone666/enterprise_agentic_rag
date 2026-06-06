"""Milvus vector store — REST API, no SDK dependency.

Gracefully falls back when Milvus is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "enterprise_knowledge_base")


class MilvusStore:
    """Milvus vector database via REST API with graceful fallback."""

    def __init__(
        self,
        collection_name: str | None = None,
        vector_size: int = 768,
        host: str | None = None,
        port: str | None = None,
    ) -> None:
        self._collection = collection_name or DEFAULT_COLLECTION
        self._vector_size = vector_size
        self._host = host or MILVUS_HOST
        self._port = port or MILVUS_PORT
        self._base = f"http://{self._host}:{self._port}/v2/vectordb"
        self._available: bool | None = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        try:
            # Milvus health check — use the status endpoint
            resp = httpx.get(f"http://{self._host}:9091/healthz", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            logger.warning("Milvus unavailable — falling back to keyword retriever")
            return False

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def ensure_collection(self, collection_name: str | None = None) -> bool:
        col = collection_name or self._collection
        if not self.available:
            return False
        try:
            # Check if exists
            r = httpx.get(f"{self._base}/collections", timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                names = [c.get("collectionName", "") for c in data.get("data", [])]
                if col in names:
                    return True

            # Create
            payload = {
                "collectionName": col,
                "dimension": self._vector_size,
                "metricType": "COSINE",
                "primaryField": "id",
                "vectorField": "vector",
            }
            r = httpx.post(
                f"{self._base}/collections/create",
                json=payload,
                timeout=10.0,
            )
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        collection_name: str | None = None,
    ) -> int:
        col = collection_name or self._collection
        if not self.available or not self.ensure_collection(col):
            return 0

        data = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            cid_str = chunk.get("chunk_id", str(uuid.uuid4()))
            # Milvus Int64 primary key: hash the string to int
            cid = abs(hash(cid_str)) % (2**63 - 1)
            data.append({
                "id": cid,
                "vector": vec,
                "source": chunk.get("source", ""),
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "tags": json.dumps(chunk.get("tags", [])),
            })

        try:
            r = httpx.post(
                f"{self._base}/entities/insert",
                json={"collectionName": col, "data": data},
                timeout=30.0,
            )
            if r.status_code == 200:
                result = r.json()
                return result.get("data", {}).get("insertCount", len(data))
        except Exception:
            pass
        return 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        collection_name: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = collection_name or self._collection
        if not self.available:
            return []

        payload: dict[str, Any] = {
            "collectionName": col,
            "data": [query_vector],
            "limit": top_k,
            "outputFields": ["source", "content", "title", "tags"],
        }
        if filters:
            payload["filter"] = " and ".join(
                f'{k} == "{v}"' for k, v in filters.items()
            )

        try:
            r = httpx.post(f"{self._base}/entities/search", json=payload, timeout=10.0)
            if r.status_code != 200:
                return []
            results = r.json().get("data", [])
            return [
                {
                    "content": item.get("content", ""),
                    "source": item.get("source", ""),
                    "score": item.get("distance", 0.0),
                    "chunk_id": item.get("id", ""),
                    "metadata": {
                        "title": item.get("title", ""),
                        "tags": item.get("tags", ""),
                    },
                }
                for item in results
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    def delete_by_source(self, source: str, collection_name: str | None = None) -> int:
        col = collection_name or self._collection
        if not self.available:
            return 0
        try:
            r = httpx.post(
                f"{self._base}/entities/delete",
                json={
                    "collectionName": col,
                    "filter": f'source == "{source}"',
                },
                timeout=10.0,
            )
            return 1 if r.status_code == 200 else 0
        except Exception:
            return 0
