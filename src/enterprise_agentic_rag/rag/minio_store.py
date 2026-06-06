"""MinIO document store — S3-compatible object storage for raw documents.

Reads credentials from env vars; gracefully falls back when MinIO is unreachable.
"""

from __future__ import annotations

import logging
from io import BytesIO

from minio import Minio
from minio.error import S3Error

from enterprise_agentic_rag.config.settings import get_settings

logger = logging.getLogger(__name__)


class MinIOStore:
    """Wrapper around MinIO Python SDK with graceful fallback."""

    def __init__(self, bucket: str | None = None) -> None:
        s = get_settings()
        self._endpoint = s.minio.endpoint
        self._access_key = s.minio.access_key
        self._secret_key = s.minio.secret_key
        self._secure = s.minio.secure
        self._bucket = bucket or s.minio.bucket
        self._client: Minio | None = None
        self._available: bool | None = None

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------
    @property
    def client(self) -> Minio | None:
        if self._client is None and self._available is not False:
            try:
                self._client = Minio(
                    self._endpoint,
                    access_key=self._access_key,
                    secret_key=self._secret_key,
                    secure=self._secure,
                )
                # Quick connectivity check
                self._client.list_buckets()
                self._available = True
            except Exception:
                self._available = False
                self._client = None
                logger.warning("MinIO unavailable — document store disabled")
        return self._client

    @property
    def available(self) -> bool:
        if self._available is None:
            _ = self.client  # trigger check
        return self._available or False

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------
    def ensure_bucket(self, bucket: str | None = None) -> bool:
        b = bucket or self._bucket
        c = self.client
        if c is None:
            return False
        try:
            if not c.bucket_exists(b):
                c.make_bucket(b)
            return True
        except S3Error:
            return False

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------
    def upload_document(
        self,
        file_path: str,
        bucket: str | None = None,
        object_name: str | None = None,
    ) -> str | None:
        """Upload a file to MinIO. Returns the object name or None on failure."""
        c = self.client
        if c is None:
            return None
        b = bucket or self._bucket
        if not self.ensure_bucket(b):
            return None
        obj = object_name or file_path.rsplit("/", 1)[-1]
        try:
            c.fput_object(b, obj, file_path)
            return obj
        except S3Error as exc:
            logger.error("MinIO upload failed: %s", exc)
            return None

    def upload_text(
        self,
        content: str,
        object_name: str,
        bucket: str | None = None,
    ) -> str | None:
        """Upload text content as an object."""
        c = self.client
        if c is None:
            return None
        b = bucket or self._bucket
        if not self.ensure_bucket(b):
            return None
        try:
            data = BytesIO(content.encode("utf-8"))
            c.put_object(b, object_name, data, length=len(content.encode("utf-8")))
            return object_name
        except S3Error as exc:
            logger.error("MinIO upload failed: %s", exc)
            return None

    def download_document(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> str | None:
        """Download an object's content as a UTF-8 string."""
        c = self.client
        if c is None:
            return None
        b = bucket or self._bucket
        try:
            resp = c.get_object(b, object_name)
            return resp.read().decode("utf-8")
        except S3Error:
            return None
        finally:
            if "resp" in dir():
                resp.close()
                resp.release_conn()

    def list_documents(self, bucket: str | None = None) -> list[str]:
        """List object names in a bucket."""
        c = self.client
        if c is None:
            return []
        b = bucket or self._bucket
        try:
            return [obj.object_name for obj in c.list_objects(b)]
        except S3Error:
            return []
