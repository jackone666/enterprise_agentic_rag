"""Embedding provider — local model + mock fallback + extensible interface.

Config via env:
    EMBEDDING_PROVIDER=mock|local|openai|dashscope
    EMBEDDING_MODEL=mock-embedding|/path/to/model|text-embedding-3-small
    EMBEDDING_MODEL_PATH=/Users/zing/Desktop/models/embeding
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

VECTOR_SIZE = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))


# ===================================================================
# Abstract interface
# ===================================================================
class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...
    @property
    @abstractmethod
    def vector_size(self) -> int: ...
    @property
    @abstractmethod
    def provider_name(self) -> str: ...


# ===================================================================
# Mock provider — deterministic hash-based (tests, fallback)
# ===================================================================
class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, vector_size: int | None = None) -> None:
        self._vector_size = vector_size or VECTOR_SIZE

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    @property
    def vector_size(self) -> int: return self._vector_size

    @property
    def provider_name(self) -> str: return "mock"

    def _hash_to_vector(self, text: str) -> list[float]:
        vec: list[float] = []
        seed = text.encode("utf-8")
        while len(vec) < self._vector_size:
            h = hashlib.sha256(seed).digest()
            for b in h:
                if len(vec) >= self._vector_size:
                    break
                vec.append((b / 127.5) - 1.0)
            seed = h
        return vec


# ===================================================================
# Local embedding provider — sentence-transformers on local GPU/CPU
# ===================================================================
class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Loads a sentence-transformers model from a local directory.

    Config:
        EMBEDDING_MODEL_PATH  — path to model directory
        EMBEDDING_MODEL       — alias (optional)
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or os.getenv("EMBEDDING_MODEL_PATH", "")
        self._model = None
        self._vector_size = VECTOR_SIZE
        self._init_model()

    def _init_model(self) -> None:
        if not self._model_path or not os.path.isdir(self._model_path):
            logger.warning("Local embedding model path not found: %s — falling back to mock", self._model_path)
            self._model = False
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model from %s ...", self._model_path)
            self._model = SentenceTransformer(self._model_path)
            # Get actual embedding dimension
            test_vec = self._model.encode(["test"], show_progress_bar=False)
            self._vector_size = len(test_vec[0])
            logger.info("Local embedding model loaded — dim=%d", self._vector_size)
        except Exception as e:
            logger.error("Failed to load local embedding model: %s", e)
            self._model = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is False or self._model is None:
            return MockEmbeddingProvider(self._vector_size).embed(texts)
        vectors = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def vector_size(self) -> int: return self._vector_size

    @property
    def provider_name(self) -> str:
        return f"local/{os.path.basename(self._model_path) if self._model_path else 'unknown'}"


# ===================================================================
# OpenAI embedding provider
# ===================================================================
class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        self._base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        sizes = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        self._vector_size = sizes.get(model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._api_key or self._api_key.startswith("sk-xxx"):
            return MockEmbeddingProvider(self._vector_size).embed(texts)
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            resp = client.embeddings.create(model=self._model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            return MockEmbeddingProvider(self._vector_size).embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def vector_size(self) -> int: return self._vector_size

    @property
    def provider_name(self) -> str: return f"openai/{self._model}"


# ===================================================================
# Factory
# ===================================================================
def get_embedding_provider() -> BaseEmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    model = os.getenv("EMBEDDING_MODEL", "")

    if provider_name == "local":
        return LocalEmbeddingProvider(model_path=model or os.getenv("EMBEDDING_MODEL_PATH", ""))

    if provider_name == "openai":
        return OpenAIEmbeddingProvider(model=model or "text-embedding-3-small")

    return MockEmbeddingProvider()
