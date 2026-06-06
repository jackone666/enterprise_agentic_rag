"""Application settings — all values read from environment variables.

When Docker services are unavailable, the system gracefully falls back
to in-memory mock implementations without crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "", required: bool = False) -> str:
    """Read an environment variable with an optional requirement check."""
    val = os.getenv(key, default)
    if required and not val:
        raise ValueError(
            f"Required environment variable {key} is not set. "
            f"Copy .env.example to .env and fill in values, then restart."
        )
    return val


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class PostgresSettings:
    host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("POSTGRES_PORT", "5432")))
    database: str = field(default_factory=lambda: _env("POSTGRES_DB", "enterprise_rag"))
    user: str = field(default_factory=lambda: _env("POSTGRES_USER", "rag_user"))
    password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", ""))

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class RedisSettings:
    host: str = field(default_factory=lambda: _env("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("REDIS_PORT", "6379")))
    password: str = field(default_factory=lambda: _env("REDIS_PASSWORD", ""))

    @property
    def connection_url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/0"
        return f"redis://{self.host}:{self.port}/0"


@dataclass
class MilvusSettings:
    host: str = field(default_factory=lambda: _env("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("MILVUS_PORT", "19530")))
    collection: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "enterprise_kb"))


@dataclass
class MinIOSettings:
    endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "localhost:9000"))
    access_key: str = field(default_factory=lambda: _env("MINIO_ACCESS_KEY", ""))
    secret_key: str = field(default_factory=lambda: _env("MINIO_SECRET_KEY", ""))
    bucket: str = field(default_factory=lambda: _env("MINIO_BUCKET", "enterprise-rag-docs"))
    secure: bool = field(default_factory=lambda: _env_bool("MINIO_SECURE", False))

    @property
    def s3_endpoint_url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.endpoint}"


@dataclass
class OTelSettings:
    endpoint: str = field(default_factory=lambda: _env("OTEL_EXPORTER_OTLP_ENDPOINT", ""))
    service_name: str = field(default_factory=lambda: _env("OTEL_SERVICE_NAME", "enterprise-agentic-rag"))
    enabled: bool = field(default_factory=lambda: _env_bool("OTEL_ENABLED", False))


@dataclass
class PrometheusSettings:
    enabled: bool = field(default_factory=lambda: _env_bool("PROMETHEUS_ENABLED", False))
    port: int = field(default_factory=lambda: int(_env("PROMETHEUS_PORT", "9090")))


@dataclass
class GrafanaSettings:
    port: int = field(default_factory=lambda: int(_env("GRAFANA_PORT", "3000")))
    admin_user: str = field(default_factory=lambda: _env("GRAFANA_ADMIN_USER", "admin"))
    admin_password: str = field(default_factory=lambda: _env("GRAFANA_ADMIN_PASSWORD", ""))


@dataclass
class Neo4jSettings:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "password"))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))


@dataclass
class GraphRAGSettings:
    enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_GRAPH_RAG", True))
    engine: str = field(default_factory=lambda: _env("GRAPH_ENGINE", "neo4j"))
    graph_depth: int = field(default_factory=lambda: int(_env("GRAPH_DEPTH", "2")))
    graph_top_k: int = field(default_factory=lambda: int(_env("GRAPH_TOP_K", "30")))


@dataclass
class RouterSettings:
    dynamic_router_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_DYNAMIC_ROUTER", True))
    default_mode: str = field(default_factory=lambda: _env("DEFAULT_RETRIEVAL_MODE", "parallel"))
    enable_graph_first: bool = field(default_factory=lambda: _env_bool("ENABLE_GRAPH_FIRST", True))
    enable_keyword_first: bool = field(default_factory=lambda: _env_bool("ENABLE_KEYWORD_FIRST", True))
    enable_vector_first: bool = field(default_factory=lambda: _env_bool("ENABLE_VECTOR_FIRST", True))
    enable_hybrid_fallback: bool = field(default_factory=lambda: _env_bool("ENABLE_HYBRID_FALLBACK", True))


@dataclass
class FusionSettings:
    graph_fusion_enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_GRAPH_FUSION", True))
    graph_weight_default: float = field(default_factory=lambda: float(_env("GRAPH_WEIGHT_DEFAULT", "0.2")))
    fusion_method: str = field(default_factory=lambda: _env("FUSION_METHOD", "rrf"))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    code_boost_enabled: bool = field(default_factory=lambda: _env_bool("CODE_BOOST_ENABLED", True))
    code_boost_factor: float = field(default_factory=lambda: float(_env("CODE_BOOST_FACTOR", "0.5")))


@dataclass
class DockerSettings:
    use_local_images: bool = field(default_factory=lambda: _env_bool("USE_LOCAL_DOCKER_IMAGES", True))
    force_pull: bool = field(default_factory=lambda: _env_bool("FORCE_PULL_IMAGES", False))


@dataclass
class OllamaSettings:
    """Ollama LLM provider settings."""

    base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model: str = field(
        default_factory=lambda: _env("OLLAMA_MODEL", "qwen3:1.7b")
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(_env("OLLAMA_TIMEOUT_SECONDS", "60"))
    )
    max_retries: int = field(
        default_factory=lambda: int(_env("OLLAMA_MAX_RETRIES", "2"))
    )


@dataclass
class AppSettings:
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "3")))
    retrieval_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_K", "5")))
    request_timeout_seconds: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT_SECONDS", "60")))
    max_graph_steps: int = field(default_factory=lambda: int(_env("MAX_GRAPH_STEPS", "18")))
    max_llm_calls_per_request: int = field(default_factory=lambda: int(_env("MAX_LLM_CALLS_PER_REQUEST", "6")))
    ollama: OllamaSettings = field(default_factory=OllamaSettings)


@dataclass
class RuntimeSettings:
    """Runtime mode switches.

    Production mode is intentionally stricter: silent in-memory fallbacks are
    useful for demos, but they hide data loss and degraded dependencies in prod.
    """

    environment: str = field(default_factory=lambda: _env("APP_ENV", _env("ENVIRONMENT", "development")).lower())
    allow_in_memory_fallback: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_IN_MEMORY_FALLBACK",
            _env("APP_ENV", _env("ENVIRONMENT", "development")).lower() not in ("prod", "production"),
        )
    )
    fail_open_rate_limiter: bool = field(
        default_factory=lambda: _env_bool(
            "RATE_LIMITER_FAIL_OPEN",
            _env("APP_ENV", _env("ENVIRONMENT", "development")).lower() not in ("prod", "production"),
        )
    )
    allow_local_code_execution: bool = field(
        default_factory=lambda: _env_bool(
            "ALLOW_LOCAL_CODE_EXECUTION",
            _env("APP_ENV", _env("ENVIRONMENT", "development")).lower() not in ("prod", "production"),
        )
    )

    @property
    def is_production(self) -> bool:
        return self.environment in ("prod", "production")


@dataclass
class AgenticRAGSettings:
    """Agentic RAG configuration (Phase 1 upgrade)."""
    enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_AGENTIC_RAG", True))
    max_iterations: int = field(default_factory=lambda: int(_env("AGENT_MAX_ITERATIONS", "5")))
    confidence_threshold: float = field(default_factory=lambda: float(_env("AGENT_CONFIDENCE_THRESHOLD", "0.3")))
    use_llm_deep_intent: bool = field(default_factory=lambda: _env_bool("AGENT_USE_LLM_DEEP_INTENT", True))
    use_llm_answer: bool = field(default_factory=lambda: _env_bool("AGENT_USE_LLM_ANSWER", True))
    enable_parallel_tools: bool = field(default_factory=lambda: _env_bool("AGENT_ENABLE_PARALLEL_TOOLS", True))
    fallback_to_original_rag: bool = field(default_factory=lambda: _env_bool("AGENT_FALLBACK_TO_ORIGINAL", True))


@dataclass
class ElasticsearchSettings:
    host: str = field(default_factory=lambda: _env("ES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("ES_PORT", "9200")))
    index: str = field(default_factory=lambda: _env("ES_INDEX", "enterprise_kb"))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class CodeAnalysisSettings:
    """AST-level code symbol extraction settings."""

    enable_ast_parsing: bool = field(default_factory=lambda: _env_bool("ENABLE_AST_PARSING", True))
    fallback_to_regex: bool = field(default_factory=lambda: _env_bool("AST_FALLBACK_TO_REGEX", True))


@dataclass
class CodeExecutionSettings:
    """Sandbox code execution settings."""

    sandbox_timeout_seconds: float = field(default_factory=lambda: float(_env("CODE_SANDBOX_TIMEOUT", "15.0")))
    sandbox_max_memory_mb: int = field(default_factory=lambda: int(_env("CODE_SANDBOX_MAX_MEMORY_MB", "256")))
    max_code_retries: int = field(default_factory=lambda: int(_env("MAX_CODE_RETRIES", "2")))
    allowed_languages: list[str] = field(default_factory=lambda: _env("CODE_ALLOWED_LANGUAGES", "javascript,typescript,python,bash,arkts").split(","))


@dataclass
class RerankerSettings:
    """Cross-Encoder reranker settings."""

    enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", True))
    model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "pdurugyan/qwen3-reranker-0.6b-q8_0:latest"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    timeout_seconds: float = field(default_factory=lambda: float(_env("RERANKER_TIMEOUT", "30.0")))
    batch_size: int = field(default_factory=lambda: int(_env("RERANKER_BATCH_SIZE", "20")))


@dataclass
class SemanticCacheSettings:
    """Semantic cache settings."""

    enabled: bool = field(default_factory=lambda: _env_bool("SEMANTIC_CACHE_ENABLED", True))
    ttl_seconds: int = field(default_factory=lambda: int(_env("SEMANTIC_CACHE_TTL", "3600")))
    similarity_threshold: float = field(default_factory=lambda: float(_env("SEMANTIC_CACHE_SIMILARITY", "0.92")))
    max_entries: int = field(default_factory=lambda: int(_env("SEMANTIC_CACHE_MAX_ENTRIES", "1000")))


@dataclass
class ExternalSearchSettings:
    """External knowledge source retrieval settings."""

    enabled: bool = field(
        default_factory=lambda: _env_bool(
            "EXTERNAL_SEARCH_ENABLED",
            _env("APP_ENV", _env("ENVIRONMENT", "development")).lower() not in ("prod", "production"),
        )
    )
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN", ""))
    github_repos: list[str] = field(default_factory=lambda: _env("GITHUB_REPOS", "").split(",") if _env("GITHUB_REPOS", "") else [])
    stackexchange_key: str = field(default_factory=lambda: _env("STACKEXCHANGE_KEY", ""))
    web_search_provider: str = field(default_factory=lambda: _env("WEB_SEARCH_PROVIDER", "none"))
    web_search_api_key: str = field(default_factory=lambda: _env("WEB_SEARCH_API_KEY", ""))
    external_timeout_seconds: float = field(default_factory=lambda: float(_env("EXTERNAL_SEARCH_TIMEOUT", "5.0")))
    external_weight_default: float = field(default_factory=lambda: float(_env("EXTERNAL_WEIGHT_DEFAULT", "0.15")))


@dataclass
class Settings:
    """Top-level settings aggregator — reads from env vars with safe defaults."""

    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    milvus: MilvusSettings = field(default_factory=MilvusSettings)
    minio: MinIOSettings = field(default_factory=MinIOSettings)
    otel: OTelSettings = field(default_factory=OTelSettings)
    prometheus: PrometheusSettings = field(default_factory=PrometheusSettings)
    grafana: GrafanaSettings = field(default_factory=GrafanaSettings)
    neo4j: Neo4jSettings = field(default_factory=Neo4jSettings)
    graph_rag: GraphRAGSettings = field(default_factory=GraphRAGSettings)
    router: RouterSettings = field(default_factory=RouterSettings)
    fusion: FusionSettings = field(default_factory=FusionSettings)
    docker: DockerSettings = field(default_factory=DockerSettings)
    app: AppSettings = field(default_factory=AppSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    elasticsearch: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)
    code_analysis: CodeAnalysisSettings = field(default_factory=CodeAnalysisSettings)
    code_execution: CodeExecutionSettings = field(default_factory=CodeExecutionSettings)
    external_search: ExternalSearchSettings = field(default_factory=ExternalSearchSettings)
    agentic_rag: AgenticRAGSettings = field(default_factory=AgenticRAGSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    semantic_cache: SemanticCacheSettings = field(default_factory=SemanticCacheSettings)

    @classmethod
    def from_env(cls) -> Settings:
        return cls()

    def check_services(self) -> dict[str, bool]:
        """Check which Docker services are reachable (best-effort)."""
        import socket

        checks: dict[str, tuple[str, int]] = {
            "postgres": (self.postgres.host, self.postgres.port),
            "redis": (self.redis.host, self.redis.port),
            "milvus": (self.milvus.host, self.milvus.port),
            "minio": (
                self.minio.endpoint.split(":")[0],
                int(self.minio.endpoint.split(":")[1]) if ":" in self.minio.endpoint else 9000,
            ),
            "elasticsearch": (self.elasticsearch.host, self.elasticsearch.port),
            "neo4j": self._neo4j_socket_target(),
        }
        results: dict[str, bool] = {}
        for name, (host, port) in checks.items():
            try:
                sock = socket.create_connection((host, port), timeout=1.0)
                sock.close()
                results[name] = True
            except (OSError, ConnectionRefusedError, TimeoutError):
                results[name] = False
        return results

    def _neo4j_socket_target(self) -> tuple[str, int]:
        uri_without_scheme = self.neo4j.uri.split("://", 1)[-1]
        host = uri_without_scheme.split(":", 1)[0]
        port = int(uri_without_scheme.rsplit(":", 1)[-1]) if ":" in uri_without_scheme else 7687
        return host, port


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_settings: Settings | None = None
_settings_env_snapshot: tuple[tuple[str, str | None], ...] | None = None

_SETTINGS_ENV_KEYS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_PASSWORD",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "MILVUS_COLLECTION",
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "MINIO_BUCKET",
    "MINIO_SECURE",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "OTEL_ENABLED",
    "PROMETHEUS_ENABLED",
    "PROMETHEUS_PORT",
    "GRAFANA_PORT",
    "GRAFANA_ADMIN_USER",
    "GRAFANA_ADMIN_PASSWORD",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "ENABLE_GRAPH_RAG",
    "GRAPH_ENGINE",
    "GRAPH_DEPTH",
    "GRAPH_TOP_K",
    "ENABLE_DYNAMIC_ROUTER",
    "DEFAULT_RETRIEVAL_MODE",
    "ENABLE_GRAPH_FIRST",
    "ENABLE_KEYWORD_FIRST",
    "ENABLE_VECTOR_FIRST",
    "ENABLE_HYBRID_FALLBACK",
    "ENABLE_GRAPH_FUSION",
    "GRAPH_WEIGHT_DEFAULT",
    "FUSION_METHOD",
    "RRF_K",
    "USE_LOCAL_DOCKER_IMAGES",
    "FORCE_PULL_IMAGES",
    "LOG_LEVEL",
    "MAX_RETRIES",
    "RETRIEVAL_K",
    "REQUEST_TIMEOUT_SECONDS",
    "MAX_GRAPH_STEPS",
    "MAX_LLM_CALLS_PER_REQUEST",
    "APP_ENV",
    "ENVIRONMENT",
    "ALLOW_IN_MEMORY_FALLBACK",
    "RATE_LIMITER_FAIL_OPEN",
    "ALLOW_LOCAL_CODE_EXECUTION",
    "ES_HOST",
    "ES_PORT",
    "ES_INDEX",
    "ENABLE_AST_PARSING",
    "AST_FALLBACK_TO_REGEX",
    "CODE_SANDBOX_TIMEOUT",
    "CODE_SANDBOX_MAX_MEMORY_MB",
    "MAX_CODE_RETRIES",
    "CODE_ALLOWED_LANGUAGES",
    "EXTERNAL_SEARCH_ENABLED",
    "GITHUB_TOKEN",
    "GITHUB_REPOS",
    "STACKEXCHANGE_KEY",
    "WEB_SEARCH_PROVIDER",
    "WEB_SEARCH_API_KEY",
    "EXTERNAL_SEARCH_TIMEOUT",
    "EXTERNAL_WEIGHT_DEFAULT",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT_SECONDS",
    "OLLAMA_MAX_RETRIES",
    "RERANKER_TIMEOUT",
    "RERANKER_BATCH_SIZE",
    "SEMANTIC_CACHE_ENABLED",
    "SEMANTIC_CACHE_TTL",
    "SEMANTIC_CACHE_SIMILARITY",
    "SEMANTIC_CACHE_MAX_ENTRIES",
    "CODE_BOOST_ENABLED",
    "CODE_BOOST_FACTOR",
    "ENABLE_AGENTIC_RAG",
    "AGENT_MAX_ITERATIONS",
    "AGENT_CONFIDENCE_THRESHOLD",
    "AGENT_USE_LLM_DEEP_INTENT",
    "AGENT_USE_LLM_ANSWER",
    "AGENT_ENABLE_PARALLEL_TOOLS",
    "AGENT_FALLBACK_TO_ORIGINAL",
)


def _current_env_snapshot() -> tuple[tuple[str, str | None], ...]:
    return tuple((key, os.getenv(key)) for key in _SETTINGS_ENV_KEYS)


def get_settings() -> Settings:
    global _settings, _settings_env_snapshot
    snapshot = _current_env_snapshot()
    if _settings is None or _settings_env_snapshot != snapshot:
        _settings = Settings.from_env()
        _settings_env_snapshot = snapshot
    return _settings
