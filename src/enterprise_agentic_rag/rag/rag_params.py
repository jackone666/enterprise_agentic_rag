"""RAG 调参中心 — 所有可调参数集中管理，修改无需改业务代码"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class RAGParams:
    # A. 文档预处理
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_chunk_chars: int = 2000

    # B. 混合检索
    retrieval_top_k: int = 5
    fusion_mode: str = "rrf"          # rrf | weighted_sum
    vector_weight: float = 0.6
    keyword_weight: float = 0.4
    rrf_k: int = 60

    # C. 重排序
    rerank_enabled: bool = False
    rerank_top_n: int = 3
    rerank_max_chars_per_doc: int = 500

    # D. 查询改写
    rewrite_enabled: bool = False
    hyde_enabled: bool = False
    rewrite_temperature: float = 0.3
    query_decomposition_enabled: bool = False

    # E. 大模型生成
    generation_temperature: float = 0.3
    system_prompt: str = (
        "你是一个企业级知识库问答助手。请基于参考文档回答用户问题。"
        "在回答中使用 [1]、[2] 标记引用来源。如果信息不足，请明确说明。"
    )

    # F. 评估
    judge_temperature: float = 0.0
    eval_score_threshold: float = 0.6
    eval_timeout_seconds: int = 10

    # G. 缓存与限流
    cache_ttl: int = 300
    near_duplicate_threshold: float = 0.95
    rate_limit_per_minute: int = 60

    # H. 嵌入模型
    embedding_batch_size: int = 32

    # I. 图谱增强检索
    enable_graph_rag: bool = True
    graph_depth: int = 2
    graph_top_k: int = 30
    graph_weight_default: float = 0.2
    enable_dynamic_router: bool = True
    enable_graph_fusion: bool = True
    graph_engine: str = "neo4j"

    @classmethod
    def from_env(cls) -> "RAGParams":
        return cls(
            retrieval_top_k=int(os.getenv("RETRIEVAL_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            enable_graph_rag=os.getenv("ENABLE_GRAPH_RAG", "true").lower() in ("1", "true", "yes", "on"),
            graph_depth=int(os.getenv("GRAPH_DEPTH", "2")),
            graph_top_k=int(os.getenv("GRAPH_TOP_K", "30")),
            graph_weight_default=float(os.getenv("GRAPH_WEIGHT_DEFAULT", "0.2")),
            enable_dynamic_router=os.getenv("ENABLE_DYNAMIC_ROUTER", "true").lower() in ("1", "true", "yes", "on"),
            enable_graph_fusion=os.getenv("ENABLE_GRAPH_FUSION", "true").lower() in ("1", "true", "yes", "on"),
            graph_engine=os.getenv("GRAPH_ENGINE", "neo4j"),
        )


# Global instance
_params: RAGParams | None = None

def get_rag_params() -> RAGParams:
    global _params
    if _params is None:
        _params = RAGParams.from_env()
    return _params
