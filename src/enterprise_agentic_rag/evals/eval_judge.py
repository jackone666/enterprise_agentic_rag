"""LLM judge evaluation — 6-dimension automated assessment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    overall: float = 0.0
    passing: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class EvalJudge:
    """LLM-based evaluation judge for RAG quality assessment."""

    def __init__(self) -> None:
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            from enterprise_agentic_rag.llm.provider_factory import get_llm_provider
            self._provider = get_llm_provider()
        return self._provider

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: list[dict[str, Any]],
        ground_truth: str = "",
    ) -> EvalResult:
        """Run full 6-dimension evaluation."""
        # Fast-path: rule-based when no LLM
        if self.provider.provider_name == "mock":
            return self._rule_eval(query, answer, retrieved_docs)

        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                return loop.run_until_complete(self._llm_eval(query, answer, retrieved_docs, ground_truth))
        except Exception:
            pass
        return self._rule_eval(query, answer, retrieved_docs)

    def _rule_eval(self, query: str, answer: str, docs: list[dict[str, Any]]) -> EvalResult:
        """Heuristic evaluation when LLM unavailable."""
        has_answer = len(answer.strip()) >= 20
        has_citation = any(m in answer for m in ("[1]", "[2]", "参考来源", "引用"))
        has_docs = len(docs) > 0
        keywords_in_answer = sum(1 for kw in query.split() if kw in answer) / max(len(query.split()), 1)

        precision = 0.8 if has_answer and has_docs else 0.3
        recall = min(len(docs) / 5, 1.0) if has_docs else 0.0
        mrr = 1.0 if has_docs else 0.0
        hit = 1.0 if has_docs else 0.0
        faithfulness = 0.8 if has_citation else 0.3
        relevance = min(keywords_in_answer + 0.3, 1.0)
        overall = round((precision + recall + mrr + hit + faithfulness + relevance) / 6, 4)
        return EvalResult(
            precision=round(precision, 4), recall=round(recall, 4), mrr=round(mrr, 4),
            hit_rate=round(hit, 4), faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4), overall=overall,
            passing=overall >= 0.6,
            details={"method": "rule"},
        )

    async def _llm_eval(self, query: str, answer: str, docs: list, gt: str) -> EvalResult:
        prompt = (
            "你是一个专业的RAG系统评估裁判。请根据以下信息对回答质量进行评分（0-1）。\n\n"
            f"用户问题: {query}\n"
            f"生成回答: {answer[:1000]}\n"
            f"检索到的文档数: {len(docs)}\n"
            f"参考标准答案（如有）: {gt[:500]}\n\n"
            "请返回JSON格式，包含以下字段（每个0-1之间的浮点数）:\n"
            "- precision: 回答中正确信息的比例\n"
            "- recall: 回答覆盖了问题多少关键点\n"
            "- faithfulness: 回答是否忠实于参考文档\n"
            "- relevance: 回答是否与问题相关\n"
            '示例: {"precision":0.85,"recall":0.7,"faithfulness":0.9,"relevance":0.95}'
        )
        resp = await self.provider.generate(prompt, temperature=0.0, max_tokens=256)
        if not resp.success:
            return self._rule_eval(query, answer, docs)
        try:
            data = json.loads(resp.content)
            p, r, f, rel = data.get("precision", 0), data.get("recall", 0), data.get("faithfulness", 0), data.get("relevance", 0)
            overall = round((p + r + f + rel) / 4, 4)
            return EvalResult(
                precision=round(p, 4), recall=round(r, 4), mrr=1.0 if docs else 0.0,
                hit_rate=1.0 if docs else 0.0, faithfulness=round(f, 4),
                relevance=round(rel, 4), overall=overall,
                passing=overall >= 0.6,
                details={"method": "llm", "model": self.provider.model_name},
            )
        except Exception:
            return self._rule_eval(query, answer, docs)
