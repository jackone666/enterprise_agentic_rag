"""Evaluation system and data flywheel.

Components:
- Dataset: Load / save eval cases and failed cases
- RAGEval: hit@k, recall@k, MRR, avg_retrieval_score
- AnswerEval: citation_present, groundedness, relevance, refusal
- RegressionEval: End-to-end pipeline pass/fail runner
- OnlineFeedback: thumbs_up/down capture + auto case mining
"""

from enterprise_agentic_rag.evals.dataset import EvalDataset
from enterprise_agentic_rag.evals.online_feedback import FeedbackHandler

__all__ = ["EvalDataset", "FeedbackHandler"]
