"""RAG 效果评估 — 自实现 LLM-as-Judge（不依赖 ragas 库）"""
from app.evaluation.evaluator import EvalReport, QuestionMetrics, RAGEvaluator, summarize
from app.evaluation.judge import JudgeCache, LLMJudge, parse_llm_json
from app.evaluation.metrics import (
    MetricResult,
    citation_validity,
    context_precision_ranked,
    faithfulness,
    false_refusal_rate,
    hallucination_rate,
    hit_rate_at_k,
    mrr,
    refusal_accuracy,
    source_precision,
)
from app.evaluation.prompts import JUDGE_PROMPTS, PROMPT_VERSION, get_judge_prompt

__all__ = [
    "EvalReport",
    "JUDGE_PROMPTS",
    "JudgeCache",
    "LLMJudge",
    "MetricResult",
    "PROMPT_VERSION",
    "QuestionMetrics",
    "RAGEvaluator",
    "citation_validity",
    "context_precision_ranked",
    "faithfulness",
    "false_refusal_rate",
    "get_judge_prompt",
    "hallucination_rate",
    "hit_rate_at_k",
    "mrr",
    "parse_llm_json",
    "refusal_accuracy",
    "source_precision",
    "summarize",
]
