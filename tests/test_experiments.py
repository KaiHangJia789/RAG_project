"""
Prompt 实验编排器测试（mock LLMClient）
"""
import pytest

from app.llm.client import LLMResponse, LLMUsage
from app.llm.experiments import PromptExperimentRunner


class FakeLLM:
    """mock LLMClient，返回固定响应"""

    def __init__(self):
        self.calls = []

    async def generate(self, system, user_message, **kwargs):
        self.calls.append({"system": system, "user": user_message, **kwargs})
        return LLMResponse(
            text=f"[{len(self.calls)}] 模拟输出",
            model="deepseek-v4-pro",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            latency_ms=120.0,
        )


class TestPromptExperimentRunner:
    @pytest.mark.asyncio
    async def test_run_creates_report(self):
        runner = PromptExperimentRunner(FakeLLM())
        report = await runner.run(
            "什么是RAG？",
            prompt_names=["baseline", "structured", "concise"],
        )
        assert len(report.results) == 3
        assert report.input_text == "什么是RAG？"

    @pytest.mark.asyncio
    async def test_run_calls_each_prompt(self):
        fake = FakeLLM()
        runner = PromptExperimentRunner(fake)
        await runner.run("测试", prompt_names=["baseline", "concise"])
        assert len(fake.calls) == 2

    @pytest.mark.asyncio
    async def test_comparison_table_generated(self):
        runner = PromptExperimentRunner(FakeLLM())
        report = await runner.run("测试", prompt_names=["baseline", "structured"])
        assert "Prompt 实验对比表" in report.comparison_table
        assert "| baseline |" in report.comparison_table
        assert "| structured |" in report.comparison_table

    @pytest.mark.asyncio
    async def test_metrics_computed(self):
        runner = PromptExperimentRunner(FakeLLM())
        report = await runner.run("测试", prompt_names=["baseline"])
        metrics = report.results[0].metrics
        assert "char_count" in metrics
        assert "word_count" in metrics
        assert "has_json" in metrics
        assert "token_total" in metrics

    @pytest.mark.asyncio
    async def test_unknown_prompt_skipped(self):
        runner = PromptExperimentRunner(FakeLLM())
        report = await runner.run("测试", prompt_names=["baseline", "nonexistent"])
        assert len(report.results) == 1  # 不存在的 prompt 被跳过
