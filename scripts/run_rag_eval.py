"""
RAG 端到端评测 — 对评测集跑完整链路并计算全部指标

用法:
    python scripts/run_rag_eval.py --limit 5              # 先跑 5 题试水（强烈建议）
    python scripts/run_rag_eval.py                        # 全量
    python scripts/run_rag_eval.py --strategy sentence --final-k 3 --min-score 0.4
    python scripts/run_rag_eval.py --local                # 不上报 LangFuse
    python scripts/run_rag_eval.py --resume               # 跳过已完成的题

产物:
    docs/week10/eval_{config}.md   — 指标报告
    data/eval/result_{config}.jsonl — 逐题原始结果（断点续跑的依据）

成本提示:
    单题约 5 次 LLM 调用（1 次生成 + 2 次忠实度 + 1 次上下文 + 1 次相关性）。
    50 题 × N 个配置，务必先 --limit 3 验证链路再跑全量。
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.embedding.dashscope_embedding import DashscopeEmbeddingClient  # noqa: E402
from app.evaluation.evaluator import QuestionMetrics, RAGEvaluator, summarize  # noqa: E402
from app.evaluation.judge import JudgeCache, LLMJudge  # noqa: E402
from app.rag.models import RetrievalConfig  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.services.index_service import IndexService  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,          # 默认安静，避免每题刷屏
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════
# 评测集
# ═══════════════════════════════════════════════════════════════

def load_eval_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not items:
        raise ValueError(f"评测集为空: {path}")
    return items


# ═══════════════════════════════════════════════════════════════
# 断点续跑
# ═══════════════════════════════════════════════════════════════

def load_done(path: Path) -> dict[str, dict]:
    """
    读取已完成的结果（resume 用）。

    **出错的结果不算完成** —— 否则一次网络抖动就会让那道题永远停在错误状态，
    重跑多少次都跳过它。这是「断点续跑」最容易被做错的地方。
    """
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    errored = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            qid = item["question_id"]
        except (json.JSONDecodeError, KeyError):
            continue
        if item.get("error"):
            errored += 1
            done.pop(qid, None)      # 后写入的成功结果覆盖先前的失败
            continue
        done[qid] = item
    if errored:
        logger.info("断点续跑：%d 条历史记录含错误，将重新评估", errored)
    return done


def append_result(path: Path, metrics: QuestionMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics.model_dump(), ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

async def run(args) -> None:
    items = load_eval_set(args.evalset)
    if args.limit:
        items = items[: args.limit]

    config = RetrievalConfig(
        strategy=args.strategy,
        final_k=args.final_k,
        min_score=args.min_score,
        retrieve_k=args.retrieve_k,
        prompt_name=args.prompt,
    )
    config_name = args.name or f"{args.strategy}_k{args.final_k}_s{args.min_score}"

    result_path = settings.EVAL_DIR / f"result_{config_name}.jsonl"
    report_path = Path("docs/week10") / f"eval_{config_name}.md"

    done = load_done(result_path) if args.resume else {}
    if done:
        logger.info("断点续跑：已完成 %d 题，跳过", len(done))

    # ── 装配链路 ──
    embedder = DashscopeEmbeddingClient()
    index = IndexService(embedder, strategy=args.strategy)
    index.load()
    if not index.ready:
        logger.error(
            "索引为空（策略=%s）。请先运行：python scripts/build_index.py --strategy %s",
            args.strategy, args.strategy,
        )
        sys.exit(1)
    logger.info("索引就绪: %d 条向量", index.size)

    pipeline = RagPipeline(index_service=index)
    evaluator = RAGEvaluator(
        judge=LLMJudge(cache=JudgeCache(settings.EVAL_DIR / "judge_cache.jsonl"))
    )

    logger.info("配置: %s", config.describe())
    logger.info("开始评估 %d 题 ...", len(items))

    metrics_list: list[QuestionMetrics] = []
    t0 = time.monotonic()

    for i, item in enumerate(items, start=1):
        qid = item["id"]

        if qid in done:
            metrics_list.append(QuestionMetrics(**done[qid]))
            continue

        try:
            result = await pipeline.answer(item["question"], config)
            m = await evaluator.evaluate_one(
                result,
                question_id=qid,
                answerable=item.get("answerable", True),
                reference_answer=item.get("ground_truth", ""),
                reference_docs=[
                    s["document"] for s in item.get("reference_sources", [])
                ],
            )
        except Exception as e:
            logger.error("[%d/%d] %s 失败: %s", i, len(items), qid, e)
            m = QuestionMetrics(
                question_id=qid, question=item["question"], error=str(e)
            )

        metrics_list.append(m)
        append_result(result_path, m)

        # 进度：显示关键信号便于早发现问题
        flag = "拒答" if m.refused else "回答"
        extra = f"F={m.faithfulness:.2f}" if m.faithfulness is not None else ""
        logger.info(
            "[%2d/%d] %s %-6s top1=%.3f %s",
            i, len(items), qid, flag, m.max_score or 0.0, extra,
        )

    elapsed = time.monotonic() - t0

    # ── 汇总 ──
    report = summarize(metrics_list, config=config.describe())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown(f"RAG 评估报告 — {config_name}"), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"评测完成：{report.total} 题，用时 {elapsed:.0f}s")
    print("=" * 70)
    print(f"  Faithfulness        {report.faithfulness:.4f}")
    print(f"  Answer Relevance    {report.answer_relevance:.4f}")
    print(f"  Context Precision   {report.context_precision:.4f}  (判官版)")
    print(f"  Context Precision   {report.context_precision_source:.4f}  (来源版，零成本)")
    print(f"  Hit Rate            {report.hit_rate:.4f}")
    print(f"  MRR                 {report.mrr:.4f}")
    print(f"  Citation Validity   {report.citation_validity:.4f}")
    print(f"  幻觉率              {report.hallucination_rate:.4f}")
    print(f"  拒答正确率          {report.refusal_accuracy:.4f}")
    print(f"  误拒率              {report.false_refusal_rate:.4f}")
    print(f"  平均延迟            {report.avg_latency_ms:.0f} ms")
    print("=" * 70)
    print(f"报告: {report_path}")
    print(f"明细: {result_path}")

    # ── LangFuse 上报 ──
    if not args.local:
        try:
            await upload_to_langfuse(metrics_list, config_name, config)
        except Exception as e:
            logger.error("LangFuse 上报失败（本地报告已生成）: %s", e)


async def upload_to_langfuse(
    metrics_list: list[QuestionMetrics], config_name: str, config: RetrievalConfig
) -> None:
    """
    把逐题指标上报到 LangFuse（每题一个 observation + 多个 score）。

    ⚠️ LangFuse 4.x 的 API 与 3.x 不同：
      - 没有 client.trace()（3.x 的写法），改用 start_as_current_observation()
      - span 上的 .score() 也没了，改用 client.score_current_trace()
    这里包在 start_as_current_observation 上下文里，score_current_trace
    才能把分数挂到当前 trace 上。
    """
    from app.llm.observability import flush, get_langfuse, langfuse_enabled

    if not langfuse_enabled():
        logger.info("LangFuse 未配置，跳过上报")
        return

    client = get_langfuse()
    uploaded = 0
    failed = 0

    for m in metrics_list:
        try:
            metadata = {
                "config": config_name,
                "strategy": config.strategy,
                "final_k": config.final_k,
                "min_score": config.min_score,
                "answerable": m.answerable,
                "refused": m.refused,
                "retrieved_count": m.retrieved_count,
                "max_score": m.max_score,
            }

            with client.start_as_current_observation(
                name=f"rag-eval:{m.question_id}",
                as_type="span",
                input={"question": m.question},
                output={"answer": m.answer, "refused": m.refused},
                metadata=metadata,
            ):
                scores = {
                    "faithfulness": m.faithfulness,
                    "answer_relevance": m.answer_relevance,
                    "context_precision": m.context_precision,
                    "context_precision_source": m.context_precision_source,
                    "hit_rate": m.hit_rate,
                    "mrr": m.mrr,
                    "citation_validity": m.citation_validity,
                }
                for name, value in scores.items():
                    if value is not None:
                        client.score_current_trace(name=name, value=float(value))

                client.score_current_trace(
                    name="refused", value=1.0 if m.refused else 0.0
                )
            uploaded += 1
        except Exception as e:
            failed += 1
            if failed <= 3:      # 只打前几条，避免刷屏
                logger.warning("上报 %s 失败: %s", m.question_id, e)

    flush()
    if failed:
        logger.warning("LangFuse 上报 %d 条成功，%d 条失败", uploaded, failed)
    else:
        logger.info("LangFuse 已上报 %d 条 trace", uploaded)


def main() -> None:
    p = argparse.ArgumentParser(description="RAG 端到端评测")
    p.add_argument("--evalset", type=Path, default=settings.EVAL_DIR / "rag_eval_set.json")
    p.add_argument("--strategy", default=settings.CHUNK_STRATEGY)
    p.add_argument("--final-k", type=int, default=settings.RAG_FINAL_K)
    p.add_argument("--retrieve-k", type=int, default=settings.RAG_RETRIEVE_K)
    p.add_argument("--min-score", type=float, default=settings.RAG_MIN_SCORE)
    p.add_argument("--prompt", default="rag_context_cited")
    p.add_argument("--name", default="", help="配置名（影响产物文件名）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题（试水用）")
    p.add_argument("--local", action="store_true", help="不上报 LangFuse")
    p.add_argument("--resume", action="store_true", help="跳过已完成的题")
    args = p.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
