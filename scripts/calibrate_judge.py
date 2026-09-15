"""
判官标定 — 跑全量评测前的**准入门槛**

用法:
    python scripts/calibrate_judge.py

原理:
    用 5 组人工构造的金标准样本（完全忠实 / 部分忠实 / 完全幻觉 / 拒答 /
    数字主体不符）测试判官，断言它给出预期判定。

    不通过就不要跑全量评测 —— 判官不可靠时所有指标都是噪声，
    而且你无法分辨「配置变好了」还是「判官这次抖动了」。
    这一步花 30 秒，能省下几小时的无效评测。
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台默认 GBK，输出 ✓/✗/emoji 会 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.evaluation.judge import JudgeCache, LLMJudge  # noqa: E402
from app.evaluation.metrics import faithfulness  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s | %(message)s")

# 判官核验断言的依据
CONTEXT = """[1] RAG 的拒答判断要放在生成之前。设置相似度阈值，最高相似度低于阈值时
直接返回拒答，不调用大语言模型。这样做响应快（毫秒级而非数秒级），成本低。
[2] FAISS 的 add_with_ids 对重复 id 不会报错，只是追加一条新记录。
结果是两条不同向量共享同一个 id，检索时返回重复 id 挤占结果名额。"""

# (名称, 答案, 期望忠实度, 期望是否判为拒答, 容差)
CASES = [
    (
        "完全忠实",
        "拒答判断应放在生成之前[1]。这样响应更快也更省钱[1]。",
        1.0, False, 0.26,
    ),
    (
        "部分忠实",
        "拒答判断应放在生成之前[1]。FAISS 对重复 id 会报错[2]。",
        0.5, False, 0.26,
    ),
    (
        "完全幻觉",
        "Redis 的持久化有两种方式，分别是 RDB 和 AOF 快照。",
        0.0, False, 0.26,
    ),
    (
        "拒答",
        "根据提供的资料无法回答该问题。",
        1.0, True, 0.26,
    ),
    (
        "数字/主体不符",
        "FAISS 对重复 id 会直接抛出异常[2]，必须捕获处理。",
        0.0, False, 0.26,
    ),
]


async def main() -> None:
    cache = JudgeCache(settings.EVAL_DIR / "judge_cache.jsonl")
    judge = LLMJudge(cache=cache)

    print("=" * 76)
    print("判官标定 — 5 组金标准样本")
    print("=" * 76)

    passed = 0
    failures: list[str] = []

    for name, answer, expected_f, expected_refusal, tol in CASES:
        try:
            claims, is_refusal = await judge.decompose_claims(answer)

            if is_refusal:
                verdicts: list[bool] = []
                score = 1.0
            else:
                verdicts = await judge.verify_claims(claims, CONTEXT)
                score = faithfulness(sum(verdicts), len(verdicts)).value

            ok_score = abs(score - expected_f) <= tol
            ok_refusal = is_refusal == expected_refusal
            ok = ok_score and ok_refusal

            if ok:
                passed += 1
            else:
                reasons = []
                if not ok_score:
                    reasons.append(f"忠实度 {score:.2f} 期望 {expected_f}")
                if not ok_refusal:
                    reasons.append(f"拒答判定 {is_refusal} 期望 {expected_refusal}")
                failures.append(f"{name}: {'; '.join(reasons)}")

            print(
                f"{'✓' if ok else '✗'} {name:<14} "
                f"faithfulness={score:.2f} (期望≈{expected_f})  "
                f"断言{len(claims)}条 支持{sum(verdicts)}条 拒答={is_refusal}"
            )
        except Exception as e:
            failures.append(f"{name}: 调用失败 {e}")
            print(f"✗ {name:<14} 调用失败: {e}")

    print("=" * 76)
    print(f"通过 {passed}/{len(CASES)}")
    print(f"判官缓存: {cache.stats}")

    if failed := len(CASES) - passed:
        print()
        print(f"⚠️  有 {failed} 组未通过，判官不可靠。")
        print("   处理建议：")
        print("   1. 检查 app/evaluation/prompts.py 的判官 prompt 是否给了足够反例")
        print("   2. 确认 JUDGE_THINKING_ENABLED=False 且 JUDGE_TEMPERATURE=0")
        print("   3. 删掉 data/eval/judge_cache.jsonl 后重跑（排除旧缓存干扰）")
        print("   4. 以上都无效则更换判官模型（改 .env 的 JUDGE_MODEL）")
        print()
        print("   **在标定通过前不要跑全量评测** —— 指标会是噪声。")
        sys.exit(1)

    print("✅ 判官标定通过，可以进入全量评测")
    print()
    print("下一步：")
    print("   python scripts/calibrate_threshold.py --strategy sentence")
    print("   python scripts/run_rag_eval.py --strategy sentence --limit 3")


if __name__ == "__main__":
    asyncio.run(main())
