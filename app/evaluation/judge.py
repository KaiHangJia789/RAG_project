"""
LLMJudge — 用大语言模型做评估判官

关键设计:
  1. **关思考 + 温度 0** —— 对推理模型而言，关思考比降温度对稳定性的贡献大一个
     数量级，且让调用快 5-10 倍。50 题 × N 配置的评估不关思考会跑到天亮。
  2. **批量判定** —— 断言核验、片段相关性判定都是一次调用处理全部条目。
     逐条调用的话调用量翻 N 倍，成本不可承受。
  3. **JSON 容错解析** —— 模型即使被要求只输出 JSON，仍可能包 markdown 围栏
     或前后带废话。统一剥壳 + 失败重试一次。
  4. **缓存** —— 判官结果按 (模型, prompt 版本, 输入) 哈希缓存，重跑近乎零成本。
"""
import hashlib
import json
import logging
import re
from pathlib import Path

from app.config import settings
from app.evaluation.prompts import PROMPT_VERSION, get_judge_prompt
from app.llm.client import LLMClient

logger = logging.getLogger("rag_api.eval.judge")


# ═══════════════════════════════════════════════════════════════
# JSON 解析容错
# ═══════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_llm_json(text: str) -> dict:
    """
    从模型输出里解析 JSON。

    依次尝试：直接解析 → 剥 markdown 围栏 → 截取首尾花括号之间的内容。
    全部失败抛 ValueError。
    """
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    raw = text.strip()

    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. 剥 ```json ... ``` 围栏
    m = _FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 截取首个 { 到末个 } 之间的内容
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    snippet = raw[:120].replace("\n", " ")
    raise ValueError(f"无法从模型输出中解析 JSON：{snippet}...")


# ═══════════════════════════════════════════════════════════════
# 判官缓存
# ═══════════════════════════════════════════════════════════════

class JudgeCache:
    """
    判官结果缓存（JSONL 追加存储）。

    缓存键包含 prompt 版本 —— 改了 prompt 历史缓存自然失效，
    不会用旧口径的结果污染新报告。
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else settings.EVAL_DIR / "judge_cache.jsonl"
        self._mem: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                self._mem[item["key"]] = item["value"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("判官缓存加载失败（将从头开始）: %s", e)
            self._mem = {}

    @staticmethod
    def make_key(model: str, prompt_name: str, payload: str) -> str:
        h = hashlib.sha256(
            f"{model}|{PROMPT_VERSION}|{prompt_name}|{payload}".encode()
        ).hexdigest()
        return h[:32]

    def get(self, key: str) -> dict | None:
        v = self._mem.get(key)
        if v is None:
            self._misses += 1
        else:
            self._hits += 1
        return v

    def put(self, key: str, value: dict) -> None:
        if key in self._mem:
            return
        self._mem[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "cached": len(self._mem)}


# ═══════════════════════════════════════════════════════════════
# 判官
# ═══════════════════════════════════════════════════════════════

class LLMJudge:
    """基于大语言模型的评估判官"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        cache: JudgeCache | None = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.cache = cache if cache is not None else JudgeCache()
        self.model = settings.JUDGE_MODEL or settings.DEEPSEEK_MODEL

    # ── 底层调用 ──────────────────────────────────────────

    async def _call_json(self, prompt_name: str, **kwargs) -> dict:
        """渲染模板 → 调用模型 → 解析 JSON（带缓存与一次重试）"""
        template = get_judge_prompt(prompt_name)
        system, user_message = template.render(**kwargs)

        key = JudgeCache.make_key(self.model, prompt_name, user_message)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self.llm.generate(
                    system=system,
                    user_message=user_message,
                    thinking_enabled=settings.JUDGE_THINKING_ENABLED,
                    temperature=settings.JUDGE_TEMPERATURE,
                    reasoning_effort=settings.JUDGE_REASONING_EFFORT,
                    json_mode=True,
                )
                data = parse_llm_json(resp.text or "")
                self.cache.put(key, data)
                return data
            except ValueError as e:
                # JSON 解析失败 → 重试一次（模型偶发输出格式漂移）
                last_err = e
                logger.warning(
                    "判官 JSON 解析失败（第 %d/2 次）: %s", attempt, e
                )
            except Exception as e:
                last_err = e
                logger.warning("判官调用失败（第 %d/2 次）: %s", attempt, e)

        raise RuntimeError(f"判官调用最终失败 [{prompt_name}]: {last_err}")

    # ── 四个判官能力 ─────────────────────────────────────

    async def decompose_claims(self, answer: str) -> tuple[list[str], bool]:
        """
        把答案拆解为原子断言。

        Returns:
            (断言列表, 是否为拒答)
        """
        data = await self._call_json("faithfulness_decompose", answer=answer)
        claims = data.get("claims") or []
        # 过滤空串与非字符串（模型偶尔会塞 null 或对象进来）
        claims = [str(c).strip() for c in claims if c and str(c).strip()]
        return claims, bool(data.get("is_refusal"))

    async def verify_claims(self, claims: list[str], context: str) -> list[bool]:
        """
        批量核验断言，返回逐条是否被支持。

        严格的长度对齐：判定数少于断言数时，缺失的按**不支持**处理
        （宁可低估忠实度），并记 warning —— 判定数对不齐是判官不稳定的
        头号表现，必须显式暴露而不是悄悄补齐。
        """
        if not claims:
            return []

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        data = await self._call_json(
            "faithfulness_verify", context=context, claims=numbered
        )
        verdicts = data.get("verdicts") or []

        # 按 id 对齐（模型可能乱序返回）
        by_id: dict[int, bool] = {}
        for v in verdicts:
            try:
                by_id[int(v.get("id"))] = bool(v.get("supported"))
            except (TypeError, ValueError):
                continue

        result = [by_id.get(i, False) for i in range(1, len(claims) + 1)]
        missing = [i for i in range(1, len(claims) + 1) if i not in by_id]
        if missing:
            logger.warning(
                "判官漏判 %d 条断言（%s），按「不支持」处理",
                len(missing), missing[:5],
            )
        return result

    async def reverse_questions(self, answer: str, n: int = 3) -> list[str]:
        """根据答案反推问题（Answer Relevance 的输入）"""
        data = await self._call_json("relevance_reverse_questions", answer=answer)
        qs = [str(q).strip() for q in (data.get("questions") or []) if q]
        return qs[:n]

    async def judge_context_relevance(
        self, question: str, chunks: list[str], reference_answer: str = ""
    ) -> list[bool]:
        """批量判定检索片段的相关性（Context Precision 的输入）"""
        if not chunks:
            return []

        numbered = "\n".join(
            f"[{i}] {t}" for i, t in enumerate(chunks, start=1)
        )
        data = await self._call_json(
            "context_relevance_judge",
            question=question,
            reference_answer=reference_answer or "（未提供）",
            chunks=numbered,
        )
        judgments = data.get("judgments") or []

        by_id: dict[int, bool] = {}
        for j in judgments:
            try:
                by_id[int(j.get("id"))] = bool(j.get("relevant"))
            except (TypeError, ValueError):
                continue

        result = [by_id.get(i, False) for i in range(1, len(chunks) + 1)]
        missing = [i for i in range(1, len(chunks) + 1) if i not in by_id]
        if missing:
            logger.warning(
                "判官漏判 %d 个片段（%s），按「不相关」处理", len(missing), missing[:5]
            )
        return result

    async def score_relevance_direct(self, question: str, answer: str) -> float:
        """判官直接给答案相关性打分（与反推问题法交叉验证用）"""
        data = await self._call_json(
            "relevance_direct_score", question=question, answer=answer
        )
        try:
            return max(0.0, min(1.0, float(data.get("score", 0.0))))
        except (TypeError, ValueError):
            logger.warning("判官返回的 score 非法: %r，按 0 处理", data.get("score"))
            return 0.0
