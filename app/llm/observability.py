"""
LangFuse 可观测性封装

追踪机制:
  - langfuse.openai.AsyncOpenAI 包装器：自动追踪每次 chat.completions.create 调用，
    记录 input/output/model/usage（含 prompt_tokens/completion_tokens/total_tokens）
  - @observe() 装饰器（langfuse 4.x 顶层导出）：在实验编排层创建 trace，
    把多组 prompt 的 generation 归到一个 trace 下

关键点（DeepSeek + OpenAI 协议，非 Claude）:
  - 不能用 @observe 直接包 anthropic 调用，因为 SDK 换成了 openai
  - 用 langfuse.openai 包装器才能追踪到 token 消耗
"""
import os
import logging

logger = logging.getLogger("rag_api.llm.observability")


def langfuse_enabled() -> bool:
    """
    判断 LangFuse 是否配置。

    从 settings 读取（而非 os.getenv），因为导入 settings 会触发 config.py 的
    load_dotenv()，把 .env 写入 os.environ —— 这是 langfuse 库读取凭据的前提。
    """
    from app.config import settings
    return bool(
        settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
    )


def get_langfuse():
    """懒加载 LangFuse 客户端（未配置时返回 None）"""
    if not langfuse_enabled():
        return None
    from langfuse import Langfuse
    return Langfuse()


def flush() -> None:
    """刷新 LangFuse 缓冲（脚本结束时调用，确保数据上报）"""
    if not langfuse_enabled():
        return
    try:
        get_langfuse().flush()
        logger.info("LangFuse 数据已 flush")
    except Exception as e:
        logger.warning("LangFuse flush 失败: %s", e)


def observe(func=None, **kwargs):
    """
    LangFuse @observe 装饰器，无 key 时降级为直通。

    langfuse 4.x 从顶层 langfuse 导出 observe，签名支持两种形式：
      @observe         → func 直接传入
      @observe(...)    → 返回装饰器
    """
    if not langfuse_enabled():
        if func is not None:
            return func                       # @observe 形式
        return lambda f: f                    # @observe(...) 形式 → no-op 装饰器

    from langfuse import observe as _observe
    if func is not None:
        return _observe(func, **kwargs)
    return _observe(**kwargs)
