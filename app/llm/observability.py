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


_client_cache: object | None = None


def get_langfuse():
    """
    懒加载 LangFuse 客户端（未配置时返回 None）。

    做进程内缓存：Langfuse() 每次构造都会新建 OpenTelemetry exporter 和后台线程，
    在评估脚本里每题调一次会迅速耗尽资源。
    """
    global _client_cache
    if not langfuse_enabled():
        return None
    if _client_cache is None:
        from langfuse import Langfuse
        _client_cache = Langfuse()
    return _client_cache


def _in_active_span() -> bool:
    """
    当前是否处于一个有效的 trace/span 上下文中。

    先做这层检查再调 LangFuse 的 get_current_trace_id()：后者在没有活跃 span 时
    会通过 OTel 打一条 "Context error: No active span" 的 ERROR 日志 ——
    对"未包在 @observe 里的普通调用"来说这是预期情况，不该刷错误日志。
    """
    try:
        from opentelemetry import trace as otel_trace
        span = otel_trace.get_current_span()
        return span is not None and span.get_span_context().is_valid
    except Exception:
        return False


def current_trace_id() -> str | None:
    """
    取当前上下文的 trace id（供 API 返回给前端做"查看追踪"链接）。

    @observe() 装饰的函数体内调用才会拿到值；未配置或不在 trace 上下文中返回 None。
    """
    client = get_langfuse()
    if client is None or not _in_active_span():
        return None
    try:
        return client.get_current_trace_id()
    except Exception as e:
        logger.debug("获取 trace_id 失败: %s", e)
        return None


def trace_url(trace_id: str | None = None) -> str | None:
    """取 LangFuse Dashboard 上该 trace 的可点击链接"""
    client = get_langfuse()
    if client is None:
        return None
    if trace_id is None and not _in_active_span():
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception as e:
        logger.debug("获取 trace_url 失败: %s", e)
        return None


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
