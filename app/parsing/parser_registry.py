"""
ParserRegistry — 解析器注册表

根据文件扩展名路由到对应的解析器实例。
扩展新格式时只需 register(new_parser)，无需改其他代码。
"""
import logging

from app.parsing.base import BaseParser

logger = logging.getLogger("rag_api.parsing.registry")


class ParserRegistry:
    """解析器注册表"""

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        """注册一个解析器（提取 supported_types 建立路由）"""
        for ext in parser.supported_types:
            ext_lower = ext.lower()
            self._parsers[ext_lower] = parser
            logger.info("ParserRegistry: registered %s → %s", ext_lower, type(parser).__name__)

    def get(self, file_type: str) -> BaseParser | None:
        """根据扩展名获取解析器，不存在返回 None"""
        return self._parsers.get(file_type.lower())

    @property
    def supported_types(self) -> list[str]:
        """列出所有已注册的文件类型"""
        return sorted(self._parsers.keys())


# ── 全局单例 ──

_default_registry: ParserRegistry | None = None


def get_default_registry() -> ParserRegistry:
    """获取默认的解析器注册表（懒加载单例）"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ParserRegistry()
        # 延迟导入避免循环依赖
        from app.parsing.pdf_parser import PdfParser
        from app.parsing.markdown_parser import MarkdownParser
        from app.parsing.txt_parser import TxtParser

        _default_registry.register(PdfParser())
        _default_registry.register(MarkdownParser())
        _default_registry.register(TxtParser())
    return _default_registry
