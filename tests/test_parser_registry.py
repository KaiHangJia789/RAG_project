"""
解析器注册表测试
"""
import pytest
from app.parsing.parser_registry import ParserRegistry, get_default_registry
from app.parsing.pdf_parser import PdfParser
from app.parsing.markdown_parser import MarkdownParser
from app.parsing.txt_parser import TxtParser


class TestParserRegistry:
    def test_register_and_get(self):
        registry = ParserRegistry()
        registry.register(PdfParser())
        parser = registry.get(".pdf")
        assert isinstance(parser, PdfParser)

    def test_case_insensitive(self):
        registry = ParserRegistry()
        registry.register(PdfParser())
        assert registry.get(".PDF") is not None
        assert registry.get(".Pdf") is not None

    def test_get_unsupported_type(self):
        registry = ParserRegistry()
        registry.register(TxtParser())
        assert registry.get(".unknown") is None
        assert registry.get(".exe") is None

    def test_supported_types(self):
        registry = ParserRegistry()
        registry.register(PdfParser())
        registry.register(MarkdownParser())
        registry.register(TxtParser())
        types = registry.supported_types
        assert ".pdf" in types
        assert ".md" in types
        assert ".txt" in types

    def test_get_default_registry(self):
        registry = get_default_registry()
        assert registry.get(".pdf") is not None
        assert registry.get(".md") is not None
        assert registry.get(".txt") is not None

    def test_default_registry_is_singleton(self):
        r1 = get_default_registry()
        r2 = get_default_registry()
        assert r1 is r2
