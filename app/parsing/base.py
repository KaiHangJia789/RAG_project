"""
抽象解析器基类 + 通用工具方法
"""
import re
from abc import ABC, abstractmethod

import chardet

from app.parsing.models import ContentBlock, ParsedDocument


class BaseParser(ABC):
    """
    所有文件解析器的抽象基类。

    子类必须实现:
      - supported_types: 支持的文件扩展名元组  (".pdf",) / (".md",) / (".txt",)
      - parse():        核心解析方法
    """

    supported_types: tuple[str, ...]

    @abstractmethod
    async def parse(self, filename: str, content: bytes) -> ParsedDocument:
        """核心解析方法 — 子类必须实现"""
        ...

    # ═══════════════════════════════════════════════════════════
    # 通用工具方法（所有子类可用）
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def detect_and_decode(content: bytes) -> tuple[str, dict]:
        """
        自动检测编码并解码。

        Returns:
            (decoded_text, encoding_info)
            encoding_info: {"encoding": "utf-8", "confidence": 0.99, "has_bom": False}
        """
        result = chardet.detect(content)
        encoding = result.get("encoding") or "utf-8"
        confidence = round(result.get("confidence", 1.0), 4)

        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 回退: 用 errors="replace" 兜底
            text = content.decode(encoding, errors="replace")
            # 统计替换字符占比
            replacement_count = text.count("�")
            if replacement_count > len(text) * 0.05:
                text = content.decode("utf-8", errors="replace")

        return text, {
            "encoding": encoding,
            "confidence": confidence,
        }

    @staticmethod
    def is_blank_text(text: str, threshold: int = 10) -> bool:
        """判定文本是否为空（去除空白后不足 threshold 个字符）"""
        cleaned = re.sub(r"\s+", "", text)
        return len(cleaned) < threshold

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA256 去重标识"""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()
