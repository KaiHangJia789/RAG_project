"""
解析模块专用异常
"""

class ParsingError(Exception):
    """解析模块基异常"""


class UnsupportedFileTypeError(ParsingError):
    """不支持的文件类型"""

    def __init__(self, file_type: str) -> None:
        self.file_type = file_type
        super().__init__(f"不支持的文件类型: '{file_type}'")


class ParseFailureError(ParsingError):
    """解析过程失败"""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(f"解析失败 '{filename}': {reason}")


class EncodingDetectionError(ParsingError):
    """编码检测失败"""

    def __init__(self, detail: str) -> None:
        super().__init__(f"编码检测失败: {detail}")
