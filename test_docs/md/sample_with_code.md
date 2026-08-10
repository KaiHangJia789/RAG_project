# 编码检测工具

## 接口定义

```python
def detect_encoding(content: bytes) -> tuple[str, float]:
    """检测文件编码"""
    result = chardet.detect(content)
    return result["encoding"], result["confidence"]
```

## 支持的编码

- UTF-8
- GBK
- UTF-16

## 测试数据

| 编码 | 样例 | 检测置信度 |
|------|------|-----------|
| UTF-8 | Hello 世界 | 0.99 |
| GBK | 中文测试 | 0.85 |
| UTF-16 | 🚀 Emoji | 0.95 |
