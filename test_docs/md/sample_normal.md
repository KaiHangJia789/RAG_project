# RAG 项目技术文档

## 项目概述

RAG（检索增强生成）是结合信息检索与 LLM 生成的技术。

## 核心组件

1. **文档解析**：支持 PDF、Markdown、TXT 三种格式
2. **Chunk 切分**：按语义边界将长文档切分为可控大小的块
3. **向量检索**：使用 Embedding 模型将文本转为向量
4. **生成回答**：结合检索到的上下文生成回答

## 快速开始

```python
from app.parsing.parser_registry import get_default_registry

registry = get_default_registry()
parser = registry.get(".md")
result = await parser.parse("doc.md", content)
```

## 注意事项

- 扫描件 PDF 需要 OCR 处理
- 编码检测依赖 chardet 库
- 长文本块需要切分以避免检索精度下降
