# 使用说明

## 第一步：上传文档

通过 API 上传文档文件：

```
POST /api/v1/upload
Content-Type: multipart/form-data
```

上传成功后系统会自动解析文档内容。

## 第二步：查看文档列表

```
GET /api/v1/documents
```

返回所有已上传的文档，支持关键词搜索和分页。

## 第三步：获取文档详情

```
GET /api/v1/documents/{id}
```

返回文档的完整信息，包括解析状态和元数据。

## 常见问题

### Q: 支持哪些文件格式？

目前支持 PDF、Markdown 和纯文本文件。

### Q: 上传的文件大小限制是多少？

单个文件最大 50MB。
