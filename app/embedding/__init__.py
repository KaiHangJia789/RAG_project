"""Embedding 模块 — 阿里云百炼 qwen3.7-text-embedding-flash"""
from app.embedding.base import EmbeddingClient
from app.embedding.dashscope_embedding import DashscopeEmbeddingClient

__all__ = ["EmbeddingClient", "DashscopeEmbeddingClient"]
