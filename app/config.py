"""
应用配置模块
使用 Pydantic Settings 管理所有环境变量和应用配置
"""
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用全局配置，自动从 .env 文件和环境变量加载"""
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- 应用 ---
    APP_NAME: str = "RAG API"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "RAG项目 — 第4周 FastAPI基础：文档上传与管理API"
    DEBUG: bool = True

    # --- 文件上传 ---
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set[str] = {
        ".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".xml"
    }

    # --- 分页 ---
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100


settings = Settings()
