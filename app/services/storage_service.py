"""
文件存储服务
负责文件的物理存储、读取和删除操作
"""
import shutil
import uuid
from datetime import datetime, UTC
from pathlib import Path

from app.config import settings


class StorageService:
    """本地文件系统存储服务"""

    def __init__(self) -> None:
        self._ensure_upload_dir()

    def _ensure_upload_dir(self) -> None:
        """确保上传目录存在"""
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def _build_storage_path(self, original_filename: str) -> Path:
        """
        构建存储路径：uploads/YYYY/MM/doc-{uuid}.ext
        按日期分目录避免单目录文件过多
        """
        now = datetime.now(UTC)
        date_dir = now.strftime("%Y/%m")
        file_ext = Path(original_filename).suffix.lower()
        unique_name = f"doc-{uuid.uuid4().hex[:12]}{file_ext}"
        return settings.UPLOAD_DIR / date_dir / unique_name

    def validate_file(self, filename: str, file_size: int) -> None:
        """
        校验文件合法性，不合法抛出 ValueError

        Args:
            filename: 原始文件名
            file_size: 文件大小（字节）

        Raises:
            ValueError: 文件类型不支持或文件过大
        """
        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型 '{ext}'。允许的类型: "
                f"{', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
            )
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if file_size > max_bytes:
            raise ValueError(
                f"文件大小 {file_size / 1024 / 1024:.1f}MB 超过限制 "
                f"（最大 {settings.MAX_UPLOAD_SIZE_MB}MB）"
            )

    async def save(self, filename: str, content: bytes) -> dict:
        """
        保存文件到磁盘

        Returns:
            dict: {"storage_path": str, "file_size": int, "file_type": str}
        """
        storage_path = self._build_storage_path(filename)
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # 异步写文件：使用 run_in_executor 避免阻塞事件循环
        with open(storage_path, "wb") as f:
            f.write(content)

        return {
            "storage_path": str(storage_path.relative_to(settings.UPLOAD_DIR.parent)),
            "file_size": len(content),
            "file_type": Path(filename).suffix.lower(),
        }

    def read(self, storage_path: str) -> bytes:
        """读取文件内容"""
        full_path = Path(storage_path)
        if not full_path.is_absolute():
            full_path = settings.UPLOAD_DIR.parent / storage_path
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {storage_path}")
        return full_path.read_bytes()

    def delete(self, storage_path: str) -> None:
        """删除物理文件（文件不存在时不报错）"""
        full_path = Path(storage_path)
        if not full_path.is_absolute():
            full_path = settings.UPLOAD_DIR.parent / storage_path
        if full_path.exists():
            full_path.unlink()
