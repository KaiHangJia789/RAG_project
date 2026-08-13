"""
前端页面路由
GET / → 上传页面（static/index.html）
"""
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["前端"])

_STATIC_DIR = Path(__file__).resolve().parents[3] / "static"


@router.get(
    "/",
    response_class=HTMLResponse,
    summary="上传页面",
    description="返回文档上传与管理页面",
)
async def index() -> HTMLResponse:
    """返回前端上传页面"""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>前端页面未找到</h1>", status_code=404)
