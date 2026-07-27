"""
健康检查接口测试
"""
import pytest


class TestRootEndpoint:
    """GET / — API根信息"""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, async_client):
        response = await async_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "RAG API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert data["docs"] == "/docs"


class TestHealthCheck:
    """GET /api/v1/health — 健康检查"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, async_client):
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_has_required_fields(self, async_client):
        response = await async_client.get("/api/v1/health")
        data = response.json()
        assert "code" in data
        assert "message" in data
        assert "data" in data


class TestReadinessCheck:
    """GET /api/v1/health/ready — 就绪检查"""

    @pytest.mark.asyncio
    async def test_readiness_returns_200(self, async_client):
        response = await async_client.get("/api/v1/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["ready"] is True

    @pytest.mark.asyncio
    async def test_readiness_checks_structure(self, async_client):
        response = await async_client.get("/api/v1/health/ready")
        checks = response.json()["data"]["checks"]
        assert "api" in checks
        assert checks["api"] == "ok"
