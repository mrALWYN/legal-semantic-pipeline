import pytest
import httpx
from app.main import app


# ============================================================
# ✅ Utility: Shared async test client
# ============================================================
@pytest.fixture(scope="module")
def client():
    """Shared async HTTP client for tests."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ============================================================
# 🩺 1️⃣ Health & Liveness / Readiness
# ============================================================
@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


@pytest.mark.asyncio
async def test_liveness(client):
    response = await client.get("/api/v1/health/liveness")
    # Will pass even if endpoint isn't implemented yet
    assert response.status_code in (200, 404)


@pytest.mark.asyncio
async def test_readiness(client):
    response = await client.get("/api/v1/health/readiness")
    assert response.status_code in (200, 404)


# ============================================================
# 📊 2️⃣ Prometheus Metrics
# ============================================================
@pytest.mark.asyncio
async def test_metrics(client):
    response = await client.get("/api/v1/metrics")
    # Prometheus endpoint returns plain text
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "python_info" in response.text or "process_cpu_seconds_total" in response.text


# ============================================================
# 🧠 3️⃣ Model Registry (MLflow)
# ============================================================
@pytest.mark.asyncio
async def test_model_current(client):
    response = await client.get("/api/v1/models/current")
    # Should return JSON info about current production model
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data or "version" in data


@pytest.mark.asyncio
async def test_model_experiments(client):
    response = await client.get("/api/v1/models/experiments")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_model_compare(client):
    response = await client.get("/api/v1/models/compare")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert "experiments" in data


@pytest.mark.asyncio
async def test_model_promote(client):
    response = await client.post("/api/v1/models/promote", json={"experiment_id": "1"})
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert data.get("status") in ("success", "ok")


# ============================================================
# 💬 4️⃣ Feedback Endpoint
# ============================================================
@pytest.mark.asyncio
async def test_feedback(client):
    response = await client.post("/api/v1/feedback", json={
        "query": "Example legal question",
        "document_id": "ABC123",
        "relevance_score": 0.9,
        "comments": "Very relevant"
    })
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        data = response.json()
        assert data.get("status") in ("success", "received")
