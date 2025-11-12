from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.services.metrics import registry

router = APIRouter(prefix="/api/v1")

@router.get("/metrics")
def metrics():
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
