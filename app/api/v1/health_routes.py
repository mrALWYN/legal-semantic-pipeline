# app/api/v1/health_routes.py
import logging
from fastapi import APIRouter
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter(prefix="/api/v1/health", tags=["System"])
logger = logging.getLogger(__name__)

@router.get("")
def health():
    return {"status": "ok", "version": settings.API_VERSION, "title": settings.API_TITLE}

@router.get("/readiness")
def readiness():
    try:
        vs = VectorStoreService(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, collection_name=settings.COLLECTION_NAME)
        # quick ping by listing collections (sync)
        collections = vs.client.get_collections()
        return {"ready": True, "qdrant_collections": [c.name for c in collections.collections]}
    except Exception as e:
        logger.exception("Readiness check failed")
        return {"ready": False, "error": str(e)}

@router.get("/liveness")
def liveness():
    return {"alive": True}
