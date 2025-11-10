from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import endpoints
from app.services.vector_store import VectorStoreService
from app.core.config import settings
import logging
import asyncio

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# FastAPI Initialization
# ============================================================
app = FastAPI(
    title="Legal Semantic Pipeline",
    version="1.0.0",
    description="Semantic chunking and Qdrant integration for legal documents."
)

# ============================================================
# Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# API Routers
# ============================================================
# ✅ Final endpoint paths:
# POST   /api/v1/module3/ingest-document
# GET    /api/v1/module3/query-twin
app.include_router(endpoints.router, prefix="/api/v1/module3")

# ============================================================
# Qdrant Collection Initialization (Startup Event)
# ============================================================
@app.on_event("startup")
async def startup_event():
    """
    Ensures Qdrant collection exists when the API starts up.
    """
    try:
        logger.info("[INIT] Checking Qdrant connection...")
        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME,
        )

        # Create collection asynchronously
        await asyncio.to_thread(vector_store.create_collection)
        logger.info(f"[INIT] ✅ Qdrant collection '{settings.COLLECTION_NAME}' is ready.")
    except Exception as e:
        logger.error(f"[INIT] ❌ Failed to initialize Qdrant collection: {e}")

# ============================================================
# Health Check Endpoint
# ============================================================
@app.get("/health")
async def health_check():
    """
    Simple API health endpoint.
    """
    return {"status": "ok", "message": "API is running successfully 🚀"}
