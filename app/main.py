import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Local imports
from app.api.v1 import endpoints, upload_routes
from app.services.vector_store import VectorStoreService
from app.core.config import settings

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
    version="1.2.0",
    description=(
        "⚖️ A minimal semantic legal document pipeline with OCR-PDF ingestion, "
        "Qdrant vector search, and frontend interface."
    ),
)

# ============================================================
# Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Restrict this in production (e.g. ["https://yourdomain.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Frontend Setup (Templates + Static)
# ============================================================
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend(request: Request):
    """
    Serves the minimal upload + query HTML UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# Routers (API Endpoints)
# ============================================================
# ✅ Module 3 core endpoints: ingest-document & query-twin
app.include_router(endpoints.router, prefix="/api/v1/module3")

# ✅ Upload route (PDF + OCR ingestion)
app.include_router(upload_routes.router)

# ============================================================
# Startup Event: Qdrant Initialization
# ============================================================
@app.on_event("startup")
async def startup_event():
    """
    Runs when the API starts up. Verifies Qdrant connection and
    ensures the vector collection exists.
    """
    try:
        logger.info("[INIT] Checking Qdrant connection...")
        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME,
        )

        # Create collection asynchronously (non-blocking)
        await asyncio.to_thread(vector_store.create_collection)
        logger.info(f"[INIT] ✅ Qdrant collection '{settings.COLLECTION_NAME}' is ready.")
    except Exception as e:
        logger.error(f"[INIT] ❌ Failed to initialize Qdrant collection: {e}")

# ============================================================
# Health Check Endpoint
# ============================================================
@app.get("/health", tags=["System"])
async def health_check():
    """
    Simple API health endpoint to verify the backend is running.
    """
    return {"status": "ok", "message": "API is running successfully 🚀"}