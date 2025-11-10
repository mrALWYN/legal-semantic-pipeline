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
    version="1.1.0",
    description="Semantic chunking, OCR PDF ingestion, and Qdrant integration for legal documents."
)

# ============================================================
# Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Frontend Setup (Templates + Static Files)
# ============================================================
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def serve_frontend(request: Request):
    """
    Serves the minimal upload HTML UI.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# Routers
# ============================================================
# ✅ Existing API Endpoints
app.include_router(endpoints.router, prefix="/api/v1/module3")

# ✅ New Upload (PDF + OCR) Endpoint
app.include_router(upload_routes.router)

# ============================================================
# Startup Event - Initialize Qdrant Collection
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
