import asyncio
import os
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.config.model_config import log_model_config
from app.services.ocr import OCRService

# ============================================================
# Local imports
# ============================================================
from app.api.v1 import (
    endpoints,
    upload_routes,
    models as model_routes,
    feedback as feedback_routes,
    metrics_endpoint,
    health_routes,
)
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
    version="1.3.0",
    description=(
        "⚖️ Semantic Legal Document Pipeline with OCR-PDF ingestion, "
        "Qdrant vector search, MLflow experiment tracking, and Prometheus metrics."
    ),
)

# ============================================================
# Middleware
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Frontend Setup (Templates + Static)
# ============================================================
templates = Jinja2Templates(directory="templates")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)


@app.get("/", include_in_schema=False)
async def serve_frontend(request: Request):
    """Serves the minimal upload + query HTML UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# Routers (API Endpoints)
# ============================================================

# ✅ Core endpoints (document ingestion + query)
app.include_router(endpoints.router, prefix="/api/v1/module3")

# ✅ Upload (PDF + OCR ingestion)
app.include_router(upload_routes.router)

# ✅ MLflow model management & experiment endpoints
app.include_router(model_routes.router)

# ✅ Feedback endpoint
app.include_router(feedback_routes.router)

# ✅ Prometheus metrics endpoint
app.include_router(metrics_endpoint.router)

# ✅ Health and readiness endpoints
app.include_router(health_routes.router)


# ============================================================
# Startup Event: Qdrant Initialization
# ============================================================
@app.on_event("startup")
async def startup_event():
    """
    Initialize models and log configuration on startup.
    """
    logger.info("🚀 Starting Legal Semantic Pipeline...")

    # Log model configuration (HF cached models etc.)
    log_model_config()

    # No EasyOCR pre-load: using PyMuPDF for fast text extraction (no OCR)
    logger.info("[STARTUP] ℹ️ Using PyMuPDF for PDF text extraction (no EasyOCR/Tesseract).")

    logger.info("✅ Legal Semantic Pipeline started!")


# ============================================================
# Health Check (basic legacy endpoint for backward compatibility)
# ============================================================
@app.get("/health", tags=["System"])
async def health_check():
    """Simple API health endpoint to verify the backend is running."""
    return {"status": "ok", "message": "API is running successfully 🚀"}
