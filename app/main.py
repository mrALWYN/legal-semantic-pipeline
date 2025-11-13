import logging
import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests

from app.config.chunking_config import ChunkingRequest, get_chunker

# Import all API routers
from app.api.v1.endpoints import router as endpoints_router
from app.api.v1.upload_routes import router as upload_router
from app.api.v1.models import router as models_router
from app.api.v1.feedback import router as feedback_router
from app.api.v1.metrics_endpoint import router as metrics_router
from app.api.v1.health_routes import router as health_router

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
    title=os.getenv("API_TITLE", "Legal Semantic Pipeline"),
    version=os.getenv("API_VERSION", "1.0.0"),
    description="⚖️ Semantic document chunking with legal-aware classification",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 📋 Register All API Routers
# ============================================================

# Core endpoints (already includes /ingest-document and /query-twin)
app.include_router(endpoints_router, prefix="/api/v1", tags=["Core Operations"])

# PDF upload endpoint
app.include_router(upload_router, prefix="/api/v1", tags=["Upload"])

# Model management endpoints
app.include_router(models_router, prefix="/api/v1", tags=["Model Management"])

# Feedback endpoints
app.include_router(feedback_router, prefix="/api/v1", tags=["Feedback"])

# Metrics endpoint (Prometheus)
app.include_router(metrics_router, prefix="/api/v1", tags=["Metrics"])

# Health endpoints
app.include_router(health_router, prefix="/api/v1", tags=["System"])

# ============================================================
# 🖥️ Serve Frontend Template
# ============================================================

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def serve_frontend(request: Request):
    """Serve the main frontend interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# 🏥 Startup Event - Check Service Dependencies
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Check if Qdrant and MLflow are available on startup (with retries)"""
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}/health"
    
    logger.info("[STARTUP] Checking service dependencies...")
    
    # Retry logic
    max_retries = 5
    retry_delay = 3  # seconds
    
    for i in range(max_retries):
        try:
            # Check Qdrant
            qdrant_response = requests.get(qdrant_url, timeout=5)
            if qdrant_response.status_code == 200:
                logger.info(f"✅ [STARTUP] Qdrant is available at {qdrant_url}")
            else:
                logger.warning(f"⚠️  [STARTUP] Qdrant returned status {qdrant_response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  [STARTUP] Qdrant not available (attempt {i+1}/{max_retries}): {str(e)}")
            if i < max_retries - 1:
                logger.info(f"[STARTUP] Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.warning("[STARTUP] ⚠️  Qdrant unavailable, but API will start. Requests may fail until Qdrant is ready.")
        
        try:
            # Check MLflow
            mlflow_response = requests.get(mlflow_uri, timeout=5)
            if mlflow_response.status_code == 200:
                logger.info(f"✅ [STARTUP] MLflow is available at {mlflow_uri}")
                break
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️  [STARTUP] MLflow not available (attempt {i+1}/{max_retries}): {str(e)}")
            if i < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.warning("[STARTUP] ⚠️  MLflow unavailable, but API will start. Experiment tracking may fail.")


# ============================================================
# 🏥 Health Check
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "legal-semantic-pipeline"}


# ============================================================
# 📄 Chunking Endpoint (Direct endpoint - not in router)
# ============================================================

@app.post("/api/v1/chunk")
async def chunk_document(request: ChunkingRequest):
    """
    Chunk a legal document with user-specified parameters.
    
    **Parameters:**
    - `min_chunk_size`: Minimum chunk size (100-2000 chars)
    - `max_chunk_size`: Maximum chunk size (1000-10000 chars)
    - `chunk_overlap`: Overlap between chunks (0-1000 chars)
    - `embedding_model`: "all-MiniLM-L6-v2" or "all-mpnet-base-v2"
    - `chunking_technique`: "semantic" or "recursive"
    - `document_id`: Unique document identifier
    - `text`: Document text to chunk
    
    **Returns:**
    - Structured chunks with metadata (type, citations, statutes, parties)
    """
    try:
        logger.info(
            f"[API] Chunking request | technique={request.chunking_technique} | "
            f"model={request.embedding_model} | doc_id={request.document_id}"
        )
        
        chunker = get_chunker(request)
        chunks = chunker.chunk_document(request.text, request.document_id)
        
        if not chunks:
            raise ValueError("No chunks produced from document")
        
        return JSONResponse({
            "status": "success",
            "document_id": request.document_id,
            "chunking_technique": request.chunking_technique,
            "embedding_model": request.embedding_model,
            "chunk_count": len(chunks),
            "parameters": {
                "min_chunk_size": request.min_chunk_size,
                "max_chunk_size": request.max_chunk_size,
                "chunk_overlap": request.chunk_overlap,
            },
            "chunks": chunks,
        })
    except ValueError as ve:
        logger.error(f"[API] Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(ve)}")
    except Exception as e:
        logger.error(f"[API] Chunking failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


# ============================================================
# 📊 Status Endpoint (Direct endpoint - not in router)
# ============================================================

@app.get("/api/v1/status")
async def status():
    """Service status and configuration"""
    return {
        "service": "Legal Semantic Pipeline",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "available_models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        "chunking_techniques": ["semantic", "recursive"],
        "environment": {
            "qdrant_host": os.getenv("QDRANT_HOST"),
            "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI"),
            "hf_home": os.getenv("HF_HOME"),
        }
    }
