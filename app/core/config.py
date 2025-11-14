from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import List, Dict


class Settings(BaseSettings):
    # ============================================================
    # Qdrant Configuration
    # ============================================================
    QDRANT_HOST: str = "qdrant"        # Docker service name
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_precedents"

    # ============================================================
    # Embedding Model Configuration
    # ============================================================
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # Default dimension size, but will be dynamically determined

    # ============================================================
    # Chunking Configuration
    # ============================================================
    CHUNK_SIZE: int = 500              # Minimum characters per chunk
    MAX_CHUNK_SIZE: int = 1000         # Maximum allowed characters per chunk
    CHUNK_OVERLAP: int = 100           # Overlap between adjacent chunks (for continuity)
    MERGE_THRESHOLD: int = 200         # Merge tiny consecutive chunks

    # ============================================================
    # Logging / API Metadata
    # ============================================================
    LOG_LEVEL: str = "INFO"

    API_TITLE: str = "Legal Semantic Pipeline"
    API_VERSION: str = "1.2.0"
    API_DESCRIPTION: str = (
        "FastAPI service for OCR-based ingestion, semantic chunking, "
        "and Qdrant vector storage of Indian legal documents."
    )

    # ============================================================
    # ⚙️ MLflow / Model Registry Configuration
    # ============================================================
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"  # Changed to use docker service name
    REGISTERED_MODEL_NAME: str = "legal-embed-model"
    EXPERIMENT_NAME: str = "ingest_experiments"

    # ============================================================
    # 📊 Prometheus Metrics Configuration
    # ============================================================
    PROMETHEUS_NAMESPACE: str = "legal_pipeline"
    METRICS_ENABLED: bool = True

    # ============================================================
    # 💬 Feedback Store Configuration
    # ============================================================
    FEEDBACK_STORE_PATH: str = "feedback/feedback.jsonl"

    # ============================================================
    # 🧠 Experiment Management Configuration
    # ============================================================
    # DVC / experiment versioning parameters
    DVC_REMOTE: str | None = None
    ENABLE_DVC: bool = True

    # Environment flags to toggle chunking strategies or experiments
    DEFAULT_CHUNK_STRATEGY: str = "semantic-legal"  # or 'recursive-legal'
    AVAILABLE_EMBED_MODELS: List[str] = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",  # Added mpnet model
        "sentence-transformers/paraphrase-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ]

    # ============================================================
    # Model Dimension Mapping (for reference)
    # ============================================================
    MODEL_DIMENSIONS: Dict[str, int] = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-MiniLM-L12-v2": 384,
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384
    }

    # ============================================================
    # Pydantic Environment Configuration
    # ============================================================
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"  # Allow extra environment variables
    )


# ============================================================
# 🧠 Model Dimension Mapping (Reverse and Additional)
# ============================================================
MODEL_DIMENSIONS_REVERSE: Dict[int, str] = {
    384: "all-MiniLM-L6-v2",
    768: "all-mpnet-base-v2",
}

MODEL_NAMES: Dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L12-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}

AVAILABLE_MODELS: Dict[str, Dict] = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, lightweight (384 dims)",
        "dimensions": 384,
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "description": "High quality (768 dims)",
        "dimensions": 768,
    },
    "paraphrase-MiniLM-L12-v2": {
        "name": "paraphrase-MiniLM-L12-v2",
        "description": "Paraphrase model (384 dims)",
        "dimensions": 384,
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "description": "QA optimized (384 dims)",
        "dimensions": 384,
    },
}


# ============================================================
# Singleton-style settings accessor
# ============================================================
@lru_cache()
def get_settings() -> Settings:
    """Singleton-style access to configuration."""
    return Settings()


# Global singleton instance
settings = get_settings()
