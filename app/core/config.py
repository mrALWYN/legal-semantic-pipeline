from pydantic_settings import BaseSettings
from functools import lru_cache


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
    EMBEDDING_DIM: int = 384  # Dimension size for the above model

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
    API_VERSION: str = "1.1.0"
    API_DESCRIPTION: str = (
        "FastAPI service for OCR-based ingestion, semantic chunking, "
        "and Qdrant vector storage of Indian legal documents."
    )

    # ============================================================
    # Pydantic Environment Configuration
    # ============================================================
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ============================================================
# Singleton-style settings accessor
# ============================================================
@lru_cache()
def get_settings() -> Settings:
    """Singleton-style access to configuration."""
    return Settings()


# Global singleton instance
settings = get_settings()
