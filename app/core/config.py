from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # ============================================================
    # Qdrant Configuration
    # ============================================================
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_precedents"

    # ============================================================
    # Embedding Model
    # ============================================================
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ============================================================
    # Chunking Configuration
    # ============================================================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ============================================================
    # API Metadata
    # ============================================================
    API_TITLE: str = "Legal Semantic Pipeline"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = (
        "FastAPI service for semantic chunking and Qdrant integration "
        "for legal document retrieval and analysis."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Singleton-style access to settings"""
    return Settings()


# Global singleton instance
settings = get_settings()
