import logging
from typing import Literal
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ============================================================
# 📋 User Input Models (Pydantic v2)
# ============================================================

class ChunkingRequest(BaseModel):
    """User-provided chunking configuration"""
    min_chunk_size: int = Field(default=500, ge=100, le=2000, description="Minimum chunk size in characters")
    max_chunk_size: int = Field(default=4500, ge=1000, le=10000, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(default=250, ge=0, le=1000, description="Overlap between chunks in characters")
    embedding_model: Literal["all-MiniLM-L6-v2", "all-mpnet-base-v2"] = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings"
    )
    chunking_technique: Literal["semantic", "recursive"] = Field(
        default="semantic",
        description="Chunking strategy: 'semantic' (LLM-aware) or 'recursive' (paragraph-based)"
    )
    document_id: str = Field(description="Unique identifier for the document")
    text: str = Field(description="Document text to chunk")

    @field_validator("max_chunk_size")
    @classmethod
    def validate_max_vs_min(cls, v, info):
        """Ensure max_chunk_size > min_chunk_size"""
        min_size = info.data.get("min_chunk_size", 100)
        if v <= min_size:
            raise ValueError(f"max_chunk_size ({v}) must be greater than min_chunk_size ({min_size})")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "min_chunk_size": 500,
                "max_chunk_size": 4500,
                "chunk_overlap": 250,
                "embedding_model": "all-MiniLM-L6-v2",
                "chunking_technique": "semantic",
                "document_id": "case_2024_001",
                "text": "Full document text here..."
            }
        }


# ============================================================
# 🏭 Chunker Factory
# ============================================================

def get_chunker(config: ChunkingRequest):
    """
    Factory function to instantiate the correct chunker based on user selection.
    
    Args:
        config: ChunkingRequest with user parameters
        
    Returns:
        SemanticLegalChunker or RecursiveChunker instance
    """
    from app.services.chunking import SemanticLegalChunker, RecursiveChunker
    
    logger.info(
        f"[FACTORY] Creating {config.chunking_technique} chunker | "
        f"min={config.min_chunk_size}, max={config.max_chunk_size}, "
        f"overlap={config.chunk_overlap}, model={config.embedding_model}"
    )
    
    if config.chunking_technique == "semantic":
        return SemanticLegalChunker(
            min_chunk_size=config.min_chunk_size,
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embedding_model,
        )
    elif config.chunking_technique == "recursive":
        return RecursiveChunker(
            min_chunk_size=config.min_chunk_size,
            max_chunk_size=config.max_chunk_size,
            chunk_overlap=config.chunk_overlap,
            embedding_model=config.embedding_model,
        )
    else:
        raise ValueError(f"Unknown chunking technique: {config.chunking_technique}")
