# app/models/schemas.py
from typing import List, Optional, Dict
from pydantic import BaseModel

class IngestDocumentRequest(BaseModel):
    document_text: str
    strategy: str = "semantic-legal"
    citation_id: Optional[str] = None

class IngestDocumentResponse(BaseModel):
    job_id: str
    total_chunks: int
    status: str
    message: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    chunk_text: str
    citation_id: str
    score: float
    type: str = ""
    metadata: Dict
    chunking_metadata: Dict

    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
