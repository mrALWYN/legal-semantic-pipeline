from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    QueryRequest,
    QueryResponse,
    SearchResult,
)
from app.core.pipeline import ingest_legal_document
from app.services.vector_store import VectorStoreService
from app.core.config import settings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import logging
import uuid

router = APIRouter()
logger = logging.getLogger(__name__)

# ✅ Document ingestion endpoint
@router.post("/ingest-document", response_model=IngestDocumentResponse)
async def ingest_document(request: IngestDocumentRequest):
    """
    Ingest a legal document, perform semantic chunking, and store embeddings in Qdrant.
    """
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"[INGEST] Starting ingestion job {job_id}")

        result = await ingest_legal_document(
            text=request.document_text,
            chunking_strategy=request.strategy,
            document_id=request.citation_id or job_id
        )

        return {
            "job_id": job_id,
            "total_chunks": result["total_chunks"],
            "status": "success",
            "message": result["message"],
        }

    except Exception as e:
        logger.exception(f"[ERROR] Failed to ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Semantic search endpoint
@router.get("/query-twin", response_model=QueryResponse)
async def query_twin(query: str, top_k: int = 3):
    """
    Perform a semantic vector search against Qdrant for relevant legal chunks.
    """
    try:
        # Reuse same client + model as in ingestion
        client = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

        query_vec = embedder.encode(query).tolist()

        search_results = client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k
        )

        results = []
        for hit in search_results:
            payload = hit.payload or {}
            results.append(SearchResult(
                chunk_text=payload.get("text", ""),
                citation_id=payload.get("citation_id", ""),
                score=hit.score,
                metadata=payload
            ))

        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.exception(f"[ERROR] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
