import uuid
import logging
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import (
    IngestDocumentRequest,
    IngestDocumentResponse,
    QueryResponse,
    SearchResult,
)
from app.core.pipeline import ingest_legal_document
from app.services.vector_store import VectorStoreService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================
# 📄 DOCUMENT INGESTION ENDPOINT
# ============================================================
@router.post("/ingest-document", response_model=IngestDocumentResponse)
async def ingest_document(request: IngestDocumentRequest):
    """
    Ingest a legal document → perform semantic chunking → embed + store in Qdrant.
    """
    try:
        job_id = str(uuid.uuid4())
        citation_id = request.citation_id or job_id

        logger.info(f"[INGEST] Starting ingestion for document: {citation_id}")

        result = await ingest_legal_document(
            text=request.document_text,
            chunking_strategy=request.strategy,
            document_id=citation_id
        )

        logger.info(f"[INGEST] ✅ Completed ingestion for {citation_id}")

        return {
            "job_id": job_id,
            "total_chunks": result["total_chunks"],
            "status": "success",
            "message": result["message"],
        }

    except Exception as e:
        logger.exception(f"[ERROR] ❌ Failed to ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔍 SEMANTIC QUERY ENDPOINT
# ============================================================
@router.get("/query-twin", response_model=QueryResponse)
async def query_twin(
    query: str = Query(..., description="Text query for semantic search"),
    top_k: int = Query(10, description="Number of top results to return (default=10)"),
    citation_id: str | None = Query(None, description="(Optional) Filter by citation/document ID")
):
    """
    Perform a semantic vector search on Qdrant.
    Optionally filter results by `citation_id` (document name or ID).
    """
    try:
        logger.info(f"[QUERY] Searching for: '{query}' (top_k={top_k}, filter={citation_id})")

        # ✅ Initialize Qdrant client
        vector_service = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME
        )

        # ✅ Get search results
        all_results = vector_service.search(query=query, top_k=top_k)

        # ✅ Optional filter by citation/document name
        if citation_id:
            filtered_results = [
                r for r in all_results
                if r.get("citation_id", "").lower() == citation_id.lower()
            ]
        else:
            filtered_results = all_results

        # ✅ Format final results (include *all metadata*)
        results = [
            SearchResult(
                chunk_text=r.get("chunk_text", ""),
                citation_id=r.get("citation_id", ""),
                score=round(r.get("score", 0), 4),
                metadata=r.get("metadata", {})  # <-- include all metadata fields
            )
            for r in filtered_results
        ]

        logger.info(f"[QUERY] ✅ Returning {len(results)} results for query '{query}'")

        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        logger.exception(f"[ERROR] ❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
