import uuid
import time
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
from app.services.metrics import (
    search_requests,
    search_duration_hist,
    search_results,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["Core Operations"])


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

        logger.info(f"[INGEST] 🚀 Starting ingestion for document: {citation_id}")

        result = await ingest_legal_document(
            text=request.document_text,
            chunking_strategy=request.strategy,
            document_id=citation_id,
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
# 🔍 SEMANTIC QUERY ENDPOINT (with Prometheus metrics)
# ============================================================
@router.get("/query-twin", response_model=QueryResponse)
async def query_twin(
    query: str = Query(..., description="Text query for semantic search"),
    top_k: int = Query(10, description="Number of top results to return (default=10)"),
    document_id: str | None = Query(
        None, description="(Optional) Filter by document ID or citation"
    ),
):
    """
    Perform a semantic vector search on Qdrant.
    Optionally filter by `document_id` (citation name).
    Returns top-k matches with full metadata.
    """
    try:
        # ✅ Increment Prometheus counters
        search_requests.inc()
        start_time = time.time()

        logger.info(
            f"[QUERY] 🔍 Searching for: '{query}' (top_k={top_k}, filter={document_id})"
        )

        # ✅ Initialize Qdrant vector service
        vector_service = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME,
        )

        # ✅ Perform semantic search
        all_results = vector_service.search(query=query, top_k=top_k)

        # ✅ Optional filter by document ID
        if document_id:
            filtered_results = [
                r
                for r in all_results
                if r.get("document_id", "").lower() == document_id.lower()
            ]
            logger.info(
                f"[QUERY] Applied filter for document_id='{document_id}', {len(filtered_results)} results left."
            )
        else:
            filtered_results = all_results

        # ✅ Calculate duration & log metrics
        duration = time.time() - start_time
        search_duration_hist.observe(duration)
        search_results.inc(len(filtered_results))

        logger.info(
            f"[METRICS] Search duration={duration:.2f}s | results={len(filtered_results)}"
        )

        # ✅ Prepare structured response - FIXED: Include type field
        results = []
        for r in filtered_results:
            # Extract metadata and include type at the root level
            metadata = r.get("metadata", {})
            
            # Create SearchResult with type at root level
            result = SearchResult(
                chunk_text=r.get("chunk_text", ""),
                citation_id=r.get("document_id", r.get("citation_id", "")),
                score=round(r.get("score", 0), 4),
                type=r.get("type", ""),  # Add type at root level
                metadata=r.get("metadata", {}),
            )
            results.append(result)

        logger.info(f"[QUERY] ✅ Returning {len(results)} results for query '{query}'")

        return {
            "query": query,
            "results": results,
            "total_found": len(results),
        }

    except Exception as e:
        logger.exception(f"[ERROR] ❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
