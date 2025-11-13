import uuid
import time
import logging

from app.core.config import settings
from app.services.chunking import SemanticLegalChunker
from app.services.vector_store import VectorStoreService

# Newly added imports
from app.services.chunking import RecursiveChunker  # supports recursive chunking experiments
from app.services.mlflow_service import (
    start_run, log_model_metadata, log_metrics, log_params, end_run
)
from app.services.metrics import (
    ingest_calls,
    ingest_chunks,
    embedding_time_hist,
    storage_time_hist,
    current_memory,
)

logger = logging.getLogger(__name__)


async def ingest_legal_document(
    text: str,
    chunking_strategy: str,
    document_id: str
):
    """
    Full semantic ingestion pipeline with:
    - Semantic or recursive chunking
    - MLflow experiment logging
    - Prometheus metrics tracking
    """
    run = None
    try:
        logger.info(f"[PIPELINE] Starting ingestion for document '{document_id}'")
        ingest_calls.inc()

        # ✅ 1. Initialize Qdrant vector store
        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME
        )

        # ✅ 2. Ensure Qdrant collection exists
        await vector_store.create_collection()
        logger.info("[PIPELINE] Qdrant collection verified ✅")

        # ✅ 3. Choose chunking strategy
        if chunking_strategy.lower().startswith("recursive"):
            chunker = RecursiveChunker(
                min_chunk_size=settings.CHUNK_SIZE,
                max_chunk_size=settings.MAX_CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                embedding_model=settings.EMBEDDING_MODEL,
            )
            strategy_used = "recursive-legal"
        else:
            chunker = SemanticLegalChunker(
                min_chunk_size=settings.CHUNK_SIZE,
                max_chunk_size=settings.MAX_CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                embedding_model=settings.EMBEDDING_MODEL,
            )
            strategy_used = "semantic-legal"

        # ✅ 4. Perform chunking
        start_chunk = time.time()
        chunks = chunker.chunk_document(text=text, document_id=document_id)
        chunk_time = time.time() - start_chunk

        if not chunks:
            raise ValueError("No valid chunks generated from document text.")

        ingest_chunks.inc(len(chunks))
        logger.info(f"[PIPELINE] ✅ Chunked document into {len(chunks)} chunks using '{strategy_used}' strategy.")

        # ✅ 5. Add citation/document metadata
        for c in chunks:
            c["citation_id"] = document_id

        # ✅ 6. Embed + insert chunks into Qdrant
        logger.info(f"[PIPELINE] 🔄 Starting embedding & storage for {len(chunks)} chunks...")
        start_embed = time.time()
        embedding_time = await vector_store.upsert_chunks(chunks, document_id)
        storage_time = time.time() - start_embed
        embedding_time_hist.observe(embedding_time)
        storage_time_hist.observe(storage_time)

        logger.info(
            f"[PIPELINE] ✅ Completed embedding in {embedding_time:.2f}s and storage in {storage_time:.2f}s."
        )

        # ✅ 7. MLflow Experiment Logging
        try:
            run = start_run(
                experiment_name=settings.EXPERIMENT_NAME,
                run_name=document_id,
            )
            
            if run:
                # Log model metadata as parameters
                log_params({
                    "model_name": settings.EMBEDDING_MODEL,
                    "embedding_dim": settings.EMBEDDING_DIM,
                    "chunk_size": settings.CHUNK_SIZE,
                    "overlap": settings.CHUNK_OVERLAP,
                    "chunking_strategy": strategy_used,
                    "pipeline_version": settings.API_VERSION
                })
                
                # Log metrics
                log_metrics({
                    "total_chunks": len(chunks),
                    "chunk_time": chunk_time,
                    "embedding_time": embedding_time,
                    "storage_time": storage_time,
                })
                
                logger.info(f"[MLFLOW] ✅ Logged run for document '{document_id}'.")
            
        except Exception as e:
            logger.warning(f"[MLFLOW] ⚠️ Logging failed: {e}")
        finally:
            if run:
                end_run()

        # ✅ 8. Return structured response
        return {
            "job_id": str(uuid.uuid4()),
            "total_chunks": len(chunks),
            "status": "success",
            "message": f"Document '{document_id}' processed successfully with strategy '{strategy_used}'.",
        }

    except Exception as e:
        logger.error(f"[PIPELINE] ❌ Ingestion failed for '{document_id}': {e}")
        # Ensure MLflow run is properly closed even on error
        if run:
            end_run()
        raise
