import uuid
import logging
from app.core.config import settings
from app.services.chunking import SemanticChunker
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

async def ingest_legal_document(
    text: str,
    chunking_strategy: str,
    document_id: str
):
    """
    Main pipeline: chunk legal document → embed → upsert into Qdrant
    """
    try:
        logger.info("[PIPELINE] Starting document ingestion pipeline...")

        # ✅ 1. Initialize Qdrant vector store service
        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME
        )

        # ✅ 2. Ensure collection exists
        await vector_store.create_collection()
        logger.info("[PIPELINE] Qdrant collection ready ✅")

        # ✅ 3. Initialize chunker
        chunker = SemanticChunker(
            strategy=chunking_strategy,
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )

        # ✅ 4. Chunk document
        chunks = chunker.chunk_document(text)
        logger.info(f"[PIPELINE] Chunked document into {len(chunks)} segments")

        # ✅ 5. Embed and insert into Qdrant
        await vector_store.ingest_chunks(chunks, document_id)
        logger.info("[PIPELINE] Successfully inserted vectors into Qdrant ✅")

        return {
            "job_id": str(uuid.uuid4()),
            "total_chunks": len(chunks),
            "status": "success",
            "message": "Document processed successfully"
        }

    except Exception as e:
        logger.error(f"[PIPELINE] ❌ Ingestion failed: {e}")
        raise e
