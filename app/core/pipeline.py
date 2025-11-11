import uuid
import logging
from app.core.config import settings
from app.services.chunking import SemanticLegalChunker
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


async def ingest_legal_document(
    text: str,
    chunking_strategy: str,
    document_id: str
):
    """
    Full semantic ingestion pipeline:
    1️⃣ Semantic chunking
    2️⃣ Legal classification
    3️⃣ Embedding + Qdrant upsert
    """
    try:
        logger.info(f"[PIPELINE] Starting ingestion for document '{document_id}'")

        # ✅ 1. Initialize Qdrant vector store
        vector_store = VectorStoreService(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            collection_name=settings.COLLECTION_NAME
        )

        # ✅ 2. Ensure Qdrant collection exists
        await vector_store.create_collection()
        logger.info("[PIPELINE] Qdrant collection verified ✅")

        # ✅ 3. Initialize semantic legal chunker
        #    Uses tuned parameters for Indian legal judgments
        chunker = SemanticLegalChunker(
            min_chunk_size=900,         # Prevents tiny fragments
            max_chunk_size=2200,        # Keeps reasoning within token limits
            chunk_overlap=300,          # Ensures continuity across chunks
            embedding_model=settings.EMBEDDING_MODEL,
        )

        # ✅ 4. Perform semantic chunking + classification
        chunks = chunker.chunk_document(text=text, document_id=document_id)

        if not chunks:
            raise ValueError("No valid chunks generated from document text.")

        logger.info(f"[PIPELINE] ✅ Chunked document into {len(chunks)} semantic chunks.")

        # ✅ 5. Add citation/document metadata to each chunk
        for c in chunks:
            c["citation_id"] = document_id

        # ✅ 6. Embed + insert chunks into Qdrant
        await vector_store.upsert_chunks(chunks, document_id)
        logger.info(f"[PIPELINE] ✅ Inserted {len(chunks)} chunks into Qdrant collection.")

        # ✅ 7. Return structured response
        return {
            "job_id": str(uuid.uuid4()),
            "total_chunks": len(chunks),
            "status": "success",
            "message": f"Document '{document_id}' processed successfully.",
        }

    except Exception as e:
        logger.error(f"[PIPELINE] ❌ Ingestion failed for '{document_id}': {e}")
        raise
