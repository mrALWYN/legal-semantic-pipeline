import uuid
import logging
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Batch
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Handles all Qdrant operations:
    - Collection management
    - Chunk ingestion
    - Semantic search
    """

    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------
    # Create or Verify Collection
    # ------------------------------------------------------------
    async def create_collection(self):
        try:
            collections = self.client.get_collections().collections
            existing = [c.name for c in collections]

            if self.collection_name not in existing:
                logger.info(f"[QDRANT] Creating collection '{self.collection_name}' ...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"[QDRANT] Collection '{self.collection_name}' already exists ✅")

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to create collection: {e}")
            raise e

    # ------------------------------------------------------------
    # Ingest / Upsert Chunks
    # ------------------------------------------------------------
    async def ingest_chunks(self, chunks: List[Dict], document_id: str):
        """
        Embed and insert chunks into Qdrant with metadata.
        """
        try:
            logger.info(f"[QDRANT] Embedding and inserting {len(chunks)} chunks for document '{document_id}'")

            vectors, payloads = [], []
            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                if not text.strip():
                    continue

                vector = self.embedder.encode(text).tolist()
                vectors.append(vector)

                payloads.append({
                    "chunk_id": chunk.get("chunk_id", str(i)),
                    "citation_id": document_id,
                    "text": text,
                    "type": chunk.get("type", "Unknown"),
                    "confidence": chunk.get("confidence"),
                    "sentence_count": chunk.get("sentence_count"),
                    "char_count": chunk.get("char_count"),
                    "has_citations": chunk.get("has_citations"),
                    "has_statutes": chunk.get("has_statutes"),
                    "has_parties": chunk.get("has_parties"),
                })

            if not vectors:
                logger.warning("[QDRANT] No valid chunks to ingest.")
                return

            self.client.upsert(
                collection_name=self.collection_name,
                points=Batch(
                    ids=[str(uuid.uuid4()) for _ in range(len(vectors))],
                    vectors=vectors,
                    payloads=payloads
                )
            )

            logger.info(f"[QDRANT] ✅ Inserted {len(vectors)} vectors into collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Vector ingestion failed: {e}")
            raise e

    # ------------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform a semantic similarity search in Qdrant.

        Returns a list of payloads + similarity scores.
        """
        try:
            query_vec = self.embedder.encode(query).tolist()
            top_k = min(top_k, 50)  # safety cap

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k
            )

            results = []
            for hit in search_results:
                payload = hit.payload or {}
                results.append({
                    "chunk_text": payload.get("text", ""),
                    "citation_id": payload.get("citation_id", ""),
                    "type": payload.get("type", "Unknown"),
                    "score": hit.score,
                    "metadata": payload
                })

            logger.info(f"[QDRANT] 🔍 Retrieved {len(results)} search results for query: '{query}'")
            return results

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Search failed: {e}")
            raise e
