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
    - Collection creation / management
    - Semantic vector embedding + upsert
    - Search and retrieval
    """

    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    # ------------------------------------------------------------
    # ✅ Create or Verify Qdrant Collection
    # ------------------------------------------------------------
    async def create_collection(self):
        """Ensure Qdrant collection exists, create if missing."""
        try:
            collections = self.client.get_collections().collections
            existing = [c.name for c in collections]

            if self.collection_name not in existing:
                logger.info(f"[QDRANT] Creating collection '{self.collection_name}' ...")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"[QDRANT] Collection '{self.collection_name}' already exists ✅")

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to create collection: {e}")
            raise

    # ------------------------------------------------------------
    # ✅ Upsert Semantic Chunks
    # ------------------------------------------------------------
    async def upsert_chunks(self, chunks: List[Dict], document_id: str):
        """
        Embed and insert/update semantic chunks into Qdrant with metadata.
        Each chunk is linked to the provided `document_id`.
        """
        try:
            if not chunks:
                logger.warning("[QDRANT] No chunks provided for upsert.")
                return

            logger.info(f"[QDRANT] Embedding and upserting {len(chunks)} chunks for '{document_id}' ...")

            vectors, payloads, ids = [], [], []

            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "").strip()
                if not text:
                    continue

                # ✅ Embed text using SentenceTransformer
                vector = self.embedder.encode(text).tolist()
                vectors.append(vector)
                ids.append(str(uuid.uuid4()))

                # ✅ Construct payload with metadata
                payloads.append({
                    "chunk_id": chunk.get("chunk_id", str(i)),
                    "citation_id": chunk.get("citation_id", document_id),
                    "document_id": document_id,
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
                logger.warning("[QDRANT] No valid vectors to upsert.")
                return

            # ✅ Perform Qdrant upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )

            logger.info(f"[QDRANT] ✅ Successfully upserted {len(vectors)} chunks into '{self.collection_name}'")

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Upsert failed: {e}")
            raise

    # ------------------------------------------------------------
    # ✅ Semantic Search
    # ------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform a semantic similarity search in Qdrant.
        Returns top chunks with their full metadata.
        """
        try:
            if not query.strip():
                raise ValueError("Empty query provided for semantic search.")

            logger.info(f"[QDRANT] 🔍 Searching for '{query}' (top_k={top_k})")

            query_vec = self.embedder.encode(query).tolist()
            top_k = min(top_k, 50)  # Safety cap

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k,
            )

            results = []
            for hit in search_results:
                payload = hit.payload or {}
                results.append({
                    "chunk_text": payload.get("text", ""),
                    "document_id": payload.get("document_id", ""),
                    "citation_id": payload.get("citation_id", ""),
                    "type": payload.get("type", "Unknown"),
                    "score": round(hit.score, 4),
                    "metadata": payload,  # ✅ send full metadata for frontend
                })

            logger.info(f"[QDRANT] ✅ Retrieved {len(results)} results for query '{query}'")
            return results

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Search failed: {e}")
            raise
