import uuid
import logging
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Batch
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(url=f"http://{host}:{port}")
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    async def create_collection(self):
        """Ensure the collection exists."""
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]
        if self.collection_name not in existing:
            logger.info(f"[QDRANT] Creating collection '{self.collection_name}'")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # 384 dimensions for MiniLM
                    distance=Distance.COSINE
                )
            )
            logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created successfully.")
        else:
            logger.info(f"[QDRANT] Collection '{self.collection_name}' already exists ✅")

    async def ingest_chunks(self, chunks: List[Dict], document_id: str):
        """Embed and insert chunks into Qdrant."""
        logger.info(f"[QDRANT] Embedding and upserting {len(chunks)} chunks...")

        vectors = []
        payloads = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            vector = self.embedder.encode(text).tolist()
            vectors.append(vector)
            payloads.append({
                "chunk_id": i,
                "citation_id": document_id,
                "text": text,
            })

        self.client.upsert(
            collection_name=self.collection_name,
            points=Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(vectors))],
                vectors=vectors,
                payloads=payloads
            )
        )
        logger.info(f"[QDRANT] ✅ Inserted {len(vectors)} vectors into collection '{self.collection_name}'")
