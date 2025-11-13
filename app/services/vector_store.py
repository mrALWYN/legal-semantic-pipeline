import uuid
import time
import tracemalloc
import logging
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    CollectionStatus,
    UpdateStatus
)
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.metrics import (
    embedding_time_hist,
    storage_time_hist,
    current_memory,
    ingest_chunks,
)

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Handles all Qdrant operations with robust error handling.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 collection_name: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize with settings fallback and robust error handling.
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection_name or settings.COLLECTION_NAME
        model_name = embedding_model or settings.EMBEDDING_MODEL
        
        # Initialize client with retry logic
        self.client = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(
                    url=f"http://{self.host}:{self.port}",
                    timeout=30,
                    # Disable internal validation to avoid version issues
                    prefer_grpc=False
                )
                # Test connection
                self.client.get_collections()
                logger.info(f"[QDRANT] ✅ Client initialized for {self.host}:{self.port}")
                break
            except Exception as e:
                logger.warning(f"[QDRANT] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"[QDRANT] ❌ Failed to initialize client after {max_retries} attempts")
                    raise
                time.sleep(2)

        # Initialize embedding model
        try:
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"[QDRANT] ✅ Embedding model '{model_name}' loaded")
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to load embedding model: {e}")
            raise

    async def create_collection_safe(self):
        """Create collection with robust error handling and compatibility."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Method 1: Try low-level API first (most reliable)
                try:
                    response = self.client._client.get(f"/collections/{self.collection_name}")
                    if response.status_code == 200:
                        logger.info(f"[QDRANT] Collection '{self.collection_name}' exists ✅")
                        return True
                except Exception:
                    pass  # Collection doesn't exist, continue to create
                
                logger.info(f"[QDRANT] Creating collection '{self.collection_name}'...")
                
                # Method 2: Use low-level API for creation
                collection_data = {
                    "name": self.collection_name,
                    "vectors": {
                        "size": settings.EMBEDDING_DIM,
                        "distance": "Cosine"
                    }
                }
                
                response = self.client._client.put(
                    f"/collections/{self.collection_name}",
                    json=collection_data
                )
                
                if response.status_code == 200:
                    logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created successfully")
                    return True
                else:
                    logger.warning(f"[QDRANT] Low-level API creation failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"[QDRANT] Method 1 failed: {e}")
                
            # Method 3: Fallback to standard API
            try:
                collections_response = self.client.get_collections()
                existing_collections = [col.name for col in collections_response.collections]
                
                if self.collection_name not in existing_collections:
                    logger.info(f"[QDRANT] Creating collection '{self.collection_name}' via standard API...")
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=settings.EMBEDDING_DIM,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created via standard API")
                else:
                    logger.info(f"[QDRANT] Collection '{self.collection_name}' exists ✅")
                return True
                
            except Exception as e:
                logger.warning(f"[QDRANT] Method 2 failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"[QDRANT] ❌ Failed to create collection after {max_retries} attempts")
                    return False

    async def upsert_chunks(self, chunks: List[Dict], document_id: str) -> float:
        """Safe upsert with comprehensive error handling and batching."""
        if not chunks:
            logger.warning("[QDRANT] No chunks provided for upsert.")
            return 0.0

        logger.info(f"[QDRANT] Embedding and upserting {len(chunks)} chunks for '{document_id}'...")

        points = []
        tracemalloc.start()
        start_embed = time.time()
        total_points = 0

        try:
            # Process chunks in smaller batches to avoid memory issues
            batch_size = 50
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]
                batch_points = []
                
                logger.info(f"[QDRANT] Processing batch {batch_start//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                for i, chunk in enumerate(batch_chunks):
                    text = chunk.get("text", "").strip()
                    if not text or len(text) < 10:  # Minimum text length
                        continue

                    # Embed text
                    vector = self.embedder.encode(text).tolist()
                    
                    # Create point with proper structure
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "chunk_id": chunk.get("chunk_id", f"{document_id}_{batch_start + i}"),
                            "citation_id": chunk.get("citation_id", document_id),
                            "document_id": document_id,
                            "text": text,
                            "type": chunk.get("type", "Unknown"),
                            "confidence": float(chunk.get("confidence", 0.0)),
                            "sentence_count": int(chunk.get("sentence_count", 0)),
                            "char_count": int(chunk.get("char_count", len(text))),
                            "has_citations": bool(chunk.get("has_citations", False)),
                            "has_statutes": bool(chunk.get("has_statutes", False)),
                            "has_parties": bool(chunk.get("has_parties", False)),
                        }
                    )
                    batch_points.append(point)

                if batch_points:
                    # Upsert batch
                    upsert_result = self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points,
                        wait=True
                    )
                    
                    total_points += len(batch_points)
                    logger.info(f"[QDRANT] ✅ Batch upserted {len(batch_points)} chunks")

            embedding_time = time.time() - start_embed
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Update metrics
            try:
                embedding_time_hist.observe(embedding_time)
                current_memory.set(peak)
                ingest_chunks.inc(total_points)
            except Exception as metric_err:
                logger.warning(f"[METRICS] Failed to record metrics: {metric_err}")

            logger.info(
                f"[QDRANT] ✅ Completed: {total_points} chunks "
                f"(embedding: {embedding_time:.2f}s, memory: {peak/1e6:.2f}MB)"
            )

            return embedding_time

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Upsert failed: {e}")
            tracemalloc.stop()
            raise

    async def ingest_chunks(self, chunks: List[Dict]) -> bool:
        """Robust ingestion with comprehensive error handling."""
        try:
            document_id = chunks[0].get("document_id", "unknown_document") if chunks else "unknown_document"
            logger.info(f"[QDRANT] 🚀 Starting ingestion for {len(chunks)} chunks (document: {document_id})")
            
            # Step 1: Ensure collection exists
            logger.info("[QDRANT] 📝 Step 1/3: Verifying collection...")
            collection_ok = await self.create_collection_safe()
            if not collection_ok:
                logger.error("[QDRANT] ❌ Collection setup failed")
                return False
            
            # Step 2: Upsert chunks
            logger.info("[QDRANT] 📝 Step 2/3: Upserting chunks...")
            try:
                embedding_time = await self.upsert_chunks(chunks, document_id)
            except Exception as e:
                logger.error(f"[QDRANT] ❌ Upsert failed: {e}")
                return False
            
            # Step 3: Verify success
            logger.info("[QDRANT] 📝 Step 3/3: Verifying ingestion...")
            try:
                # Simple verification - try to count points
                count_result = self.client.count(self.collection_name)
                points_count = count_result.count if count_result else "unknown"
                logger.info(f"[QDRANT] ✅ Total points in collection: {points_count}")
            except Exception as e:
                logger.warning(f"[QDRANT] ⚠️ Could not verify points count: {e}")
                points_count = "unknown"
            
            logger.info(f"[QDRANT] 🎉 INGESTION COMPLETE")
            logger.info(f"  • Document: {document_id}")
            logger.info(f"  • Chunks ingested: {len(chunks)}")
            logger.info(f"  • Total points: {points_count}")
            logger.info(f"  • Embedding time: {embedding_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to ingest chunks: {e}")
            return False

    def search(self, query: str, top_k: int = 10, score_threshold: float = 0.3) -> List[Dict]:
        """Safe search with error handling."""
        try:
            if not query or not query.strip():
                logger.warning("[QDRANT] Empty query provided for search")
                return []

            logger.info(f"[QDRANT] 🔍 Searching for '{query}' (top_k={top_k})")

            # Encode query
            query_vec = self.embedder.encode(query).tolist()
            top_k = min(top_k, 50)  # Safety cap

            # Search with score threshold
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=score_threshold,
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
                    "metadata": {
                        "chunk_id": payload.get("chunk_id"),
                        "confidence": payload.get("confidence"),
                        "sentence_count": payload.get("sentence_count"),
                        "char_count": payload.get("char_count"),
                        "has_citations": payload.get("has_citations"),
                        "has_statutes": payload.get("has_statutes"),
                        "has_parties": payload.get("has_parties"),
                    }
                })

            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"[QDRANT] ✅ Retrieved {len(results)} results for query '{query}'")
            return results

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Search failed: {e}")
            return []

    async def health_check(self) -> bool:
        """Health check with robust error handling."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Health check failed: {e}")
            return False

    async def get_collection_info(self) -> Dict:
        """Get collection info with fallback."""
        try:
            # Try low-level API first
            try:
                response = self.client._client.get(f"/collections/{self.collection_name}")
                if response.status_code == 200:
                    data = response.json()
                    result = data.get('result', {})
                    return {
                        "name": result.get('name', 'unknown'),
                        "vector_size": result.get('vectors', {}).get('size', 0),
                        "distance": result.get('vectors', {}).get('distance', 'unknown'),
                        "points_count": result.get('points_count', 0),
                        "status": result.get('status', 'unknown'),
                    }
            except:
                pass
            
            # Fallback to standard API
            try:
                info = self.client.get_collection(self.collection_name)
                return {
                    "name": getattr(info.config.params, 'name', 'unknown'),
                    "vector_size": getattr(info.config.params.vectors, 'size', 0),
                    "distance": getattr(info.config.params.vectors, 'distance', 'unknown'),
                    "points_count": getattr(info, 'points_count', 0),
                    "status": getattr(info, 'status', 'unknown'),
                }
            except:
                pass
                
            return {}
            
        except Exception as e:
            logger.warning(f"[QDRANT] Failed to get collection info: {e}")
            return {}

# Backward compatibility for methods that might be called elsewhere
VectorStoreService.create_collection = VectorStoreService.create_collection_safe
