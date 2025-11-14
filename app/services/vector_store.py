# app/services/vector_store.py
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
    memory_usage_bytes,
    ingest_chunks,
    search_duration_hist,
    search_results,
    processing_failures,
    embedding_similarity,
    vector_index_size_bytes,
    qdrant_points_count
)
from app.services.drift_detection import drift_detector

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Handles all Qdrant operations with robust error handling and dynamic collection management.
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 collection_name: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize with settings fallback and robust error handling.
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.base_collection_name = collection_name or settings.COLLECTION_NAME
        self.model_name = embedding_model or settings.EMBEDDING_MODEL
        
        # Initialize client with retry logic
        self.client = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(
                    url=f"http://{self.host}:{self.port}",
                    timeout=30,
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
                    processing_failures.labels(stage="qdrant_init").inc()
                    raise
                time.sleep(2)

        # Initialize embedding model
        try:
            self.embedder = SentenceTransformer(self.model_name)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"[QDRANT] ✅ Embedding model '{self.model_name}' loaded with {self.embedding_dim} dimensions")
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to load embedding model: {e}")
            processing_failures.labels(stage="model_load").inc()
            raise
            
        # Create model-specific collection name
        self.collection_name = f"{self.base_collection_name}_{self.embedding_dim}d"
        logger.info(f"[QDRANT] Using collection: {self.collection_name}")

    def _collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            collections_response = self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            return self.collection_name in existing_collections
        except Exception as e:
            logger.warning(f"[QDRANT] Failed to check collection existence: {e}")
            return False

    def _delete_collection(self) -> bool:
        """Delete collection if it exists."""
        try:
            if self._collection_exists():
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"[QDRANT] Deleted existing collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.warning(f"[QDRANT] Failed to delete collection: {e}")
            return False

    async def create_collection_safe(self):
        """Create collection with correct dimensions for the embedding model."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if collection exists with correct dimensions
                if self._collection_exists():
                    logger.info(f"[QDRANT] Collection '{self.collection_name}' exists ✅")
                    return True
                
                logger.info(f"[QDRANT] Creating collection '{self.collection_name}' with {self.embedding_dim} dimensions...")
                
                # Create new collection with correct dimensions
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"[QDRANT] ✅ Collection '{self.collection_name}' created successfully with {self.embedding_dim} dimensions")
                return True
                
            except Exception as e:
                logger.warning(f"[QDRANT] Collection creation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"[QDRANT] ❌ Failed to create collection after {max_retries} attempts")
                    processing_failures.labels(stage="collection_create").inc()
                    return False

    async def upsert_chunks(self, chunks: List[Dict], document_id: str) -> float:
        """Safe upsert with comprehensive error handling and batching."""
        if not chunks:
            logger.warning("[QDRANT] No chunks provided for upsert.")
            return 0.0

        logger.info(f"[QDRANT] Embedding and upserting {len(chunks)} chunks for '{document_id}'...")
        logger.info(f"[QDRANT] Using model: {self.model_name} ({self.embedding_dim} dimensions)")

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
                    
                    # Verify vector dimension
                    if len(vector) != self.embedding_dim:
                        logger.error(f"[QDRANT] ❌ Vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}")
                        processing_failures.labels(stage="vector_dimension").inc()
                        raise ValueError(f"Vector dimension mismatch: expected {self.embedding_dim}, got {len(vector)}")
                    
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
                            # Include chunking metadata if available
                            "chunking_metadata": chunk.get("chunking_metadata", {}),
                            # Include model info
                            "embedding_model": self.model_name,
                            "embedding_dim": self.embedding_dim,
                        }
                    )
                    batch_points.append(point)

                if batch_points:
                    # Record storage start time
                    start_storage = time.time()
                    
                    # Upsert batch
                    upsert_result = self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch_points,
                        wait=True
                    )
                    
                    # Record storage time
                    storage_time = time.time() - start_storage
                    storage_time_hist.observe(storage_time)
                    
                    total_points += len(batch_points)
                    logger.info(f"[QDRANT] ✅ Batch upserted {len(batch_points)} chunks")

            embedding_time = time.time() - start_embed
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Update metrics
            try:
                embedding_time_hist.observe(embedding_time)
                memory_usage_bytes.set(peak)
                ingest_chunks.inc(total_points)
                
                # Calculate and track costs
                try:
                    # Get current points count for cost calculation
                    try:
                        count_result = self.client.count(self.collection_name)
                        points_count = count_result.count if count_result else total_points
                    except:
                        points_count = total_points
                    
                    # Calculate and track costs
                    drift_detector.calculate_cost_metrics(
                        embedding_time=embedding_time,
                        document_length=sum(len(c.get("text", "")) for c in chunks),
                        points_count=points_count,
                        model_name=self.model_name
                    )
                    
                    # Update vector index size and points count metrics
                    vector_index_size_bytes.set(points_count * self.embedding_dim * 4)  # Approximate size
                    qdrant_points_count.set(points_count)
                    
                except Exception as cost_err:
                    logger.warning(f"[COST] Failed to track costs: {cost_err}")
                    
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
            processing_failures.labels(stage="upsert").inc()
            raise

    async def ingest_chunks(self, chunks: List[Dict]) -> bool:
        """Robust ingestion with comprehensive error handling."""
        try:
            document_id = chunks[0].get("document_id", "unknown_document") if chunks else "unknown_document"
            logger.info(f"[QDRANT] 🚀 Starting ingestion for {len(chunks)} chunks (document: {document_id})")
            logger.info(f"[QDRANT] Using model: {self.model_name} ({self.embedding_dim} dimensions)")
            
            # Step 1: Ensure collection exists
            logger.info("[QDRANT] 📝 Step 1/3: Verifying collection...")
            collection_ok = await self.create_collection_safe()
            if not collection_ok:
                logger.error("[QDRANT] ❌ Collection setup failed")
                processing_failures.labels(stage="collection_setup").inc()
                return False
            
            # Step 2: Upsert chunks
            logger.info("[QDRANT] 📝 Step 2/3: Upserting chunks...")
            try:
                embedding_time = await self.upsert_chunks(chunks, document_id)
            except Exception as e:
                logger.error(f"[QDRANT] ❌ Upsert failed: {e}")
                processing_failures.labels(stage="upsert_chunks").inc()
                return False
            
            # Step 3: Verify success
            logger.info("[QDRANT] 📝 Step 3/3: Verifying ingestion...")
            try:
                # Simple verification - try to count points
                count_result = self.client.count(self.collection_name)
                points_count = count_result.count if count_result else "unknown"
                logger.info(f"[QDRANT] ✅ Total points in collection: {points_count}")
                
                # Update metrics
                if isinstance(points_count, int):
                    qdrant_points_count.set(points_count)
                    vector_index_size_bytes.set(points_count * self.embedding_dim * 4)  # Approximate size
                    
            except Exception as e:
                logger.warning(f"[QDRANT] ⚠️ Could not verify points count: {e}")
                points_count = "unknown"
            
            logger.info(f"[QDRANT] 🎉 INGESTION COMPLETE")
            logger.info(f"  • Document: {document_id}")
            logger.info(f"  • Chunks ingested: {len(chunks)}")
            logger.info(f"  • Total points: {points_count}")
            logger.info(f"  • Embedding time: {embedding_time:.2f}s")
            logger.info(f"  • Model: {self.model_name} ({self.embedding_dim} dimensions)")
            
            return True
            
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Failed to ingest chunks: {e}")
            processing_failures.labels(stage="ingestion").inc()
            return False

    def search(self, query: str, top_k: int = 10, score_threshold: float = 0.3) -> List[Dict]:
        """Safe search with error handling and metrics tracking."""
        start_time = time.time()
        try:
            if not query or not query.strip():
                logger.warning("[QDRANT] Empty query provided for search")
                # Record search duration even for empty queries
                search_duration_hist.observe(time.time() - start_time)
                return []

            logger.info(f"[QDRANT] 🔍 Searching for '{query}' (top_k={top_k})")
            logger.info(f"[QDRANT] Using collection: {self.collection_name}")

            # Encode query
            query_vec = self.embedder.encode(query).tolist()
            top_k = min(top_k, 50)  # Safety cap

            # Search with score threshold
            search_results_raw = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=score_threshold,
            )

            results = []
            scores = []
            for hit in search_results_raw:
                payload = hit.payload or {}
                score = round(hit.score, 4)
                scores.append(score)
                
                # Record similarity score for monitoring
                embedding_similarity.observe(score)
                
                results.append({
                    "chunk_text": payload.get("text", ""),
                    "document_id": payload.get("document_id", ""),
                    "citation_id": payload.get("citation_id", ""),
                    "type": payload.get("type", "Unknown"),
                    "score": score,
                    "metadata": {
                        "chunk_id": payload.get("chunk_id"),
                        "confidence": payload.get("confidence"),
                        "sentence_count": payload.get("sentence_count"),
                        "char_count": payload.get("char_count"),
                        "has_citations": payload.get("has_citations"),
                        "has_statutes": payload.get("has_statutes"),
                        "has_parties": payload.get("has_parties"),
                        "embedding_model": payload.get("embedding_model"),
                        "embedding_dim": payload.get("embedding_dim"),
                    },
                    # Include chunking metadata in search results
                    "chunking_metadata": payload.get("chunking_metadata", {})
                })

            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Record search metrics
            search_duration = time.time() - start_time
            search_duration_hist.observe(search_duration)
            search_results.inc(len(results))
            
            logger.info(f"[QDRANT] ✅ Retrieved {len(results)} results for query '{query}' in {search_duration:.2f}s")
            return results

        except Exception as e:
            logger.error(f"[QDRANT] ❌ Search failed: {e}")
            processing_failures.labels(stage="search").inc()
            # Record search duration even for failed requests
            search_duration_hist.observe(time.time() - start_time)
            return []

    async def health_check(self) -> bool:
        """Health check with robust error handling."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"[QDRANT] ❌ Health check failed: {e}")
            processing_failures.labels(stage="health_check").inc()
            return False

    async def get_collection_info(self) -> Dict:
        """Get collection info."""
        try:
            if not self._collection_exists():
                return {"error": "Collection does not exist"}
                
            info = self.client.get_collection(self.collection_name)
            
            # Update metrics with current collection info
            points_count = getattr(info, 'points_count', 0)
            if points_count:
                qdrant_points_count.set(points_count)
                vector_index_size_bytes.set(points_count * self.embedding_dim * 4)  # Approximate size
                
            return {
                "name": self.collection_name,
                "vector_size": getattr(info.config.params.vectors, 'size', 0),
                "distance": getattr(info.config.params.vectors, 'distance', 'unknown'),
                "points_count": points_count,
                "status": getattr(info, 'status', 'unknown'),
                "embedding_model": self.model_name,
                "embedding_dim": self.embedding_dim,
            }
        except Exception as e:
            logger.warning(f"[QDRANT] Failed to get collection info: {e}")
            processing_failures.labels(stage="collection_info").inc()
            return {"error": str(e)}
