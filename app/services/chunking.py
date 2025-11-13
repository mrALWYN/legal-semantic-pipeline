import re
import time
import logging
import os
import warnings
from typing import List, Dict
from collections import Counter
from dataclasses import dataclass

# ✅ Correct imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Internal imports
from app.services.metrics import (
    ingest_chunks, 
    embedding_time_hist,
    chunk_size_distribution,
    anomalies_detected,
    processing_failures
)
from app.services.mlflow_service import (
    start_run, log_model_metadata, log_metrics, log_params, end_run
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================
# 🎯 Available Chunking Models
# ============================================================

AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, lightweight (384 dims)",
        "dimensions": 384,
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "description": "High quality (768 dims)",
        "dimensions": 768,
    },
    "paraphrase-MiniLM-L12-v2": {
        "name": "paraphrase-MiniLM-L12-v2",
        "description": "Paraphrase model (384 dims)",
        "dimensions": 384,
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "description": "QA optimized (384 dims)",
        "dimensions": 384,
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# ⚖️ Lightweight Legal Chunk Classifier
# ============================================================

@dataclass
class ChunkClassification:
    category: str
    confidence: float


class LegalChunkClassifier:
    """Heuristic classifier to label chunks by legal context (facts, reasoning, etc.)"""

    def __init__(self):
        self.category_patterns = {
            "Facts": [r"\bfacts\b", r"\bbackground\b", r"\bincident\b", r"\bchronology\b"],
            "Arguments_Petitioner": [r"\bpetitioner\b", r"\bappellant\b", r"\bargues\b", r"\bcontends\b"],
            "Arguments_Respondent": [r"\brespondent\b", r"\bdefendant\b", r"\bsubmits\b"],
            "Precedent_Analysis": [r"\bheld in\b", r"\breliance\b", r"\bcase law\b", r"\bSCC\b"],
            "Section_Analysis": [r"\bSection\b", r"\bArticle\b", r"\bprovision\b", r"\bAct\b"],
            "Issues": [r"\bissue\b", r"\bquestion of law\b", r"\bwhether\b"],
            "Court_Reasoning": [r"\bCourt\b", r"\bfinds\b", r"\bview\b", r"\bopinion\b", r"\bwe hold\b"],
            "Conclusion": [r"\ballowed\b", r"\bdismissed\b", r"\bdisposed of\b", r"\bfinal order\b"],
        }

    def classify_chunk(self, text: str) -> ChunkClassification:
        scores = {cat: 0 for cat in self.category_patterns}
        text_lower = text.lower()

        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    scores[category] += len(matches)

        if max(scores.values()) == 0:
            return ChunkClassification(category="Court_Reasoning", confidence=0.1)

        top_cat, score = max(scores.items(), key=lambda x: x[1])
        total = sum(scores.values())
        confidence = round(score / total if total else 1.0, 3)
        return ChunkClassification(category=top_cat, confidence=confidence)


# ============================================================
# 🧠 Semantic Legal Chunker (multi-model support)
# ============================================================

class SemanticLegalChunker:
    """
    Uses LangChain's SemanticChunker with size bounding and sliding window context.
    Supports multiple sentence-transformer models.
    """

    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        chunk_overlap: int,
        embedding_model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            min_chunk_size: Minimum character count per chunk.
            max_chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlap between adjacent chunks for contextual flow.
            embedding_model: Model name from AVAILABLE_MODELS.
        """
        # Normalize model name (handle both formats)
        normalized_model = embedding_model.replace("sentence-transformers/", "")
        full_model_name = f"sentence-transformers/{normalized_model}" if not embedding_model.startswith("sentence-transformers/") else embedding_model
        
        if normalized_model in [m.replace("sentence-transformers/", "") for m in AVAILABLE_MODELS.keys()]:
            # Find the full model name
            for key in AVAILABLE_MODELS.keys():
                if normalized_model in key:
                    full_model_name = key
                    break
        
        if full_model_name not in AVAILABLE_MODELS:
            logger.warning(f"[INIT] Model {full_model_name} not available. Using {DEFAULT_MODEL}")
            full_model_name = DEFAULT_MODEL

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = full_model_name
        self.model_config = AVAILABLE_MODELS[full_model_name]

        # ✅ Use local cached models via HF_HOME environment variable
        logger.info(f"[CHUNKER] 🚀 Initializing embedding model: {full_model_name}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            self.embedder = HuggingFaceEmbeddings(model_name=full_model_name)
        
        self.classifier = LegalChunkClassifier()

        logger.info(
            f"[CHUNKER] ✅ SemanticLegalChunker initialized | "
            f"min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap}, "
            f"model={full_model_name} ({self.model_config['description']})"
        )

    # --------------------------------------------------------
    def _semantic_split(self, text: str) -> List[str]:
        """Perform size-controlled semantic splitting."""
        logger.info(f"[CHUNKER] 📊 Starting semantic split | text_length: {len(text)} chars")
        
        splitter = SemanticChunker(self.embedder)

        # LangChain's semantic splitter doesn't directly expose size limits,
        # so we manually re-merge chunks post-split.
        logger.info("[CHUNKER] 🔄 Creating documents with semantic splitter...")
        docs = splitter.create_documents([text])
        raw_chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]

        logger.info(f"[CHUNKER] 📋 Semantic split produced {len(raw_chunks)} raw chunks")
        
        if not raw_chunks:
            logger.warning("[CHUNKER] ⚠️ No chunks produced from semantic splitter")
            anomalies_detected.labels(type="no_chunks_produced").inc()
            return [text[:self.max_chunk_size]]

        # Merge small chunks to meet size requirements
        logger.info("[CHUNKER] 🔄 Merging chunks to meet size requirements...")
        merged_chunks = []
        buffer = ""

        for i, chunk in enumerate(raw_chunks):
            logger.debug(f"[CHUNKER] Processing raw chunk {i+1}/{len(raw_chunks)} | size: {len(chunk)}")
            
            if len(buffer) + len(chunk) <= self.max_chunk_size:
                buffer += " " + chunk
                logger.debug(f"[CHUNKER]   Added to buffer | buffer size: {len(buffer)}")
            else:
                if len(buffer) >= self.min_chunk_size:
                    merged_chunks.append(buffer.strip())
                    logger.debug(f"[CHUNKER]   ✅ Saved merged chunk {len(merged_chunks)} | size: {len(buffer)}")
                else:
                    logger.debug(f"[CHUNKER]   ⚠️ Buffer too small ({len(buffer)}), discarding")
                    anomalies_detected.labels(type="small_buffer_discarded").inc()
                buffer = chunk

        if len(buffer) >= self.min_chunk_size:
            merged_chunks.append(buffer.strip())
            logger.debug(f"[CHUNKER] ✅ Saved final buffer chunk | size: {len(buffer)}")
        elif buffer and merged_chunks:
            # Append remaining text to last chunk if it's too small
            merged_chunks[-1] += " " + buffer
            logger.debug(f"[CHUNKER] 🔄 Appended remaining text to last chunk")

        # Add sliding window overlap for continuity
        logger.info("[CHUNKER] 🔄 Applying sliding window overlap...")
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            window_text = chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_tail = merged_chunks[i - 1][-self.chunk_overlap:]
                window_text = prev_tail + "\n" + chunk
            final_chunks.append(window_text.strip())
            logger.debug(f"[CHUNKER]   Chunk {i+1}: {len(window_text)} chars")

        logger.info(f"[CHUNKER] ✅ Semantic split complete | {len(final_chunks)} final chunks")
        return final_chunks

    # --------------------------------------------------------
    def chunk_document(self, text: str, document_id: str) -> List[Dict]:
        """Perform semantic chunking, classification, and metadata enrichment."""
        start_time = time.time()
        logger.info(f"[CHUNKER] 🚀 Starting chunking for document: {document_id}")
        
        if not text or not isinstance(text, str):
            logger.warning("[CHUNKER] ❌ Empty or invalid text input.")
            processing_failures.labels(stage="chunking_input_validation").inc()
            return []

        # Step 1: Perform semantic splitting
        logger.info(f"[CHUNKER] 📝 Step 1/3: Semantic splitting...")
        chunk_start_time = time.time()
        try:
            chunks = self._semantic_split(text)
        except Exception as e:
            logger.error(f"[CHUNKER] ❌ Semantic splitting failed: {e}")
            processing_failures.labels(stage="semantic_splitting").inc()
            return []
        chunk_time = time.time() - chunk_start_time
        logger.info(f"[CHUNKER] ✅ Step 1 Complete | {len(chunks)} chunks in {chunk_time:.2f}s")

        if not chunks:
            return []

        # Step 2: Classify and enrich chunks
        logger.info(f"[CHUNKER] 📝 Step 2/3: Classifying and enriching chunks...")
        classified_chunks = []
        classification_start = time.time()
        
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                logger.debug(f"[CHUNKER]   Skipping empty chunk {i+1}")
                continue
            
            logger.debug(f"[CHUNKER]   Classifying chunk {i+1}/{len(chunks)} | size: {len(chunk_text)}")
            try:
                classification = self.classifier.classify_chunk(chunk_text)
            except Exception as e:
                logger.error(f"[CHUNKER] ❌ Classification failed for chunk {i+1}: {e}")
                processing_failures.labels(stage="chunk_classification").inc()
                continue
            
            chunk_data = {
                "chunk_id": f"{document_id}_{i}",
                "document_id": document_id,
                "text": chunk_text,
                "type": classification.category,
                "char_count": len(chunk_text),
                "sentence_count": len(re.split(r"[.!?]", chunk_text)),
                "confidence": classification.confidence,
                "has_citations": bool(re.search(r"\(\d{4}\)\s+\d+\s+(?:SCC|AIR)", chunk_text)),
                "has_statutes": bool(re.search(r"\b(?:Section|Article)\s+\d+", chunk_text)),
                "has_parties": bool(re.search(r"\b(?:petitioner|respondent|appellant|defendant)\b", chunk_text, re.IGNORECASE)),
                # Add chunking metadata
                "chunking_metadata": {
                    "chunking_technique": "semantic",
                    "embedding_model": self.model_name,
                    "model_dimensions": self.model_config["dimensions"],
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            }
            classified_chunks.append(chunk_data)
            
            # Record chunk size for monitoring
            chunk_size_distribution.observe(len(chunk_text))
            
            # Progress update every 10 chunks
            if (i + 1) % 10 == 0:
                logger.info(f"[CHUNKER]   ✅ Classified {i+1}/{len(chunks)} chunks")

        classification_time = time.time() - classification_start
        logger.info(f"[CHUNKER] ✅ Step 2 Complete | {len(classified_chunks)} classified in {classification_time:.2f}s")

        if not classified_chunks:
            logger.warning("[CHUNKER] ❌ No valid chunks produced after filtering.")
            anomalies_detected.labels(type="no_valid_chunks").inc()
            return []

        # Step 3: Calculate statistics and log metrics
        logger.info(f"[CHUNKER] 📝 Step 3/3: Calculating statistics and logging metrics...")
        avg_size = sum(c["char_count"] for c in classified_chunks) / len(classified_chunks)
        type_dist = Counter(c["type"] for c in classified_chunks)

        # Data quality checks
        if len(classified_chunks) < 1:
            anomalies_detected.labels(type="low_chunk_count").inc()
            
        if avg_size < 100:
            anomalies_detected.labels(type="small_chunks").inc()
        elif avg_size > 8000:
            anomalies_detected.labels(type="large_chunks").inc()

        # ✅ Update metrics + log to MLflow with model info
        ingest_chunks.inc(len(classified_chunks))
        embedding_time_hist.observe(chunk_time)

        # ✅ MLflow logging with proper error handling
        run = None
        try:
            run = start_run(experiment_name="chunking_experiments")
            if run:
                log_params({
                    "chunking_model": self.model_name,
                    "model_dimensions": self.model_config["dimensions"],
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunking_technique": "semantic",
                })
                
                log_metrics({
                    "chunk_count": len(classified_chunks),
                    "chunk_time": chunk_time,
                    "classification_time": classification_time,
                    "avg_chunk_size": avg_size,
                })
                
                # Log type distribution as metrics
                for t, v in type_dist.items():
                    log_metrics({f"type_{t}_count": v})
                
                logger.info("[CHUNKER] ✅ MLflow metrics logged successfully")
                
        except Exception as e:
            logger.warning(f"[CHUNKER] ⚠️ MLflow logging failed: {e}")
            processing_failures.labels(stage="mlflow_logging").inc()
        finally:
            if run:
                end_run()

        total_time = time.time() - start_time
        logger.info(
            f"[CHUNKER] 🎉 CHUNKING COMPLETE\n"
            f"  • Document: {document_id}\n"
            f"  • Total chunks: {len(classified_chunks)}\n"
            f"  • Avg chunk size: {avg_size:.0f} chars\n"
            f"  • Type distribution: {dict(type_dist)}\n"
            f"  • Total time: {total_time:.2f}s\n"
            f"  • Model: {self.model_name}\n"
            f"  • Chunking technique: semantic"
        )

        return classified_chunks


# ============================================================
# 🧩 Recursive Chunker (multi-model support)
# ============================================================

class RecursiveChunker:
    """
    Simple paragraph/sentence-based recursive chunker with multi-model support.
    """

    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        chunk_overlap: int,
        embedding_model: str = DEFAULT_MODEL
    ):
        # Normalize model name (handle both formats)
        normalized_model = embedding_model.replace("sentence-transformers/", "")
        full_model_name = f"sentence-transformers/{normalized_model}" if not embedding_model.startswith("sentence-transformers/") else embedding_model
        
        if normalized_model in [m.replace("sentence-transformers/", "") for m in AVAILABLE_MODELS.keys()]:
            # Find the full model name
            for key in AVAILABLE_MODELS.keys():
                if normalized_model in key:
                    full_model_name = key
                    break
        
        if full_model_name not in AVAILABLE_MODELS:
            logger.warning(f"[INIT] Model {full_model_name} not available. Using {DEFAULT_MODEL}")
            full_model_name = DEFAULT_MODEL

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = full_model_name
        self.model_config = AVAILABLE_MODELS[full_model_name]
        self.classifier = LegalChunkClassifier()

        logger.info(
            f"[CHUNKER] ✅ RecursiveChunker initialized | "
            f"min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap}, "
            f"model={full_model_name} ({self.model_config['description']})"
        )

    # --------------------------------------------------------
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text by paragraphs or sentences within size bounds."""
        logger.debug(f"[CHUNKER-RECURSIVE] Recursive split | text length: {len(text)}")
        
        if len(text) <= self.max_chunk_size:
            return [text.strip()] if text.strip() else []

        parts = [p for p in text.split("\n\n") if p.strip()]
        if len(parts) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, buffer = [], ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(buffer) + len(s) <= self.max_chunk_size:
                    buffer += (" " if buffer else "") + s
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = s
            if buffer:
                chunks.append(buffer.strip())
            return chunks

        result = []
        for p in parts:
            result += self._recursive_split(p)
        return result

    def chunk_document(self, text: str, document_id: str) -> List[Dict]:
        start_time = time.time()
        logger.info(f"[CHUNKER] 🚀 Starting recursive chunking for document: {document_id}")
        
        if not text or not isinstance(text, str):
            logger.warning("[CHUNKER-RECURSIVE] ❌ Empty or invalid text input")
            processing_failures.labels(stage="chunking_input_validation").inc()
            return []

        # Step 1: Recursive splitting
        logger.info(f"[CHUNKER] 📝 Step 1/3: Recursive splitting...")
        chunk_start_time = time.time()
        try:
            chunks = self._recursive_split(text)
        except Exception as e:
            logger.error(f"[CHUNKER-RECURSIVE] ❌ Recursive splitting failed: {e}")
            processing_failures.labels(stage="recursive_splitting").inc()
            return []
        chunk_time = time.time() - chunk_start_time
        
        if not chunks:
            logger.warning("[CHUNKER-RECURSIVE] ❌ No chunks produced")
            anomalies_detected.labels(type="no_chunks_produced").inc()
            return []
        
        logger.info(f"[CHUNKER] ✅ Step 1 Complete | {len(chunks)} raw chunks in {chunk_time:.2f}s")

        # Step 2: Apply overlap and classify
        logger.info(f"[CHUNKER] 📝 Step 2/3: Applying overlap and classifying...")
        final_chunks = []
        classification_start = time.time()
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk
            if i > 0 and self.chunk_overlap > 0:
                tail = chunks[i - 1][-self.chunk_overlap:]
                chunk_text = tail + "\n" + chunk
            
            try:
                classification = self.classifier.classify_chunk(chunk_text)
            except Exception as e:
                logger.error(f"[CHUNKER] ❌ Classification failed for chunk {i+1}: {e}")
                processing_failures.labels(stage="chunk_classification").inc()
                continue
                
            chunk_data = {
                "chunk_id": f"{document_id}_{i}",
                "document_id": document_id,
                "text": chunk_text,
                "type": classification.category,
                "char_count": len(chunk_text),
                "sentence_count": len(re.split(r"[.!?]", chunk_text)),
                "confidence": classification.confidence,
                "has_citations": bool(re.search(r"\(\d{4}\)\s+\d+\s+(?:SCC|AIR)", chunk_text)),
                "has_statutes": bool(re.search(r"\b(?:Section|Article)\s+\d+", chunk_text)),
                "has_parties": bool(re.search(r"\b(?:petitioner|respondent|appellant|defendant)\b", chunk_text, re.IGNORECASE)),
                # Add chunking metadata
                "chunking_metadata": {
                    "chunking_technique": "recursive",
                    "embedding_model": self.model_name,
                    "model_dimensions": self.model_config["dimensions"],
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                }
            }
            final_chunks.append(chunk_data)
            
            # Record chunk size for monitoring
            chunk_size_distribution.observe(len(chunk_text))
            
            # Progress update every 20 chunks
            if (i + 1) % 20 == 0:
                logger.info(f"[CHUNKER]   ✅ Processed {i+1}/{len(chunks)} chunks")

        classification_time = time.time() - classification_start
        logger.info(f"[CHUNKER] ✅ Step 2 Complete | {len(final_chunks)} classified in {classification_time:.2f}s")

        # Step 3: Calculate statistics and log metrics
        logger.info(f"[CHUNKER] 📝 Step 3/3: Calculating statistics and logging metrics...")
        avg_size = sum(c["char_count"] for c in final_chunks) / len(final_chunks) if final_chunks else 0
        type_dist = Counter(c["type"] for c in final_chunks)

        # Data quality checks
        if len(final_chunks) < 1:
            anomalies_detected.labels(type="low_chunk_count").inc()
            
        if avg_size < 100:
            anomalies_detected.labels(type="small_chunks").inc()
        elif avg_size > 8000:
            anomalies_detected.labels(type="large_chunks").inc()

        # ✅ Metrics + MLflow with model info
        ingest_chunks.inc(len(final_chunks))
        embedding_time_hist.observe(chunk_time)
        
        # ✅ MLflow logging with proper error handling
        run = None
        try:
            run = start_run(experiment_name="chunking_experiments")
            if run:
                log_params({
                    "chunking_model": self.model_name,
                    "model_dimensions": self.model_config["dimensions"],
                    "chunking_technique": "recursive",
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                })
                
                log_metrics({
                    "chunk_count": len(final_chunks),
                    "chunk_time": chunk_time,
                    "classification_time": classification_time,
                    "avg_chunk_size": avg_size,
                })
                
                # Log type distribution as metrics
                for t, v in type_dist.items():
                    log_metrics({f"type_{t}_count": v})
                    
                logger.info("[CHUNKER] ✅ MLflow metrics logged successfully")
                
        except Exception as e:
            logger.warning(f"[CHUNKER] ⚠️ MLflow logging failed: {e}")
            processing_failures.labels(stage="mlflow_logging").inc()
        finally:
            if run:
                end_run()

        total_time = time.time() - start_time
        logger.info(
            f"[CHUNKER] 🎉 RECURSIVE CHUNKING COMPLETE\n"
            f"  • Document: {document_id}\n"
            f"  • Total chunks: {len(final_chunks)}\n"
            f"  • Avg chunk size: {avg_size:.0f} chars\n"
            f"  • Type distribution: {dict(type_dist)}\n"
            f"  • Total time: {total_time:.2f}s\n"
            f"  • Model: {self.model_name}\n"
            f"  • Chunking technique: recursive"
        )

        return final_chunks
