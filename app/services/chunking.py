import re
import time
import logging
import os
from typing import List, Dict
from collections import Counter
from dataclasses import dataclass
import torch

# ✅ Correct imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Internal imports
from app.services.metrics import ingest_chunks, embedding_time_hist
from app.services import mlflow_service

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
    Uses LangChain's SemanticChunker with GPU acceleration.
    Supports multiple sentence-transformer models.
    """

    def __init__(
        self,
        min_chunk_size,
        max_chunk_size,
        chunk_overlap,
        embedding_model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            min_chunk_size: Minimum character count per chunk.
            max_chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlap between adjacent chunks for contextual flow.
            embedding_model: Model name from AVAILABLE_MODELS.
        """
        if embedding_model not in AVAILABLE_MODELS:
            logger.warning(f"[INIT] Model {embedding_model} not available. Using {DEFAULT_MODEL}")
            embedding_model = DEFAULT_MODEL

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = embedding_model
        self.model_config = AVAILABLE_MODELS[embedding_model]

        # ✅ Use local cached models + GPU acceleration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[INIT] Using device: {device}")
        
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device}
        )
        self.classifier = LegalChunkClassifier()

        logger.info(
            f"[INIT] SemanticLegalChunker(semantic_split, "
            f"min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap}, "
            f"model={embedding_model} ({self.model_config['description']}), device={device})"
        )

    # --------------------------------------------------------
    def _semantic_split(self, text: str) -> List[str]:
        """Perform size-controlled semantic splitting."""
        splitter = SemanticChunker(self.embedder)

        # LangChain's semantic splitter doesn't directly expose size limits,
        # so we manually re-merge chunks post-split.
        docs = splitter.create_documents([text])
        raw_chunks = [d.page_content.strip() for d in docs if d.page_content.strip()]

        merged_chunks = []
        buffer = ""

        for chunk in raw_chunks:
            if len(buffer) + len(chunk) <= self.max_chunk_size:
                buffer += " " + chunk
            else:
                if len(buffer) >= self.min_chunk_size:
                    merged_chunks.append(buffer.strip())
                buffer = chunk

        if len(buffer) >= self.min_chunk_size:
            merged_chunks.append(buffer.strip())

        # Add sliding window overlap for continuity
        final_chunks = []
        for i, chunk in enumerate(merged_chunks):
            window_text = chunk
            if i > 0:
                prev_tail = merged_chunks[i - 1][-self.chunk_overlap :]
                window_text = prev_tail + "\n" + chunk
            final_chunks.append(window_text.strip())

        return final_chunks

    # --------------------------------------------------------
    def chunk_document(self, text: str, document_id: str) -> List[Dict]:
        """Perform semantic chunking, classification, and metadata enrichment."""
        if not text or not isinstance(text, str):
            logger.warning("[CHUNKER] Empty or invalid text input.")
            return []

        start_time = time.time()
        logger.info(f"[CHUNKER] Starting semantic chunking with model: {self.model_name}...")
        chunks = self._semantic_split(text)
        chunk_time = time.time() - start_time
        logger.info(f"[CHUNKER] ✅ Generated {len(chunks)} legal-aware chunks in {chunk_time:.2f}s")

        classified_chunks = []
        for i, chunk_text in enumerate(chunks):
            classification = self.classifier.classify_chunk(chunk_text)
            chunk_data = {
                "chunk_id": str(i),
                "document_id": document_id,
                "text": chunk_text,
                "type": classification.category,
                "char_count": len(chunk_text),
                "sentence_count": len(re.split(r"[.!?]", chunk_text)),
                "confidence": classification.confidence,
                "has_citations": bool(re.search(r"\(\d{4}\)\s+\d+\s+(?:SCC|AIR)", chunk_text)),
                "has_statutes": bool(re.search(r"\b(?:Section|Article)\s+\d+", chunk_text)),
                "has_parties": bool(re.search(r"\b(?:petitioner|respondent|appellant|defendant)\b", chunk_text, re.IGNORECASE)),
            }
            classified_chunks.append(chunk_data)

        if not classified_chunks:
            logger.warning("[CHUNKER] No valid chunks produced after filtering.")
            return []

        avg_size = sum(c["char_count"] for c in classified_chunks) / len(classified_chunks)
        type_dist = Counter(c["type"] for c in classified_chunks)

        # ✅ Update metrics + log to MLflow with model info
        ingest_chunks.inc(len(classified_chunks))
        embedding_time_hist.observe(chunk_time)

        try:
            run = mlflow_service.start_run(experiment_name="chunking_experiments")
            mlflow_service.log_metrics({
                "chunk_count": len(classified_chunks),
                "chunk_time": chunk_time,
                "avg_chunk_size": avg_size,
            })
            # ✅ Log which model was used
            mlflow_service.log_params({
                "chunking_model": self.model_name,
                "model_dimensions": self.model_config["dimensions"],
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "chunk_overlap": self.chunk_overlap,
            })
            for t, v in type_dist.items():
                mlflow_service.log_metrics({f"type_{t}_count": v})
            mlflow_service.log_model_metadata(
                model_name=self.model_name,
                embedding_dim=self.model_config["dimensions"],
                chunk_size=self.min_chunk_size,
                overlap=self.chunk_overlap,
            )
            mlflow_service.mlflow.end_run()
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log chunking experiment: {e}")

        logger.info(
            f"[CHUNKER] ✅ Finalized {len(classified_chunks)} chunks "
            f"(avg size {avg_size:.0f} chars) | Type dist: {dict(type_dist)} | Model: {self.model_name}"
        )

        return classified_chunks


# ============================================================
# 🧩 Recursive Chunker (multi-model support)
# ============================================================

class RecursiveChunker:
    """
    Simple paragraph/sentence-based recursive chunker with multi-model support.
    """

    def __init__(self, min_chunk_size, max_chunk_size, chunk_overlap, embedding_model: str = DEFAULT_MODEL):
        if embedding_model not in AVAILABLE_MODELS:
            embedding_model = DEFAULT_MODEL

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = embedding_model
        self.model_config = AVAILABLE_MODELS[embedding_model]
        self.classifier = LegalChunkClassifier()

    # --------------------------------------------------------
    def _recursive_split(self, text: str) -> List[str]:
        """Recursively split text by paragraphs or sentences within size bounds."""
        if len(text) <= self.max_chunk_size:
            return [text.strip()]

        parts = [p for p in text.split("\n\n") if p.strip()]
        if len(parts) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks, buffer = [], ""
            for s in sentences:
                if len(buffer) + len(s) <= self.max_chunk_size:
                    buffer += " " + s
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
        if not text or not isinstance(text, str):
            return []

        start_time = time.time()
        chunks = self._recursive_split(text)
        chunk_time = time.time() - start_time
        logger.info(f"[CHUNKER-RECURSIVE] Generated {len(chunks)} chunks in {chunk_time:.2f}s with model: {self.model_name}")

        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk
            if i > 0:
                tail = chunks[i - 1][-self.chunk_overlap:]
                chunk_text = tail + "\n" + chunk
            classification = self.classifier.classify_chunk(chunk_text)
            final_chunks.append({
                "chunk_id": str(i),
                "document_id": document_id,
                "text": chunk_text,
                "type": classification.category,
                "char_count": len(chunk_text),
                "sentence_count": len(re.split(r"[.!?]", chunk_text)),
                "confidence": classification.confidence,
                "has_citations": bool(re.search(r"\(\d{4}\)\s+\d+\s+(?:SCC|AIR)", chunk_text)),
                "has_statutes": bool(re.search(r"\b(?:Section|Article)\s+\d+", chunk_text)),
                "has_parties": bool(re.search(r"\b(?:petitioner|respondent|appellant|defendant)\b", chunk_text, re.IGNORECASE)),
            })

        avg_size = sum(c["char_count"] for c in final_chunks) / len(final_chunks)
        type_dist = Counter(c["type"] for c in final_chunks)

        # ✅ Metrics + MLflow with model info
        ingest_chunks.inc(len(final_chunks))
        embedding_time_hist.observe(chunk_time)
        try:
            run = mlflow_service.start_run(experiment_name="chunking_experiments")
            mlflow_service.log_metrics({
                "chunk_count": len(final_chunks),
                "chunk_time": chunk_time,
                "avg_chunk_size": avg_size,
            })
            mlflow_service.log_params({
                "chunking_model": self.model_name,
                "model_dimensions": self.model_config["dimensions"],
                "chunking_strategy": "recursive",
            })
            for t, v in type_dist.items():
                mlflow_service.log_metrics({f"type_{t}_count": v})
            mlflow_service.mlflow.end_run()
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log recursive chunking: {e}")

        logger.info(
            f"[CHUNKER-RECURSIVE] ✅ {len(final_chunks)} chunks | avg size {avg_size:.0f} | "
            f"dist: {dict(type_dist)} | Model: {self.model_name}"
        )
        return final_chunks
