import re
import logging
from typing import List, Dict
from collections import Counter
from dataclasses import dataclass

# ✅ Correct imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
# 🧠 Semantic Legal Chunker (optimized with window + size control)
# ============================================================

class SemanticLegalChunker:
    """
    Uses LangChain’s SemanticChunker with size bounding and sliding window context.
    """

    def __init__(
        self,
        min_chunk_size: int = 800,       # avoid small fragments
        max_chunk_size: int = 1800,      # prevent too-long reasoning blocks
        chunk_overlap: int = 250,        # sliding window to preserve continuity
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            min_chunk_size: Minimum character count per chunk.
            max_chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlap between adjacent chunks for contextual flow.
            embedding_model: Semantic embedding model name.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # ✅ LangChain-compatible embedding model
        self.embedder = HuggingFaceEmbeddings(model_name=embedding_model)
        self.classifier = LegalChunkClassifier()

        logger.info(
            f"[INIT] SemanticLegalChunker(semantic_split, "
            f"min={min_chunk_size}, max={max_chunk_size}, overlap={chunk_overlap}, model={embedding_model})"
        )

    # --------------------------------------------------------
    def _semantic_split(self, text: str) -> List[str]:
        """Perform size-controlled semantic splitting."""
        splitter = SemanticChunker(self.embedder)

        # LangChain’s semantic splitter doesn’t directly expose size limits,
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

        logger.info("[CHUNKER] Starting semantic chunking...")
        chunks = self._semantic_split(text)
        logger.info(f"[CHUNKER] ✅ Generated {len(chunks)} legal-aware chunks.")

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

        logger.info(
            f"[CHUNKER] ✅ Finalized {len(classified_chunks)} chunks "
            f"(avg size {avg_size:.0f} chars) | Type dist: {dict(type_dist)}"
        )

        return classified_chunks
