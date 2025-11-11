import re
import logging
from typing import List, Dict, Optional, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Logger Setup
# ============================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SemanticChunker:
    """
    Legal-aware semantic chunker with global sliding window continuity.

    Features:
      ✅ Multiple strategies (semantic-legal, fixed, hybrid)
      ✅ Sliding window continuity applied to all chunks
      ✅ Auto-detects section headers
      ✅ Plug & play for RAG pipelines
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        chunk_size: int = 1000,
        overlap: int = 200,
        dynamic_markers: bool = True,
        custom_markers: Optional[List[str]] = None,
        preprocessor: Optional[Callable[[str], str]] = None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.dynamic_markers = dynamic_markers
        self.preprocessor = preprocessor or self._default_preprocessor

        # Default markers
        self.legal_section_markers = custom_markers or [
            r"\bFACTS\b",
            r"\bHELD\b",
            r"\bORDER\b",
            r"\bJUDGMENT\b",
            r"\bARGUMENTS\b",
            r"\bAPPEAL\b",
            r"\bBACKGROUND\b",
            r"\bCASE\s+SUMMARY\b",
            r"\bREASONS\b",
            r"\bANALYSIS\b",
            r"\bISSUE[S]?\b",
            r"\bDECISION\b",
            r"\bORDER\b",
        ]

        logger.info(
            f"[INIT] SmartLegalChunker(strategy={strategy}, chunk_size={chunk_size}, "
            f"overlap={overlap}, dynamic_markers={dynamic_markers})"
        )

    # ============================================================
    # Helpers
    # ============================================================
    def _default_preprocessor(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _detect_markers(self, text: str) -> List[str]:
        """Auto-detect section titles (CAPS or numbered)."""
        if not self.dynamic_markers:
            return self.legal_section_markers
        headings = re.findall(r"(?:^|\n)([A-Z][A-Z\s]{3,})", text)
        numbered = re.findall(r"(?:^|\n)(\d+\.\s+[A-Z][^\n]{3,})", text)
        auto_markers = list({h.strip() for h in headings + numbered})
        logger.debug(f"[AUTO-MARKERS] Detected {len(auto_markers)} headings.")
        return self.legal_section_markers + [re.escape(m) for m in auto_markers]

    def _split_by_legal_sections(self, text: str) -> List[str]:
        markers = self._detect_markers(text)
        pattern = re.compile("|".join(markers), flags=re.IGNORECASE)
        sections = re.split(pattern, text)
        return [s.strip() for s in sections if s.strip()]

    def _recursive_split(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n## ", "\n\n", "\n", ". ", " "],
            length_function=len,
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]

    def _apply_sliding_window(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping continuity windows across existing chunks.
        Each output chunk includes overlap context from the previous one.
        """
        if not chunks:
            return []

        final_chunks = []
        for i, chunk in enumerate(chunks):
            # For the first chunk — no overlap from before
            if i == 0:
                final_chunks.append(chunk)
                continue

            prev_tail = chunks[i - 1][-self.overlap:] if len(chunks[i - 1]) > self.overlap else chunks[i - 1]
            merged = (prev_tail + " " + chunk).strip()
            final_chunks.append(merged)

        logger.debug(f"[WINDOW] Applied sliding window continuity to {len(final_chunks)} chunks.")
        return final_chunks

    # ============================================================
    # Main entry
    # ============================================================
    def chunk_document(self, text: str) -> List[Dict]:
        """
        Generate chunks with global sliding continuity.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("[CHUNKER] Empty or invalid text provided.")
            return []

        text = self.preprocessor(text)
        logger.info(f"[CHUNKER] Using strategy: {self.strategy}")

        base_chunks: List[str] = []

        # Strategy 1: semantic-legal
        if self.strategy == "semantic-legal":
            for section in self._split_by_legal_sections(text):
                base_chunks.extend(self._recursive_split(section))

        # Strategy 2: fixed
        elif self.strategy == "fixed":
            base_chunks.extend(self._recursive_split(text))

        # Strategy 3: hybrid (structure + size)
        elif self.strategy == "hybrid":
            for section in self._split_by_legal_sections(text):
                if len(section) <= self.chunk_size:
                    base_chunks.append(section)
                else:
                    base_chunks.extend(self._recursive_split(section))
        else:
            logger.warning(f"[CHUNKER] Unknown strategy '{self.strategy}', falling back to fixed.")
            base_chunks.extend(self._recursive_split(text))

        # ✅ Apply global sliding continuity across all final chunks
        continuous_chunks = self._apply_sliding_window(base_chunks)

        # Build final metadata objects
        final = [
            {
                "chunk_id": str(i),
                "text": chunk,
                "char_count": len(chunk),
                "strategy_used": self.strategy,
            }
            for i, chunk in enumerate(continuous_chunks)
        ]

        logger.info(f"[CHUNKER] Generated {len(final)} total chunks (with sliding continuity).")
        return final
