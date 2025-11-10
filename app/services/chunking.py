import re
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Logger Setup
# ============================================================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SemanticChunker:
    """
    A semantic-aware text chunker for legal documents.

    It intelligently splits text into sections based on
    legal markers (like FACTS, HELD, JUDGMENT) or by
    recursive character-based splitting.
    """

    def __init__(self, strategy: str = "semantic-legal", chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the chunker with strategy and parameters.

        :param strategy: "semantic-legal" or "fixed-size"
        :param chunk_size: maximum characters per chunk
        :param overlap: overlap between chunks
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

        # ✅ FIX: No inline flags like (?i) — use re.IGNORECASE instead later
        self.legal_section_markers = [
            r"\bFACTS\b",
            r"\bHELD\b",
            r"\bJUDGMENT\b",
            r"\bORDER\b",
            r"\bARGUMENTS\b",
            r"\bAPPEAL\b",
            r"\bCASE\b",
            r"\bREASONS\b",
            r"\bANALYSIS\b",
        ]

        logger.info(f"[INIT] SemanticChunker initialized with strategy='{strategy}', "
                    f"chunk_size={chunk_size}, overlap={overlap}")

    # --------------------------------------------------------
    # Helper: Split by legal sections
    # --------------------------------------------------------
    def _split_by_legal_sections(self, text: str) -> List[str]:
        """
        Split text based on known legal section markers like 'FACTS', 'HELD', etc.
        """
        # ✅ FIX: Compile regex safely with IGNORECASE instead of using (?i)
        pattern = re.compile("|".join(self.legal_section_markers), flags=re.IGNORECASE)
        parts = re.split(pattern, text)
        sections = [p.strip() for p in parts if p.strip()]
        logger.debug(f"[CHUNKER] Split text into {len(sections)} legal sections")
        return sections

    # --------------------------------------------------------
    # Helper: Recursive text splitter
    # --------------------------------------------------------
    def _recursive_split(self, text: str) -> List[str]:
        """
        Split long sections recursively using LangChain's splitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n## ", "\n\n", "\n", ". ", " "],
            length_function=len,
        )
        chunks = splitter.split_text(text)
        chunks = [c.strip() for c in chunks if c.strip()]
        logger.debug(f"[CHUNKER] Recursively split text into {len(chunks)} chunks")
        return chunks

    # --------------------------------------------------------
    # Main entry point
    # --------------------------------------------------------
    def chunk_document(self, text: str) -> List[Dict]:
        """
        Main method to generate semantic chunks from a raw document.

        Returns a list of chunk dictionaries with:
          - chunk_id
          - text
          - char_count
        """
        logger.info(f"[CHUNKER] Starting chunking using strategy: {self.strategy}")

        # Clean text: normalize whitespace
        if not isinstance(text, str) or not text.strip():
            logger.warning("[CHUNKER] Empty or invalid text provided.")
            return []

        text = re.sub(r"\s+", " ", text).strip()
        all_chunks = []

        # Semantic (legal) strategy
        if self.strategy == "semantic-legal":
            sections = self._split_by_legal_sections(text)
            for i, section in enumerate(sections):
                sub_chunks = self._recursive_split(section)
                for j, chunk in enumerate(sub_chunks):
                    all_chunks.append({
                        "chunk_id": f"{i}-{j}",
                        "text": chunk,
                        "char_count": len(chunk)
                    })
            logger.info(f"[CHUNKER] Generated {len(all_chunks)} semantic chunks.")

        # Fixed-size fallback
        else:
            sub_chunks = self._recursive_split(text)
            for i, chunk in enumerate(sub_chunks):
                all_chunks.append({
                    "chunk_id": str(i),
                    "text": chunk,
                    "char_count": len(chunk)
                })
            logger.info(f"[CHUNKER] Generated {len(all_chunks)} fixed-size chunks.")

        return all_chunks