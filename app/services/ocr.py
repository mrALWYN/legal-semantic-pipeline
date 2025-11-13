import logging
import gc
from typing import List
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ============================================================
# 🖨️ PDF text extraction using PyMuPDF (fast, no OCR)
# ============================================================
class OCRService:
    """
    Lightweight PDF text extractor using PyMuPDF.
    This replaces EasyOCR/Tesseract. It extracts embedded / selectable text.
    Scanned-image-only PDFs will not be OCR'd (by design per user request).
    """

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes, dpi: int = 150) -> str:
        """
        Extract text from a PDF bytes buffer using PyMuPDF.
        Returns concatenated page text.
        """
        logger.info("[PDF] 🔄 Starting PDF text extraction via PyMuPDF...")
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page_texts: List[str] = []
            total_pages = doc.page_count
            logger.info(f"[PDF] 📄 Opened PDF with {total_pages} pages")

            for i, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text")
                    page_texts.append(text or "")
                    logger.info(f"[PDF] Extracted text from page {i}/{total_pages} (chars: {len(text) if text else 0})")
                except Exception as e:
                    logger.warning(f"[PDF] ⚠️ Failed to extract page {i}: {e}")
                    page_texts.append("")

                # light cleanup per page
                gc.collect()

            doc.close()
            full_text = "\n".join(page_texts).strip()
            logger.info(f"[PDF] ✅ Extraction complete (total chars: {len(full_text)})")
            return full_text

        except Exception as e:
            logger.error(f"[PDF] ❌ Extraction failed: {e}", exc_info=True)
            raise
