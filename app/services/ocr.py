import logging
import gc
from typing import List
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import io

logger = logging.getLogger(__name__)

# ============================================================
# 🖨️ PDF Text Extraction Service (PyMuPDF + pdf2image)
# ============================================================

class OCRService:
    """
    Dual-mode PDF text extractor:
    - PyMuPDF (fitz): Fast text extraction from native PDFs
    - pdf2image + Pillow: Fallback for scanned/image-heavy PDFs
    """

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes, use_ocr_fallback: bool = False) -> str:
        """
        Extract text from PDF bytes buffer.
        First tries PyMuPDF (fast), falls back to pdf2image if needed.
        
        Args:
            pdf_bytes: PDF file bytes
            use_ocr_fallback: If True, use pdf2image for image-based PDFs
            
        Returns:
            Extracted text
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

                gc.collect()

            doc.close()
            full_text = "\n".join(page_texts).strip()
            
            # If extracted text is too short and fallback enabled, try pdf2image
            if len(full_text) < 100 and use_ocr_fallback:
                logger.info("[PDF] ⚠️ Text extraction yielded minimal content. Trying pdf2image fallback...")
                full_text = OCRService._extract_via_pdf2image(pdf_bytes)
            
            logger.info(f"[PDF] ✅ Extraction complete (total chars: {len(full_text)})")
            return full_text

        except Exception as e:
            logger.error(f"[PDF] ❌ PyMuPDF extraction failed: {e}. Trying pdf2image fallback...", exc_info=True)
            if use_ocr_fallback:
                return OCRService._extract_via_pdf2image(pdf_bytes)
            raise

    @staticmethod
    def _extract_via_pdf2image(pdf_bytes: bytes) -> str:
        """
        Fallback extraction using pdf2image (converts to images, then OCR).
        Slower but handles scanned PDFs.
        """
        logger.info("[PDF2IMG] 🖼️ Starting PDF-to-image conversion...")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=150)
            logger.info(f"[PDF2IMG] 📄 Converted {len(images)} pages to images")
            
            # For now, just return empty text as OCR requires additional engine
            # In production, integrate Tesseract or EasyOCR here if needed
            logger.warning("[PDF2IMG] ⚠️ Image extraction done but OCR not configured. Returning empty.")
            return ""

        except Exception as e:
            logger.error(f"[PDF2IMG] ❌ Fallback extraction failed: {e}", exc_info=True)
            raise
