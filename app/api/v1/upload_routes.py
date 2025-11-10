import os
import io
import pytesseract
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
from app.core.pipeline import ingest_legal_document

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text via OCR, and ingest it into Qdrant.
    """
    try:
        # =========================
        # 1️⃣ Validate file type
        # =========================
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        logger.info(f"[UPLOAD] Received PDF: {file.filename}")

        # =========================
        # 2️⃣ Read file bytes
        # =========================
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # =========================
        # 3️⃣ Convert PDF pages to images
        # =========================
        try:
            images = convert_from_bytes(pdf_bytes)
        except Exception as e:
            logger.error(f"[OCR] Failed to convert PDF: {e}")
            raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")

        if not images:
            raise HTTPException(status_code=400, detail="No readable pages found in the PDF")

        # =========================
        # 4️⃣ Perform OCR on each page
        # =========================
        text_chunks = []
        for i, page in enumerate(images, start=1):
            ocr_text = pytesseract.image_to_string(page)
            if ocr_text.strip():
                text_chunks.append(ocr_text)
            logger.info(f"[OCR] Processed page {i} → {len(ocr_text.strip())} chars")

        text = "\n".join(text_chunks).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # =========================
        # 5️⃣ Ingest OCR text into pipeline
        # =========================
        result = await ingest_legal_document(
            text=text,
            chunking_strategy="semantic-legal",
            document_id=file.filename,
        )

        logger.info(f"[UPLOAD] Ingestion complete for {file.filename}: {result['total_chunks']} chunks")

        # =========================
        # ✅ Return structured response
        # =========================
        return {
            "filename": file.filename,
            "pages": len(images),
            "total_chunks": result["total_chunks"],
            "status": "success",
            "message": "✅ PDF uploaded, OCR completed, and document ingested successfully!"
        }

    except HTTPException:
        raise  # Re-raise FastAPI HTTP errors directly

    except Exception as e:
        logger.exception(f"[ERROR] Unexpected failure during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
