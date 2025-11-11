import os
import io
import pytesseract
import logging
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pdf2image import convert_from_bytes
from app.core.pipeline import ingest_legal_document

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    citation_id: str = Form(None)
):
    """
    Upload a PDF → extract text via OCR → run semantic chunking pipeline → store in Qdrant.
    """
    try:
        # ===============================
        # 1️⃣ Validate file type
        # ===============================
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        logger.info(f"[UPLOAD] 📄 Received PDF: {file.filename}")

        # ===============================
        # 2️⃣ Read file bytes
        # ===============================
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # ===============================
        # 3️⃣ Convert PDF pages → Images
        # ===============================
        try:
            images = await asyncio.to_thread(convert_from_bytes, pdf_bytes)
        except Exception as e:
            logger.error(f"[OCR] ❌ Failed to convert PDF: {e}")
            raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")

        if not images:
            raise HTTPException(status_code=400, detail="No readable pages found in the PDF")

        logger.info(f"[OCR] 🖼️ Converted {len(images)} pages from PDF")

        # ===============================
        # 4️⃣ Perform OCR on each page
        # ===============================
        text_chunks = []
        for i, page in enumerate(images, start=1):
            ocr_text = await asyncio.to_thread(pytesseract.image_to_string, page)
            ocr_text = ocr_text.strip()
            if ocr_text:
                text_chunks.append(ocr_text)
                logger.info(f"[OCR] ✅ Page {i}: {len(ocr_text)} chars extracted")
            else:
                logger.warning(f"[OCR] ⚠️ Page {i} produced no readable text")

        text = "\n".join(text_chunks).strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # ===============================
        # 5️⃣ Ingest extracted text
        # ===============================
        document_id = citation_id or os.path.splitext(file.filename)[0]

        result = await ingest_legal_document(
            text=text,
            chunking_strategy="semantic-legal",
            document_id=document_id,
        )

        logger.info(
            f"[UPLOAD] ✅ Ingestion complete for '{document_id}' → "
            f"{result['total_chunks']} semantic chunks stored."
        )

        # ===============================
        # ✅ Structured response
        # ===============================
        return {
            "filename": file.filename,
            "citation_id": document_id,
            "pages": len(images),
            "total_chunks": result["total_chunks"],
            "status": "success",
            "message": "✅ PDF uploaded, OCR completed, and document ingested successfully!"
        }

    except HTTPException:
        raise  # Forward HTTP errors directly

    except Exception as e:
        logger.exception(f"[ERROR] Unexpected failure during upload: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")
