import os
import io
import logging
import asyncio
import gc
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from app.services.ocr import OCRService
from app.services.chunking import SemanticLegalChunker, DEFAULT_MODEL
from app.services.vector_store import VectorStoreService
from app.core.config import settings

router = APIRouter(prefix="", tags=["Upload"])

logger = logging.getLogger(__name__)

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    embedding_model: str = Form(DEFAULT_MODEL),
):
    """
    Upload and process a legal PDF.
    Uses pre-cached EasyOCR models for fast OCR processing.
    Memory optimized for large documents.
    """
    try:
        if not file.filename.endswith(('.pdf', '.PDF')):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        logger.info(f"[UPLOAD] 📄 Received PDF: {file.filename}")

        # Read PDF bytes
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="PDF file is empty")

        logger.info(f"[UPLOAD] 📥 PDF received successfully ({len(pdf_bytes) / (1024*1024):.2f} MB)")

        # ✅ Use centralized OCRService with proper GPU settings
        use_gpu = os.getenv("EASYOCR_USE_GPU", "false").lower() == "true"
        logger.info(f"[UPLOAD] 🚀 Starting OCR with GPU={use_gpu}")
        
        extracted_text = OCRService.extract_text_from_pdf(pdf_bytes)
        
        if not extracted_text or len(extracted_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        logger.info(f"[UPLOAD] ✅ OCR completed. Extracted {len(extracted_text)} characters")

        # ✅ Cleanup
        del pdf_bytes
        gc.collect()

        # ✅ Perform chunking with selected model
        logger.info(f"[UPLOAD] 🔄 Starting chunking with model: {embedding_model}")
        
        chunker = SemanticLegalChunker(
            min_chunk_size=settings.CHUNK_SIZE,
            max_chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            embedding_model=embedding_model,
        )

        document_id = file.filename.replace(".pdf", "").replace(" ", "_")
        chunks = chunker.chunk_document(extracted_text, document_id)

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from the PDF")

        logger.info(f"[UPLOAD] ✅ Created {len(chunks)} chunks")

        # ✅ Cleanup
        del extracted_text
        gc.collect()

        # ✅ Store in Qdrant vector store
        logger.info("[UPLOAD] 📊 Storing chunks in Qdrant...")
        vector_store = VectorStoreService(embedding_model=embedding_model)
        await vector_store.ingest_chunks(chunks)

        logger.info(f"[UPLOAD] ✅ Successfully ingested {len(chunks)} chunks into Qdrant")

        return JSONResponse({
            "status": "success",
            "message": f"PDF processed successfully. {len(chunks)} chunks created and indexed.",
            "document_id": document_id,
            "chunks_count": len(chunks),
            "embedding_model": embedding_model,
            "text_length": len(extracted_text) if 'extracted_text' in locals() else 0,
        }, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Unexpected failure during upload: {e}", exc_info=True)
        # Cleanup on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")