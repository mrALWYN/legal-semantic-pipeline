import os
import pytesseract
from fastapi import APIRouter, UploadFile, File, HTTPException
from pdf2image import convert_from_bytes
from app.core.pipeline import ingest_legal_document

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Read file bytes
        pdf_bytes = await file.read()

        # Convert PDF pages to images
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for page in images:
            text += pytesseract.image_to_string(page)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted")

        # Call existing ingestion pipeline
        result = await ingest_legal_document(
            text=text,
            chunking_strategy="semantic-legal",
            document_id=file.filename,
        )

        return {
            "filename": file.filename,
            "total_chunks": result["total_chunks"],
            "status": "success",
            "message": "PDF uploaded, OCR done, and ingested successfully!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
