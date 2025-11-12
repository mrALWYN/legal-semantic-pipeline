# app/api/v1/feedback.py
import os
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from app.core.config import settings
from app.services.metrics import feedback_counter

router = APIRouter(prefix="/api/v1", tags=["Feedback"])
logger = logging.getLogger(__name__)

class FeedbackEntry(BaseModel):
    query: str
    chunk_id: str
    run_id: str | None = None
    relevance: int  # 0/1 or 0-5 rating
    comment: str | None = None

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackEntry):
    os.makedirs(os.path.dirname(settings.FEEDBACK_STORE_PATH) or ".", exist_ok=True)
    try:
        entry = feedback.dict()
        entry["_ts"] = datetime.utcnow().isoformat()
        with open(settings.FEEDBACK_STORE_PATH, "a+", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        feedback_counter.inc()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to store feedback")
        raise HTTPException(status_code=500, detail=str(e))
