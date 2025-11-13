# app/api/v1/models.py
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from app.services.mlflow_service import get_current_production_model, list_experiments, promote_model_version, compare_experiments
from app.core.config import settings

router = APIRouter(prefix="/models", tags=["Model Management"])
logger = logging.getLogger(__name__)

class PromoteRequest(BaseModel):
    registered_model_name: str
    version: int
    stage: str = "Production"

@router.get("/current")
async def get_current_model():
    model = get_current_production_model(settings.REGISTERED_MODEL_NAME)
    if not model:
        raise HTTPException(status_code=404, detail="No production model found")
    return model

@router.get("/experiments")
async def get_experiments():
    return list_experiments()

@router.post("/promote")
async def promote_model(body: PromoteRequest):
    ok = promote_model_version(body.registered_model_name, body.version, stage=body.stage)
    if not ok:
        raise HTTPException(status_code=500, detail="Promotion failed")
    return {"status": "ok", "promoted": True}

@router.get("/compare")
async def compare_models(names: Optional[str] = None):
    """
    names: comma-separated experiment names to compare
    """
    names_list = names.split(",") if names else [ "ingest_experiments" ]
    return compare_experiments(names_list)
