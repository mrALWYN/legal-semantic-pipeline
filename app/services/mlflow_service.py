# app/services/mlflow_service.py
import logging
import time
from typing import Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from app.core.config import settings

logger = logging.getLogger(__name__)

# initialize tracking URI
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
_mlflow_client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)


def start_run(experiment_name: str, run_name: Optional[str] = None):
    exp = _mlflow_client.get_experiment_by_name(experiment_name)
    if not exp:
        _mlflow_client.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_model_metadata(model_name: str, embedding_dim: int, chunk_size: int, overlap: int):
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("chunk_size", chunk_size)
    mlflow.log_param("overlap", overlap)


def log_metrics(metrics: Dict[str, float]):
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v))
        except Exception:
            logger.exception("Failed to log metric to mlflow: %s", k)


def register_model(run_id: str, registered_model_name: str):
    """Register the model from a run's logged model artifact. This assumes you logged a model artifact path 'model'."""
    try:
        result = _mlflow_client.create_registered_model(registered_model_name)
    except Exception:
        # model may already exist
        pass

    # If the run logged a model artifact path "model", register it.
    try:
        mv = _mlflow_client.create_model_version(
            name=registered_model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        return mv
    except Exception as e:
        logger.exception("Model registration failed: %s", e)
        raise


def promote_model_version(registered_model_name: str, version: int, stage: str = "Production"):
    """Promote model version to given stage (Production / Staging)."""
    try:
        _mlflow_client.transition_model_version_stage(
            name=registered_model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        return True
    except Exception:
        logger.exception("Failed to promote model version")
        return False


def get_current_production_model(registered_model_name: str):
    try:
        versions = _mlflow_client.get_latest_versions(name=registered_model_name, stages=["Production"])
        if not versions:
            return None
        v = versions[0]
        return {
            "name": registered_model_name,
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "creation_timestamp": v.creation_timestamp,
        }
    except Exception:
        logger.exception("Failed to get production model")
        return None


def list_experiments() -> List[Dict]:
    exps = _mlflow_client.list_experiments()
    return [{"id": e.experiment_id, "name": e.name, "artifact_location": e.artifact_location} for e in exps]


def compare_experiments(experiment_names: List[str]) -> Dict:
    """
    Simple comparison: fetch latest run per experiment and return main metrics.
    """
    out = {}
    for name in experiment_names:
        exp = _mlflow_client.get_experiment_by_name(name)
        if not exp:
            out[name] = {"error": "Experiment not found"}
            continue
        runs = _mlflow_client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=5)
        out[name] = []
        for r in runs:
            out[name].append({
                "run_id": r.info.run_id,
                "start_time": r.info.start_time,
                "metrics": r.data.metrics,
                "params": r.data.params,
            })
    return out
