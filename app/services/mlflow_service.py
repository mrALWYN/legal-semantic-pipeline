# app/services/mlflow_service.py
import logging
import time
from typing import Dict, List, Optional, Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from app.core.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# 🧠 MLflow Client Initialization
# ============================================================

# Initialize MLflow tracking URI
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

# Global client instance
_mlflow_client = None

def get_mlflow_client() -> MlflowClient:
    """Get or create MLflow client with error handling."""
    global _mlflow_client
    if _mlflow_client is None:
        try:
            _mlflow_client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)
            logger.info(f"✅ MLflow client initialized for {settings.MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MLflow client: {e}")
            # Create a mock client for fallback
            class MockClient:
                def get_experiment_by_name(self, name): return None
                def create_experiment(self, name): return "mock_exp_id"
                def search_runs(self, *args, **kwargs): return []
                def get_latest_versions(self, *args, **kwargs): return []
                def transition_model_version_stage(self, *args, **kwargs): pass
                def search_experiments(self, view_type=None): return []
                def list_registered_models(self): return []
            _mlflow_client = MockClient()
    return _mlflow_client


# ============================================================
# 🏃‍♂️ Experiment and Run Management
# ============================================================

def start_run(experiment_name: str, run_name: Optional[str] = None) -> Optional[mlflow.ActiveRun]:
    """
    Start an MLflow run with proper error handling.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Optional run name
        
    Returns:
        ActiveRun object or None if failed
    """
    try:
        client = get_mlflow_client()
        
        # Ensure experiment exists
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"📁 Creating new experiment: {experiment_name}")
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start the run
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={"pipeline": "legal-semantic", "version": settings.API_VERSION}
        )
        
        logger.info(f"🚀 Started MLflow run: {run.info.run_id} in experiment: {experiment_name}")
        return run
        
    except Exception as e:
        logger.error(f"❌ Failed to start MLflow run: {e}")
        return None


def end_run():
    """End the current active run."""
    try:
        mlflow.end_run()
        logger.info("✅ MLflow run ended successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to end MLflow run: {e}")


# ============================================================
# 📊 Logging Functions
# ============================================================

def log_model_metadata(model_name: str, embedding_dim: int, chunk_size: int, overlap: int):
    """
    Log model and chunking parameters to MLflow.
    """
    try:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", overlap)
        mlflow.log_param("pipeline_version", settings.API_VERSION)
        
        logger.debug(f"📝 Logged model metadata: {model_name}, dim={embedding_dim}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to log model metadata: {e}")


def log_metrics(metrics: Dict[str, float]):
    """
    Log metrics to MLflow with error handling.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    for metric_name, metric_value in metrics.items():
        try:
            mlflow.log_metric(metric_name, float(metric_value))
            logger.debug(f"📊 Logged metric: {metric_name} = {metric_value}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log metric {metric_name}: {e}")


def log_params(params: Dict[str, Any]):
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter names and values
    """
    for param_name, param_value in params.items():
        try:
            mlflow.log_param(param_name, str(param_value))
            logger.debug(f"🔧 Logged parameter: {param_name} = {param_value}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log parameter {param_name}: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    """
    Log an artifact to MLflow.
    
    Args:
        local_path: Path to the local file
        artifact_path: Relative path in artifact store
    """
    try:
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"📎 Logged artifact: {local_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to log artifact {local_path}: {e}")


# ============================================================
# 🏷️ Model Registry Functions
# ============================================================

def register_model(run_id: str, registered_model_name: str) -> Optional[Dict]:
    """
    Register a model from a run to the model registry.
    
    Args:
        run_id: ID of the MLflow run
        registered_model_name: Name for the registered model
        
    Returns:
        Model version info or None if failed
    """
    try:
        client = get_mlflow_client()
        
        # Create registered model if it doesn't exist
        try:
            client.create_registered_model(registered_model_name)
            logger.info(f"📋 Created new registered model: {registered_model_name}")
        except Exception:
            logger.info(f"📋 Using existing registered model: {registered_model_name}")
        
        # Register model version
        model_version = client.create_model_version(
            name=registered_model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        
        logger.info(f"✅ Registered model version: {model_version.version} for {registered_model_name}")
        
        return {
            "name": registered_model_name,
            "version": model_version.version,
            "run_id": run_id,
            "status": "SUCCESS"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        return None


def promote_model_version(registered_model_name: str, version: int, stage: str = "Production") -> bool:
    """
    Promote a model version to a specific stage.
    
    Args:
        registered_model_name: Name of the registered model
        version: Version number to promote
        stage: Target stage (Production, Staging, Archived)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_mlflow_client()
        
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        
        logger.info(f"🎯 Promoted {registered_model_name} v{version} to {stage} stage")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to promote model version: {e}")
        return False


def get_current_production_model(registered_model_name: str) -> Optional[Dict]:
    """
    Get the current production model information.
    
    Args:
        registered_model_name: Name of the registered model
        
    Returns:
        Model info dictionary or None if not found
    """
    try:
        client = get_mlflow_client()
        
        production_versions = client.get_latest_versions(
            name=registered_model_name,
            stages=["Production"]
        )
        
        if not production_versions:
            logger.warning(f"⚠️ No production model found for {registered_model_name}")
            return None
        
        latest_prod = production_versions[0]
        
        return {
            "name": registered_model_name,
            "version": latest_prod.version,
            "stage": latest_prod.current_stage,
            "run_id": latest_prod.run_id,
            "creation_timestamp": latest_prod.creation_timestamp,
            "status": "READY"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get production model: {e}")
        return None


# ============================================================
# 🔍 Experiment Management
# ============================================================

def list_experiments() -> List[Dict]:
    """
    List all MLflow experiments.
    
    Returns:
        List of experiment information dictionaries
    """
    try:
        client = get_mlflow_client()
        # Use search_experiments instead of list_experiments for better compatibility
        experiments = client.search_experiments(view_type=ViewType.ALL)
        
        experiment_list = []
        for exp in experiments:
            experiment_list.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": dict(exp.tags) if exp.tags else {}
            })
        
        logger.info(f"📂 Found {len(experiment_list)} experiments")
        return experiment_list
        
    except AttributeError as e:
        # Fallback for older MLflow versions
        logger.warning(f"⚠️ list_experiments not available, trying alternative: {e}")
        try:
            client = get_mlflow_client()
            # Try to get default experiment as fallback
            default_exp = client.get_experiment("0")
            return [{
                "experiment_id": default_exp.experiment_id,
                "name": default_exp.name,
                "artifact_location": default_exp.artifact_location,
                "lifecycle_stage": default_exp.lifecycle_stage,
                "tags": dict(default_exp.tags) if default_exp.tags else {}
            }]
        except Exception as fallback_e:
            logger.error(f"❌ Failed to list experiments (fallback also failed): {fallback_e}")
            return []
    except Exception as e:
        logger.error(f"❌ Failed to list experiments: {e}")
        return []


def compare_experiments(experiment_names: List[str]) -> Dict[str, Any]:
    """
    Compare experiments by fetching their latest runs and metrics.
    
    Args:
        experiment_names: List of experiment names to compare
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        client = get_mlflow_client()
        comparison_result = {}
        
        for exp_name in experiment_names:
            experiment = client.get_experiment_by_name(exp_name)
            if not experiment:
                comparison_result[exp_name] = {"error": "Experiment not found"}
                continue
            
            # Get latest runs for this experiment
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=3  # Compare latest 3 runs
            )
            
            exp_runs = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                exp_runs.append(run_info)
            
            comparison_result[exp_name] = {
                "experiment_id": experiment.experiment_id,
                "runs_count": len(exp_runs),
                "latest_runs": exp_runs
            }
        
        logger.info(f"📊 Compared {len(experiment_names)} experiments")
        return comparison_result
        
    except Exception as e:
        logger.error(f"❌ Failed to compare experiments: {e}")
        return {exp_name: {"error": str(e)} for exp_name in experiment_names}


def search_runs(experiment_name: str, filter_string: str = "", max_results: int = 10) -> List[Dict]:
    """
    Search runs within an experiment.
    
    Args:
        experiment_name: Name of the experiment
        filter_string: MLflow filter string
        max_results: Maximum number of results
        
    Returns:
        List of run information dictionaries
    """
    try:
        client = get_mlflow_client()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if not experiment:
            logger.warning(f"⚠️ Experiment {experiment_name} not found")
            return []
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        run_list = []
        for run in runs:
            run_list.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        
        logger.info(f"🔍 Found {len(run_list)} runs in {experiment_name}")
        return run_list
        
    except Exception as e:
        logger.error(f"❌ Failed to search runs: {e}")
        return []


# ============================================================
# 🧪 Testing and Validation
# ============================================================

def test_mlflow_connection() -> bool:
    """Test if MLflow server is accessible."""
    try:
        client = get_mlflow_client()
        # Use search_experiments for compatibility
        experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        logger.info("✅ MLflow connection test successful")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow connection test failed: {e}")
        return False


def get_mlflow_status() -> Dict[str, Any]:
    """Get comprehensive MLflow status information."""
    try:
        client = get_mlflow_client()
        experiments = client.search_experiments(view_type=ViewType.ALL)
        
        return {
            "status": "CONNECTED",
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "experiments_count": len(experiments),
            "registered_model": settings.REGISTERED_MODEL_NAME,
            "active_run": mlflow.active_run() is not None
        }
    except Exception as e:
        return {
            "status": "DISCONNECTED",
            "error": str(e),
            "tracking_uri": settings.MLFLOW_TRACKING_URI
        }
