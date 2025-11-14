import logging
import numpy as np
from typing import List, Dict
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from app.services.metrics import (
    document_length_distribution,
    embedding_drift_score,
    language_complexity_score,
    drift_detected,
    embedding_compute_cost,
    storage_cost,
    vector_index_size_bytes,
    api_usage_cost,
    qdrant_points_count
)
from app.services.mlflow_service import log_metrics, start_run, end_run

logger = logging.getLogger(__name__)

class DriftDetector:
    """Monitor data drift and trigger retraining when needed"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.drift_thresholds = {
            'document_length': 0.05,  # 5% change
            'embedding_drift': 0.3,   # 30% drift score
            'language_complexity': 0.1 # 10% change
        }
        
    def monitor_document_characteristics(self, document_text: str, document_id: str):
        """Monitor input document variations"""
        try:
            # Document length tracking
            doc_length = len(document_text)
            document_length_distribution.observe(doc_length)
            
            # Language complexity (simple heuristic)
            sentences = document_text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            complexity_score = min(1.0, avg_sentence_length / 50.0)  # Normalize
            language_complexity_score.observe(complexity_score)
            
            logger.debug(f"[DRIFT] Document {document_id}: length={doc_length}, complexity={complexity_score:.3f}")
            
        except Exception as e:
            logger.warning(f"[DRIFT] Failed to monitor document characteristics: {e}")
    
    def detect_embedding_drift(self, current_embeddings: List[List[float]], 
                             baseline_embeddings: List[List[float]] = None) -> float:
        """Detect embedding distribution drift using statistical tests"""
        try:
            if not baseline_embeddings or len(current_embeddings) < 10:
                logger.debug("[DRIFT] Not enough data for embedding drift detection")
                return 0.0
            
            # Convert to numpy arrays
            current_emb = np.array(current_embeddings)
            baseline_emb = np.array(baseline_embeddings)
            
            # Calculate mean embeddings
            current_mean = np.mean(current_emb, axis=0)
            baseline_mean = np.mean(baseline_emb, axis=0)
            
            # Cosine similarity between mean embeddings (1 - similarity = drift)
            similarity = cosine_similarity([current_mean], [baseline_mean])[0][0]
            drift_score = 1.0 - similarity
            
            embedding_drift_score.observe(drift_score)
            
            # Log drift detection
            if drift_score > self.drift_thresholds['embedding_drift']:
                drift_detected.labels(type="embedding").inc()
                logger.warning(f"[DRIFT] High embedding drift detected: {drift_score:.3f}")
            
            return drift_score
            
        except Exception as e:
            logger.warning(f"[DRIFT] Failed to detect embedding drift: {e}")
            return 0.0
    
    def calculate_cost_metrics(self, embedding_time: float, document_length: int, 
                             points_count: int, model_name: str):
        """Calculate and track cost metrics"""
        try:
            # Embedding compute cost (example pricing)
            compute_cost = embedding_time * 0.002  # $0.002 per second
            embedding_compute_cost.inc(compute_cost)
            
            # Storage cost (example pricing)
            storage_bytes = points_count * 1536  # Assuming 768-dim vectors * 2 bytes
            storage_cost_usd = (storage_bytes / (1024**3)) * 0.10  # $0.10 per GB
            storage_cost.inc(storage_cost_usd)
            
            # Vector index size
            vector_index_size_bytes.set(storage_bytes)
            
            # API usage cost (example)
            api_cost = 0.0001 * points_count  # $0.0001 per point processed
            api_usage_cost.inc(api_cost)
            
            # Qdrant points count
            qdrant_points_count.set(points_count)
            
            # Log to MLflow
            try:
                run = start_run(experiment_name="cost_tracking")
                if run:
                    log_metrics({
                        "embedding_compute_cost": compute_cost,
                        "storage_cost": storage_cost_usd,
                        "vector_index_size_gb": storage_bytes / (1024**3),
                        "api_usage_cost": api_cost,
                        "qdrant_points_count": points_count
                    })
                    end_run()
            except Exception as e:
                logger.warning(f"[COST] Failed to log to MLflow: {e}")
            
            logger.debug(f"[COST] Compute: ${compute_cost:.4f}, Storage: ${storage_cost_usd:.4f}")
            
        except Exception as e:
            logger.warning(f"[COST] Failed to calculate cost metrics: {e}")
    
    def check_for_retraining(self, drift_scores: Dict[str, float]) -> bool:
        """Check if retraining should be triggered based on drift scores"""
        should_retrain = False
        
        for metric_type, score in drift_scores.items():
            threshold = self.drift_thresholds.get(metric_type, 0.5)
            if score > threshold:
                should_retrain = True
                logger.info(f"[RETRAIN] Drift detected in {metric_type}: {score:.3f} > {threshold}")
        
        if should_retrain:
            from app.services.metrics import retraining_triggered
            retraining_triggered.inc()
            logger.warning("[RETRAIN] Retraining triggered due to data drift!")
        
        return should_retrain

# Global drift detector instance
drift_detector = DriftDetector()
