"""
Model initialization and caching configuration.
Ensures all models are pre-loaded and GPU-optimized.
"""
import os
import logging
import torch

logger = logging.getLogger(__name__)

# ============================================================
# 🎯 Model Configuration
# ============================================================

HF_CONFIG = {
    "home": os.getenv("HF_HOME", "/app/models"),
}

CHUNKING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Fast, lightweight (384 dims)",
        "dimensions": 384,
    },
    "all-mpnet-base-v2": {
        "name": "all-mpnet-base-v2",
        "description": "High quality (768 dims)",
        "dimensions": 768,
    },
}


def log_model_config():
    """Log current model configuration."""
    logger.info("=" * 60)
    logger.info("🤖 MODEL CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"✅ HuggingFace Home: {HF_CONFIG['home']}")
    logger.info(f"✅ Available Chunking Models: {list(CHUNKING_MODELS.keys())}")
    
    # Log GPU info
    cuda_available = torch.cuda.is_available()
    logger.info(f"✅ CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"✅ GPU Count: {torch.cuda.device_count()}")
        logger.info(f"✅ Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
    else:
        logger.info("⚠️ No GPU detected. Using CPU for embeddings.")
    
    logger.info("=" * 60)
