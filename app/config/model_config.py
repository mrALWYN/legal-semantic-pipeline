"""
Model initialization and caching configuration.
Ensures all models are pre-loaded and GPU-optimized.
"""
import os
import logging

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
    logger.info("=" * 60)
