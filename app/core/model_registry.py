# app/core/model_registry.py
"""
Model registry to manage embedding models by dimension for cross-collection search.
"""
import logging
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Cache embedding model instances
_MODEL_CACHE: dict[int, SentenceTransformer] = {}


def get_model_by_dimension(dimension: int) -> str:
    """
    Returns the correct model name based on the embedding dimension.
    Uses settings.MODEL_DIMENSIONS for mapping.
    
    Args:
        dimension: Vector dimension (384, 768, etc.)
        
    Returns:
        Model name string
        
    Raises:
        ValueError: If dimension is not supported
    """
    # Reverse mapping: dim → model_name
    # Example: {384: 'all-MiniLM-L6-v2', 768: 'all-mpnet-base-v2'}
    dim_to_model = {
        dim: model_name
        for model_name, dim in settings.MODEL_DIMENSIONS.items()
    }

    if dimension not in dim_to_model:
        raise ValueError(
            f"❌ Unsupported embedding dimension '{dimension}'. "
            f"Available: {list(dim_to_model.keys())}"
        )

    model_name = dim_to_model[dimension]
    return model_name


def get_model_instance_by_dimension(dimension: int) -> SentenceTransformer:
    """
    Returns the correct embedding model instance based on the embedding dimension.
    Uses caching to avoid reloading models.
    
    Args:
        dimension: Vector dimension (384, 768, etc.)
        
    Returns:
        SentenceTransformer model instance
    """
    global _MODEL_CACHE

    # If model already loaded → return from cache
    if dimension in _MODEL_CACHE:
        return _MODEL_CACHE[dimension]

    model_name = get_model_by_dimension(dimension)

    logger.info(f"[MODEL] Loading embedding model '{model_name}' for {dimension}d")

    # Load sentence-transformers model
    model = SentenceTransformer(model_name)

    # Cache it for future use
    _MODEL_CACHE[dimension] = model

    return model


def get_all_supported_dimensions() -> list[int]:
    """
    Get all supported embedding dimensions.
    
    Returns:
        List of supported dimensions
    """
    dim_to_model = {
        dim: model_name
        for model_name, dim in settings.MODEL_DIMENSIONS.items()
    }
    return list(dim_to_model.keys())


def get_dimension_by_model(model_name: str) -> int:
    """
    Get dimension by model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dimension integer
    """
    # Normalize model name
    normalized_name = model_name.replace("sentence-transformers/", "")
    
    # Find dimension in settings
    for model, dim in settings.MODEL_DIMENSIONS.items():
        if model.replace("sentence-transformers/", "") == normalized_name:
            return dim
    
    # Default fallback
    return 384
