"""
⚖️ Legal Semantic Pipeline API v1 endpoints
"""
from app.api.v1 import endpoints
from app.api.v1 import upload_routes
from app.api.v1 import models
from app.api.v1 import feedback as feedback_routes
from app.api.v1 import metrics_endpoint
from app.api.v1 import health_routes

__all__ = [
    "endpoints",
    "upload_routes",
    "models",
    "feedback_routes",
    "metrics_endpoint",
    "health_routes",
]
