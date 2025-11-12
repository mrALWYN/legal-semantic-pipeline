# app/services/metrics.py
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
)
from app.core.config import settings

# ============================================================
# Prometheus Collector Registry
# ============================================================
registry = CollectorRegistry()

# ============================================================
# 🧩 Ingestion-related Metrics
# ============================================================
ingest_calls = Counter(
    "ingest_requests_total",
    "Total ingest requests received by the pipeline",
    registry=registry,
)

ingest_chunks = Counter(
    "ingest_chunks_total",
    "Total chunks created and stored",
    registry=registry,
)

feedback_counter = Counter(
    "feedback_submitted_total",
    "Total user feedback submissions received",
    registry=registry,
)

# ============================================================
# ⏱️ Timing Metrics
# ============================================================
embedding_time_hist = Histogram(
    "embedding_time_seconds",
    "Time taken to embed text (seconds)",
    registry=registry,
)

storage_time_hist = Histogram(
    "storage_time_seconds",
    "Time taken to store vectors in Qdrant (seconds)",
    registry=registry,
)

# ============================================================
# 🧠 Resource Usage
# ============================================================
current_memory = Gauge(
    "memory_usage_bytes",
    "Memory used during embedding (bytes)",
    registry=registry,
)

# ============================================================
# 🔍 Search / Query Metrics (added for point #11)
# ============================================================
search_requests = Counter(
    "search_requests_total",
    "Total number of semantic search requests received",
    registry=registry,
)

search_duration_hist = Histogram(
    "search_duration_seconds",
    "Time taken to complete a semantic search (seconds)",
    registry=registry,
)

search_results = Counter(
    "search_results_total",
    "Total number of search results returned",
    registry=registry,
)

# ============================================================
# 📦 Export Utility
# ============================================================
def prometheus_metrics():
    """
    Return Prometheus metrics in text format for /api/v1/metrics endpoint.
    """
    return generate_latest(registry)
