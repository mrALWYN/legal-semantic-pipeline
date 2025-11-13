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
memory_usage_bytes = Gauge(
    "memory_usage_bytes",
    "Memory used during embedding (bytes)",
    registry=registry,
)

cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "Current CPU usage percentage",
    registry=registry,
)

# ============================================================
# 🔍 Search / Query Metrics
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
# 🌐 API Request Metrics
# ============================================================
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

# ============================================================
# 🧠 Model Performance Metrics
# ============================================================
embedding_similarity = Histogram(
    "embedding_similarity_score",
    "Embedding similarity scores distribution",
    registry=registry,
)

chunk_size_distribution = Histogram(
    "chunk_size_chars",
    "Distribution of chunk sizes in characters",
    buckets=[100, 500, 1000, 2000, 5000, 10000],
    registry=registry,
)

# ============================================================
# ⚠️ Error & Anomaly Tracking
# ============================================================
anomalies_detected = Counter(
    "anomalies_detected_total",
    "Total anomalies detected in processing",
    ["type"],
    registry=registry,
)

processing_failures = Counter(
    "processing_failures_total",
    "Total processing failures",
    ["stage"],
    registry=registry,
)

# ============================================================
# 📊 Retrieval Metrics
# ============================================================
retrieval_accuracy = Histogram(
    "retrieval_accuracy_score",
    "Retrieval accuracy scores",
    registry=registry,
)

feedback_ratings = Histogram(
    "feedback_rating",
    "User feedback ratings distribution",
    buckets=[0, 1, 2, 3, 4, 5],
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
