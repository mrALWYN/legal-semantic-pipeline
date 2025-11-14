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
# 🔄 Data Drift Detection Metrics
# ============================================================
document_length_distribution = Histogram(
    "document_length_chars",
    "Distribution of input document lengths in characters",
    buckets=[1000, 5000, 10000, 50000, 100000, 500000],
    registry=registry,
)

embedding_drift_score = Histogram(
    "embedding_drift_score",
    "Embedding distribution drift scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry,
)

language_complexity_score = Histogram(
    "language_complexity_score",
    "Document language complexity scores",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    registry=registry,
)

# ============================================================
# 💰 Cost & Resource Tracking Metrics
# ============================================================
embedding_compute_cost = Counter(
    "embedding_compute_cost_total",
    "Total embedding compute cost (USD)",
    registry=registry,
)

storage_cost = Counter(
    "storage_cost_total",
    "Total storage cost (USD)",
    registry=registry,
)

vector_index_size_bytes = Gauge(
    "vector_index_size_bytes",
    "Current vector index size in bytes",
    registry=registry,
)

api_usage_cost = Counter(
    "api_usage_cost_total",
    "Total API usage cost (USD)",
    registry=registry,
)

qdrant_points_count = Gauge(
    "qdrant_points_total",
    "Total number of points in Qdrant",
    registry=registry,
)

# ============================================================
# 🏗️ Retraining Trigger Metrics
# ============================================================
drift_detected = Counter(
    "drift_detected_total",
    "Total drift detection events",
    ["type"],
    registry=registry,
)

retraining_triggered = Counter(
    "retraining_triggered_total",
    "Total retraining trigger events",
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
