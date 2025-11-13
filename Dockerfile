# ============================================================
# ⚡ GPU Optimized Legal Semantic Pipeline — Dockerfile
# ============================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# --- System dependencies (PDF, image libs) ---
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- PIP optimization ---
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

# --- Copy requirements first (separate layer for caching) ---
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Pre-download ML models (sentence-transformers) ---
ENV HF_HOME=/app/models
RUN mkdir -p /app/models && \
    python - <<'EOF'
from sentence_transformers import SentenceTransformer
print("📥 Downloading all-MiniLM-L6-v2...")
SentenceTransformer('all-MiniLM-L6-v2')
print("📥 Downloading all-mpnet-base-v2...")
SentenceTransformer('all-mpnet-base-v2')
print("✅ Transformer models downloaded")
EOF

# --- Copy application source ---
COPY ./app /app

# --- Environment ---
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models

# --- Permissions ---
RUN chmod -R 755 /app/models || true

# --- Run Server ---
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
