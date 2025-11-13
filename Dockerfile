# ============================================================
# ⚖️ Legal Semantic Pipeline — Backend API Dockerfile
# ============================================================
FROM python:3.11-slim

WORKDIR /

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

# --- Upgrade pip ---
RUN pip install --upgrade pip setuptools wheel

# --- Copy ONLY requirements.txt first (separate layer for caching) ---
COPY requirements.txt /requirements.txt

# --- Install Python dependencies (cached if requirements.txt unchanged) ---
RUN pip install --no-cache-dir -r /requirements.txt

# --- Pre-download ML models (sentence-transformers) ---
ENV HF_HOME=/app/models
RUN mkdir -p /app/models && \
    python - <<'EOF'
from sentence_transformers import SentenceTransformer
print("Downloading all-MiniLM-L6-v2...")
SentenceTransformer('all-MiniLM-L6-v2')
print("Downloading all-mpnet-base-v2...")
SentenceTransformer('all-mpnet-base-v2')
print("✅ Sentence-transformer models downloaded")
EOF

# --- Copy Application Source (last layer - rebuilds app code without pip) ---
COPY ./app /app

# --- Environment ---
ENV PYTHONPATH=/
ENV HF_HOME=/app/models

# --- Permissions ---
RUN chmod -R 755 /app/models || true

# --- Run Server ---
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
