# ============================================================
# ⚖️ Legal Semantic Pipeline — Backend API Dockerfile
# ============================================================
FROM python:3.11-slim

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

# --- Upgrade pip ---
RUN pip install --upgrade pip setuptools wheel

# --- Copy ONLY requirements.txt first (separate layer for caching) ---
COPY requirements.txt /app/requirements.txt

# --- Install Python dependencies (cached if requirements.txt unchanged) ---
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- Create models directory (models downloaded at runtime) ---
ENV HF_HOME=/app/models
RUN mkdir -p /app/models && chmod -R 755 /app/models

# --- Copy Application Source (last layer - rebuilds app code without pip) ---
COPY ./app /app/app

# --- Environment ---
ENV PYTHONPATH=/app
ENV HF_HOME=/app/models
ENV PYTHONUNBUFFERED=1

# --- Run Server ---
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
