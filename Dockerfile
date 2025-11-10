# ============================================================
# Base Image
# ============================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# ============================================================
# System Dependencies (for OCR + PDF + Fonts)
# ============================================================
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Upgrade pip and install Python dependencies
# ============================================================
RUN pip install --upgrade pip setuptools wheel

# Pre-install heavy libraries first to leverage Docker layer caching
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers==2.3.1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy Application Code
# ============================================================
COPY ./app /app/app

# ============================================================
# Expose API Port & Set Entrypoint
# ============================================================
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
