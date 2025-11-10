FROM python:3.11-slim
WORKDIR /app

# 1. Upgrade pip early
RUN pip install --upgrade pip setuptools wheel

# 2. Preinstall heavy libs (cached separately)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers==2.3.1

# 3. Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY ./app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
