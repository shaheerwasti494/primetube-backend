# Python 3.12 to avoid pydantic-core issues on 3.13
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (for httpx performance extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source
COPY . .

# (Optional: run as non-root for security)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# IMPORTANT: main.py defines `app`
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
