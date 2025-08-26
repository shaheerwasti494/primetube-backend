# Python 3.12 = compatible with pydantic wheels (no rust needed)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Start FastAPI with uvicorn on the Cloud Run port
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
