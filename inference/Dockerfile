FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    lightgbm==4.6.0 \
    fastapi==0.115.12 \
    "uvicorn[standard]" \
    gunicorn \
    pydantic \
    mlflow \
    pandas \
    numpy \
    prometheus-client

COPY src/ ./src/
COPY api/ ./api/

EXPOSE 8000

CMD ["gunicorn", "api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-"]