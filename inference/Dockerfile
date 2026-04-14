FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
libgomp1 \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/prometheus_multiproc

ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

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
COPY inference/gunicorn.conf.py ./gunicorn.conf.py

EXPOSE 8000

CMD ["sh", "-c", "rm -rf /tmp/prometheus_multiproc/* && gunicorn api.main:app --config gunicorn.conf.py"]