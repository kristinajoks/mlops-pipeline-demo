import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
    multiprocess,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from api.model_loader import get_model
from api.preprocessor import preprocess_for_inference
from src.features.feature_columns import SENSING_FEATURES, ALL_FEATURES

os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", "/tmp/prometheus_multiproc")

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions served",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time to produce one prediction",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of predicted scores",
    buckets=list(range(0, 110, 10)),
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently serving model version",
    ["model_name", "model_alias", "version"],
    multiprocess_mode="mostrecent",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = get_model()
    MODEL_VERSION.labels(
        model_name=loaded.model_name,
        model_alias=loaded.model_alias,
        version=str(loaded.model_version),
    ).set(1)
    yield

app = FastAPI(
    title="Mental Health Prediction API",
    description="Predicts next week composite mental health score from sensing data",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    loaded = get_model()

    feature_list = ALL_FEATURES if "v2" in loaded.model_name else SENSING_FEATURES
    X = preprocess_for_inference(request.features.model_dump(), feature_list)

    start = time.time()
    try:
        score = float(loaded.model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.time() - start
    score = float(np.clip(score, 0, 100))

    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)
    PREDICTION_SCORE.observe(score)

    return PredictionResponse(
        uid=request.uid,
        year_week=request.year_week,
        predicted_score=round(score, 2),
        model_name=loaded.model_name,
        model_version=loaded.model_version,
        model_alias=loaded.model_alias,
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = get_model()
    return HealthResponse(
        status="ok",
        model_name=loaded.model_name,
        model_alias=loaded.model_alias,
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
    )

@app.get("/metrics")
async def metrics():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )