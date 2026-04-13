"""
FastAPI endpoints:
  POST /predict   - predict next week score for one student week
  GET  /health    - model version and status
  GET  /metrics   - Prometheus metrics 
"""

import time
import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response
from contextlib import asynccontextmanager
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from api.model_loader import get_model
from api.preprocessor import preprocess_for_inference
from src.features.feature_columns import SENSING_FEATURES, ALL_FEATURES

app = FastAPI(
    title="Mental Health Prediction API",
    description="Predicts next-week composite mental health score from sensing data",
    version="1.0.0",
)

# Prometheus metrics 
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of predictions served"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time to produce one prediction",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)
PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of predicted scores",
    buckets=list(range(0, 110, 10))
)
MODEL_VERSION = Gauge(
    "model_version_info",
    "Currently serving model version",
    ["model_name", "model_alias", "version"]
)

# Startup 
@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = get_model()
    MODEL_VERSION.labels(
        model_name=loaded.model_name,
        model_alias=loaded.model_alias,
        version=loaded.model_version,
    ).set(1)
    print(f"API ready - serving {loaded.model_name} v{loaded.model_version}")
    yield
    print("API shutting down...")

app = FastAPI(
    title="Mental Health Prediction API",
    description="Predicts next-week composite mental health score from sensing data",
    version="1.0.0",
    lifespan=lifespan,
)

# Endpoints 
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    loaded = get_model()

    # Determine feature list based on model
    if "v2" in loaded.model_name:
        feature_list = ALL_FEATURES
    else:
        feature_list = SENSING_FEATURES

    features_dict = request.features.model_dump()
    X = preprocess_for_inference(features_dict, feature_list)

    # Prediction
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
        mlflow_uri=str(loaded.model_name),
    )

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )