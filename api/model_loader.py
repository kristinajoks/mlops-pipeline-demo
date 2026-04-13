"""
Model loaded once at startup from MLflow Model Registry.
"""

import mlflow
import mlflow.lightgbm
import os
from dataclasses import dataclass


MODEL_NAME  = os.getenv("MODEL_NAME",  "mental_health_v1")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

@dataclass
class LoadedModel:
    model:         object
    model_name:    str
    model_alias:   str
    model_version: str

def load_model() -> LoadedModel:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    # Resolve alias
    version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    version = version_info.version

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.lightgbm.load_model(model_uri)

    print(f"Loaded: {MODEL_NAME} v{version} @{MODEL_ALIAS}")
    return LoadedModel(
        model=model,
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        model_version=str(version),
    )

# Module-level singleton
_loaded: LoadedModel | None = None

def get_model() -> LoadedModel:
    global _loaded
    if _loaded is None:
        _loaded = load_model()
    return _loaded