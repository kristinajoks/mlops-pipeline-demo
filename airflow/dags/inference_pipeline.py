"""
DAG 2: Inference Pipeline
Runs weekly on Monday morning after weekend sensing data arrives.
Simulates real-time weekly inference as if the system is live.

Tasks:
  1. validate_new_sensing    - GE check on incoming sensing data
  2. load_inference_data     - Load the week's sensing data
  3. run_inference           - POST to inference API for each student
  4. store_predictions       - Write predictions to outputs/predictions/
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
INFERENCE_API_URL   = os.getenv("INFERENCE_API_URL", "http://localhost:8000")   
EXPERIMENT_NAME     = "mental_health_prediction"

default_args = {
    "owner":            "mlops",
    "retries":          2,
    "retry_delay":      timedelta(minutes=2),
    "execution_timeout": timedelta(hours=1),
}

dag = DAG(
    dag_id="inference_pipeline",
    description="Weekly inference for student mental health scores",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 8 * * 1",   # Every Monday at 08:00
    catchup=False,
    tags=["inference", "weekly"],
)


# Validate incoming sensing data 
def validate_new_sensing_fn(**context):
    import great_expectations as gx
    import pandas as pd

    # In Phase 5 simulation: use inference_v1.csv sliced by week
    # In production: load the week's new sensing data from the data lake
    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
    inference_data = pd.read_csv(SPLITS_DIR / "inference_v1.csv")

    # For simulation: the logical_date to pick the current week
    logical_date = context["logical_date"]
    year_week = logical_date.strftime("%G-W%V")
    week_data = inference_data[inference_data["year_week"] == year_week]

    print(f"Validating data for week: {year_week}")
    print(f"  Students with data: {week_data['uid'].nunique()}")

    if week_data.empty:
        print(f"No data for {year_week} — skipping inference.")
        return

    # Basic validation
    assert week_data["uid"].notna().all(), "NaN UIDs in sensing data"
    assert week_data["year_week"].notna().all(), "NaN year_week"
    assert (week_data["sleep_duration_mean"] >= 0).all(), \
        "Negative sleep duration"

    print(f"Validation passed for {year_week}.")
    context["ti"].xcom_push(key="year_week", value=year_week)
    context["ti"].xcom_push(key="n_students", value=int(week_data["uid"].nunique()))


# Load inference data 
def load_inference_data_fn(**context):
    import pandas as pd
    import json

    year_week = context["ti"].xcom_pull(
        task_ids="validate_new_sensing", key="year_week"
    )
    if not year_week:
        print("No year_week from validation task — skipping.")
        return

    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
    inference_data = pd.read_csv(SPLITS_DIR / "inference_v1.csv")
    week_data = inference_data[inference_data["year_week"] == year_week]

    # Serialise to dict list for XCom
    records = week_data.to_dict(orient="records")
    context["ti"].xcom_push(key="week_records", value=json.dumps(records))
    print(f"Loaded {len(records)} records for {year_week}.")


# Run inference 
# Calls inference API (Kubernetes minikube service URL or localhost:8000 in docker-compose)
# for each student in the week, stores predictions to XCom
def run_inference_fn(**context):
    import requests
    import json
    import pandas as pd

    from src.features.feature_columns import SENSING_FEATURES, TARGET

    ti = context["ti"]
    year_week = ti.xcom_pull(task_ids="validate_new_sensing", key="year_week")
    records_json = ti.xcom_pull(task_ids="load_inference_data", key="week_records")

    if not records_json:
        print("No records — skipping inference.")
        return

    records = json.loads(records_json)
    predictions = []
    errors = []

    for record in records:
        uid = record["uid"]

        # SENSING_FEATURES
        features = {
            col: record.get(col)
            for col in SENSING_FEATURES
        }

        payload = {
            "uid": uid,
            "year_week": year_week,
            "features": features,
        }

        try:
            response = requests.post(
                f"{INFERENCE_API_URL}/predict",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            predictions.append({
                "uid":             uid,
                "year_week":       year_week,
                "predicted_score": result["predicted_score"],
                "model_version":   result["model_version"],
            })
        except Exception as e:
            errors.append({"uid": uid, "error": str(e)})

    print(f"Inference complete for {year_week}:")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Errors:      {len(errors)}")

    ti.xcom_push(key="predictions", value=json.dumps(predictions))
    ti.xcom_push(key="n_predictions", value=len(predictions))
    ti.xcom_push(key="n_errors", value=len(errors))


# Store predictions to outputs/predictions/<year_week>.csv
def store_predictions_fn(**context):
    import json
    import pandas as pd

    ti = context["ti"]
    year_week = ti.xcom_pull(task_ids="validate_new_sensing", key="year_week")
    predictions_json = ti.xcom_pull(task_ids="run_inference", key="predictions")

    if not predictions_json:
        print("No predictions to store.")
        return

    predictions = json.loads(predictions_json)
    df = pd.DataFrame(predictions)

    PRED_DIR = PROJECT_ROOT / "outputs" / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    out_path = PRED_DIR / f"{year_week}.csv"
    df.to_csv(out_path, index=False)

    print(f"Stored {len(df)} predictions to {out_path}")

# Tasks

t1_validate = PythonOperator(
    task_id="validate_new_sensing",
    python_callable=validate_new_sensing_fn,
    dag=dag,
)

t2_load = PythonOperator(
    task_id="load_inference_data",
    python_callable=load_inference_data_fn,
    dag=dag,
)

t3_infer = PythonOperator(
    task_id="run_inference",
    python_callable=run_inference_fn,
    dag=dag,
)

t4_store = PythonOperator(
    task_id="store_predictions",
    python_callable=store_predictions_fn,
    dag=dag,
)

# Dependencies 
t1_validate >> t2_load >> t3_infer >> t4_store 