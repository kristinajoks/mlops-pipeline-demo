"""
DAG 1: Training Pipeline
Triggered manually at the start of each academic year or when a new
dataset version is available.

Tasks:
  1. validate_data         - Run Great Expectations on raw data
  2. run_feature_engineering - Regenerate split files from raw data
  3. train_v1              - Train LightGBM v1 on pre-COVID splits
  4. evaluate_v1           - Compare new model against current production
  5. register_v1           - Register new model if it improves on current
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow
import mlflow.lightgbm
import sys
from pathlib import Path

# Ensure src/ is importable from DAG context
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME     = "mental_health_prediction"
MODEL_NAME          = "mental_health_v1"

default_args = {
    "owner":            "mlops",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    dag_id="training_pipeline",
    description="Train and register v1 mental health prediction model",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,          
    catchup=False,
    tags=["training", "v1"],
)


# Validate raw data with Great Expectations 
def validate_data_fn(**context):
    import great_expectations as gx
    import pandas as pd

    print("Running Great Expectations validation...")

    DATA_DIR = PROJECT_ROOT / "data" / "raw" / "college_experience_dataset"

    general_ema = pd.read_csv(DATA_DIR / "EMA" / "general_ema.csv")
    sensing     = pd.read_csv(DATA_DIR / "Sensing" / "sensing.csv")

    context_ge = gx.get_context(mode="ephemeral")

    # General EMA
    ds = context_ge.data_sources.add_or_update_pandas("ema_source")
    asset = ds.add_dataframe_asset("general_ema")
    batch_def = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": general_ema})

    suite = gx.ExpectationSuite(name="general_ema_raw")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="uid")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="day")
    )
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=200_000, max_value=250_000
        )
    )
    results = batch.validate(suite)
    if not results.success:
        raise ValueError(f"general_ema validation failed: {results}")

    # Sensing
    ds2 = context_ge.data_sources.add_or_update_pandas("sensing_source")
    asset2 = ds2.add_dataframe_asset("sensing")
    batch_def2 = asset2.add_batch_definition_whole_dataframe("batch")
    batch2 = batch_def2.get_batch(batch_parameters={"dataframe": sensing})

    suite2 = gx.ExpectationSuite(name="sensing_raw")
    suite2.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="uid")
    )
    suite2.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="is_ios", value_set=[0, 1]
        )
    )
    suite2.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(
            min_value=200_000, max_value=230_000
        )
    )
    results2 = batch2.validate(suite2)
    if not results2.success:
        raise ValueError(f"sensing validation failed: {results2}")

    print("All Great Expectations suites passed.")


# Feature engineering 
def run_feature_engineering_fn(**context):
    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
    required_splits = [
        "pre_covid_train.csv",
        "pre_covid_val.csv",
        "pre_covid_test.csv",
        "full_train.csv",
        "full_val.csv",
        "full_test.csv",
    ]

    for split in required_splits:
        path = SPLITS_DIR / split
        if not path.exists():
            raise FileNotFoundError(
                f"Split file missing: {path}. "
                "Run Phase 2 notebooks to generate splits."
            )
        print(f"  OK: {split} ({path.stat().st_size / 1024:.0f} KB)")

    print("Feature engineering verification complete.")


# Train v1 model 
def train_v1_fn(**context):
    from src.models.train_v1 import train_lightgbm

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Training v1 LightGBM...")
    model, metrics = train_lightgbm(
        n_estimators=1000,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.1,
        min_child_samples=20,
        run_name="v1_airflow_training",
    )

    print(f"Training complete.")
    print(f"  val MAE  : {metrics['val_mae']}")
    print(f"  test MAE : {metrics['test_mae']}")
    print(f"  val R2   : {metrics['val_r2']}")

    # XCom
    context["ti"].xcom_push(key="val_mae",  value=metrics["val_mae"])
    context["ti"].xcom_push(key="test_mae", value=metrics["test_mae"])
    context["ti"].xcom_push(key="val_r2",   value=metrics["val_r2"])

    # Find the created run ID 
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        ],
        filter_string="tags.`mlflow.runName` = 'v1_airflow_training'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id
    context["ti"].xcom_push(key="run_id", value=run_id)
    print(f"  run_id   : {run_id}")


# Evaluation
def evaluate_v1_fn(**context):
    ti = context["ti"]
    new_val_mae = ti.xcom_pull(task_ids="train_v1", key="val_mae")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get current production model val_mae
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "production")
        current_run = client.get_run(current.run_id)
        current_val_mae = current_run.data.metrics.get("val_mae", 999)
        print(f"Current production val_mae : {current_val_mae:.4f}")
    except Exception:
        # No production model yet - always register
        print("No production model found - will register new model.")
        current_val_mae = 999

    print(f"New model val_mae          : {new_val_mae:.4f}")

    if float(new_val_mae) < float(current_val_mae):
        print("New model improves on production - proceeding to registration.")
        return "register_v1"
    else:
        print("New model does not improve on production - skipping registration.")
        return "skip_registration"


# Register v1 
def register_v1_fn(**context):
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="train_v1", key="run_id")
    val_mae = ti.xcom_pull(task_ids="train_v1", key="val_mae")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="production",
        version=registered.version,
    )

    print(f"Registered: {MODEL_NAME} v{registered.version} @production")
    print(f"  val_mae : {val_mae:.4f}")
    print(f"  run_id  : {run_id}")

# Tasks

t1_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data_fn,
    dag=dag,
)

t2_features = PythonOperator(
    task_id="run_feature_engineering",
    python_callable=run_feature_engineering_fn,
    dag=dag,
)

t3_train = PythonOperator(
    task_id="train_v1",
    python_callable=train_v1_fn,
    dag=dag,
)

t4_evaluate = BranchPythonOperator(
    task_id="evaluate_v1",
    python_callable=evaluate_v1_fn,
    dag=dag,
)

t5a_register = PythonOperator(
    task_id="register_v1",
    python_callable=register_v1_fn,
    dag=dag,
)

t5b_skip = EmptyOperator(
    task_id="skip_registration",
    dag=dag,
)

t1_validate >> t2_features >> t3_train >> t4_evaluate
t4_evaluate >> [t5a_register, t5b_skip]
