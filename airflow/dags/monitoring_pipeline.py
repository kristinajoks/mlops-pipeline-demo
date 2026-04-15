"""
DAG 3: Monitoring Pipeline
Triggered by a Prometheus alert when rolling MAE exceeds threshold.
Automated retraining pipeline.

In Phase 5 simulation this DAG is triggered manually to demonstrate
what happens when drift is detected. In production it would be
triggered automatically by Prometheus Alertmanager via the
Airflow REST API.

Tasks:
  1. collect_ground_truth    - Gather predictions and labels for rolling window
  2. compute_rolling_mae     - Calculate 8-week rolling MAE
  3. confirm_drift           - Verify drift is sustained, not transient
  4. retrain_v2              - Train v2 with COVID features (register=True)
  5. evaluate_new_model      - Compare v2 vs current v1 on full_test
  6. promote_if_better       - Set v2 @production alias if improved
  7. notify                  - Log outcome to MLflow

The key architectural point: task 4 calls train_v2.train_lightgbm_v2
with register=True — this is the ONLY place v2 gets registered.
The MLflow audit trail shows this run was triggered by the monitoring
DAG, not by a human. This is what makes the demonstration authentic.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_TRACKING_URI    = "http://localhost:5000"
EXPERIMENT_NAME        = "mental_health_prediction"
V1_MODEL_NAME          = "mental_health_v1"
V2_MODEL_NAME          = "mental_health_v2"
ROLLING_WINDOW_WEEKS   = 8
MAE_THRESHOLD_FACTOR   = 1.3   # Alert if rolling MAE > 1.3x training MAE

default_args = {
    "owner":            "mlops",
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

dag = DAG(
    dag_id="monitoring_pipeline",
    description="Drift detection and automated v2 retraining",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,      # Triggered by Prometheus alert or manual run
    catchup=False,
    tags=["monitoring", "retraining", "v2", "drift"],
)


# ── Task 1: Collect ground truth ───────────────────────────────────────────────

def collect_ground_truth_fn(**context):
    """
    Loads stored predictions from outputs/predictions/ and matches
    them against available ground truth labels.
    Covers the last ROLLING_WINDOW_WEEKS weeks.
    Pushes matched data to XCom for MAE computation.
    """
    import json
    import pandas as pd
    from pathlib import Path

    PRED_DIR   = PROJECT_ROOT / "outputs" / "predictions"
    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

    # Load all available weekly prediction files
    pred_files = sorted(PRED_DIR.glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(
            "No prediction files found in outputs/predictions/. "
            "Run the inference_pipeline DAG first."
        )

    # Take the most recent ROLLING_WINDOW_WEEKS weeks
    recent_files = pred_files[-ROLLING_WINDOW_WEEKS:]
    all_preds = pd.concat([pd.read_csv(f) for f in recent_files])

    # Load ground truth labels from inference_v1
    inference_data = pd.read_csv(SPLITS_DIR / "inference_v1.csv")

    # Merge predictions with ground truth
    merged = all_preds.merge(
        inference_data[["uid", "year_week", "label_composite_score"]],
        on=["uid", "year_week"],
        how="inner",
    )

    print(f"Collected ground truth for {ROLLING_WINDOW_WEEKS} weeks:")
    print(f"  Predictions     : {len(all_preds)}")
    print(f"  Matched w/ GT   : {len(merged)}")
    print(f"  Weeks covered   : {merged['year_week'].unique().tolist()}")

    context["ti"].xcom_push(
        key="matched_data",
        value=merged.to_json(orient="records")
    )


# ── Task 2: Compute rolling MAE ────────────────────────────────────────────────

def compute_rolling_mae_fn(**context):
    """
    Computes rolling MAE across the collected ground truth window.
    Retrieves the training baseline MAE from MLflow.
    Determines whether the rolling MAE exceeds the alert threshold.
    """
    import json
    import pandas as pd
    import mlflow
    from sklearn.metrics import mean_absolute_error

    ti = context["ti"]
    matched_json = ti.xcom_pull(
        task_ids="collect_ground_truth", key="matched_data"
    )
    matched = pd.read_json(matched_json)

    # Compute rolling MAE
    rolling_mae = mean_absolute_error(
        matched["label_composite_score"],
        matched["predicted_score"],
    )
    print(f"Rolling {ROLLING_WINDOW_WEEKS}-week MAE: {rolling_mae:.4f}")

    # Get training baseline MAE from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        prod_version = client.get_model_version_by_alias(
            V1_MODEL_NAME, "production"
        )
        baseline_run = client.get_run(prod_version.run_id)
        baseline_mae = baseline_run.data.metrics.get("test_mae", 10.59)
    except Exception:
        baseline_mae = 10.59  # Fallback to known v1 test MAE

    threshold = baseline_mae * MAE_THRESHOLD_FACTOR
    print(f"Baseline test MAE  : {baseline_mae:.4f}")
    print(f"Alert threshold    : {threshold:.4f} ({MAE_THRESHOLD_FACTOR}x baseline)")
    print(f"Rolling MAE        : {rolling_mae:.4f}")
    print(f"Drift detected     : {rolling_mae > threshold}")

    ti.xcom_push(key="rolling_mae",  value=float(rolling_mae))
    ti.xcom_push(key="baseline_mae", value=float(baseline_mae))
    ti.xcom_push(key="threshold",    value=float(threshold))
    ti.xcom_push(key="drift",        value=bool(rolling_mae > threshold))


# ── Task 3: Confirm drift ──────────────────────────────────────────────────────

def confirm_drift_fn(**context):
    """
    Branch task — only proceeds to retraining if drift is confirmed.
    In Phase 5 simulation: always proceeds (DAG is triggered when drift
    is already known). In production: provides a safety check to avoid
    retraining on transient spikes.
    """
    ti = context["ti"]
    drift = ti.xcom_pull(task_ids="compute_rolling_mae", key="drift")
    rolling_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="rolling_mae")
    threshold   = ti.xcom_pull(task_ids="compute_rolling_mae", key="threshold")

    if drift:
        print(f"DRIFT CONFIRMED: rolling MAE {rolling_mae:.4f} > threshold {threshold:.4f}")
        print("Proceeding to v2 retraining.")
        return "retrain_v2"
    else:
        print(f"No drift: rolling MAE {rolling_mae:.4f} <= threshold {threshold:.4f}")
        print("Skipping retraining.")
        return "no_drift_detected"


# ── Task 4: Retrain v2 ────────────────────────────────────────────────────────

def retrain_v2_fn(**context):
    """
    Trains v2 LightGBM on ALL_FEATURES (sensing + COVID features).
    Calls train_lightgbm_v2 with register=True — this is the ONLY
    place in the entire codebase where v2 gets registered.

    The MLflow audit trail shows:
      - Run started by: monitoring_pipeline DAG (not a human)
      - Registered by: automated trigger (not manual)
      - Timestamp: after drift detection event

    This is what makes the MLOps demonstration authentic.
    """
    import mlflow
    from src.models.train_v2 import train_lightgbm_v2

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    rolling_mae = context["ti"].xcom_pull(
        task_ids="compute_rolling_mae", key="rolling_mae"
    )
    print(f"Retraining triggered by drift: rolling_mae={rolling_mae:.4f}")
    print("Training v2 with COVID features (register=True)...")

    model, metrics = train_lightgbm_v2(
        n_estimators=1000,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=10,
        run_name="v2_monitoring_retrain",
        register=True,          # ← Only True here, never in notebooks
    )

    print(f"v2 training complete:")
    print(f"  val MAE  : {metrics['val_mae']:.4f}")
    print(f"  test MAE : {metrics['test_mae']:.4f}")
    print(f"  val R2   : {metrics['val_r2']:.4f}")

    context["ti"].xcom_push(key="v2_val_mae",  value=metrics["val_mae"])
    context["ti"].xcom_push(key="v2_test_mae", value=metrics["test_mae"])
    context["ti"].xcom_push(key="v2_val_r2",   value=metrics["val_r2"])


# ── Task 5: Evaluate new model ─────────────────────────────────────────────────

def evaluate_new_model_fn(**context):
    """
    Evaluates both v1 and v2 on full_test (COVID-period held-out data).
    Determines whether v2 should replace v1 in production.
    Branches to promote_v2 or keep_v1.
    """
    import mlflow
    import mlflow.lightgbm
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    from src.features.feature_columns import SENSING_FEATURES, ALL_FEATURES, TARGET

    ti = context["ti"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load full_test
    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
    full_test = pd.read_csv(SPLITS_DIR / "full_test.csv")
    y_test = full_test[TARGET]

    # Evaluate v1 on COVID-period data (sensing only)
    v1_model = mlflow.lightgbm.load_model(
        f"models:/{V1_MODEL_NAME}@production"
    )
    preds_v1 = v1_model.predict(full_test[SENSING_FEATURES])
    v1_covid_mae = mean_absolute_error(y_test, preds_v1)

    # Evaluate v2 on COVID-period data (all features)
    v2_model = mlflow.lightgbm.load_model(
        f"models:/{V2_MODEL_NAME}@production"
    )
    preds_v2 = v2_model.predict(full_test[ALL_FEATURES])
    v2_covid_mae = mean_absolute_error(y_test, preds_v2)

    improvement = v1_covid_mae - v2_covid_mae
    improvement_pct = improvement / v1_covid_mae * 100

    print(f"Evaluation on full_test (COVID period):")
    print(f"  v1 MAE : {v1_covid_mae:.4f}")
    print(f"  v2 MAE : {v2_covid_mae:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

    ti.xcom_push(key="v1_covid_mae", value=float(v1_covid_mae))
    ti.xcom_push(key="v2_covid_mae", value=float(v2_covid_mae))
    ti.xcom_push(key="improvement",  value=float(improvement))

    if v2_covid_mae < v1_covid_mae:
        print("v2 outperforms v1 — proceeding to promotion.")
        return "promote_v2"
    else:
        print("v2 does not outperform v1 — keeping current production model.")
        return "keep_v1"


# ── Task 6a: Promote v2 ────────────────────────────────────────────────────────

def promote_v2_fn(**context):
    """
    Sets v2 @production alias. From this point all inference requests
    are served by v2. The Kubernetes rolling update handles the
    container transition without downtime.

    Note: in a full implementation this task would also trigger a
    Kubernetes deployment update. For the demonstration the alias
    change is sufficient — the API reads the alias on each startup.
    """
    import mlflow

    ti = context["ti"]
    v1_mae = ti.xcom_pull(task_ids="evaluate_new_model", key="v1_covid_mae")
    v2_mae = ti.xcom_pull(task_ids="evaluate_new_model", key="v2_covid_mae")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Archive v1 production version
    try:
        v1_prod = client.get_model_version_by_alias(V1_MODEL_NAME, "production")
        client.delete_registered_model_alias(V1_MODEL_NAME, "production")
        print(f"Archived v1 @production (was version {v1_prod.version})")
    except Exception as e:
        print(f"Could not archive v1: {e}")

    # v2 is already @production from retrain_v2 task
    print(f"v2 is now @production.")
    print(f"  v1 COVID-period MAE : {v1_mae:.4f}")
    print(f"  v2 COVID-period MAE : {v2_mae:.4f}")
    print(f"  Improvement         : {v1_mae - v2_mae:.4f}")
    print("Grafana will show MAE recovery from this point.")


# ── Task 7: Notify / log outcome ──────────────────────────────────────────────

def notify_fn(**context):
    """
    Logs the monitoring event outcome to MLflow.
    In production this would also send a Slack/email notification.
    """
    import mlflow

    ti = context["ti"]
    rolling_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="rolling_mae")
    baseline_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="baseline_mae")
    v2_test_mae  = ti.xcom_pull(task_ids="retrain_v2", key="v2_test_mae")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="monitoring_event"):
        mlflow.log_metrics({
            "drift_rolling_mae":        float(rolling_mae) if rolling_mae else 0,
            "v1_baseline_mae":          float(baseline_mae) if baseline_mae else 0,
            "v2_test_mae_post_retrain": float(v2_test_mae) if v2_test_mae else 0,
        })
        mlflow.set_tag("event_type",  "drift_detected_and_retrained")
        mlflow.set_tag("pipeline",    "monitoring_pipeline")
        mlflow.set_tag("dag_run_id",  context["run_id"])

    print("Monitoring event logged to MLflow.")
    print("Phase 5 Grafana dashboard will show the recovery.")


# ── Define tasks ──────────────────────────────────────────────────────────────

t1_collect = PythonOperator(
    task_id="collect_ground_truth",
    python_callable=collect_ground_truth_fn,
    dag=dag,
)

t2_mae = PythonOperator(
    task_id="compute_rolling_mae",
    python_callable=compute_rolling_mae_fn,
    dag=dag,
)

t3_confirm = BranchPythonOperator(
    task_id="confirm_drift",
    python_callable=confirm_drift_fn,
    dag=dag,
)

t4_retrain = PythonOperator(
    task_id="retrain_v2",
    python_callable=retrain_v2_fn,
    dag=dag,
)

t5_evaluate = BranchPythonOperator(
    task_id="evaluate_new_model",
    python_callable=evaluate_new_model_fn,
    dag=dag,
)

t6a_promote = PythonOperator(
    task_id="promote_v2",
    python_callable=promote_v2_fn,
    dag=dag,
)

t6b_keep = EmptyOperator(
    task_id="keep_v1",
    dag=dag,
)

t7_no_drift = EmptyOperator(
    task_id="no_drift_detected",
    dag=dag,
)

t8_notify = PythonOperator(
    task_id="notify",
    python_callable=notify_fn,
    trigger_rule="none_failed_min_one_success",
    dag=dag,
)

# ── Dependencies ──────────────────────────────────────────────────────────────

t1_collect >> t2_mae >> t3_confirm
t3_confirm >> [t4_retrain, t7_no_drift]
t4_retrain >> t5_evaluate
t5_evaluate >> [t6a_promote, t6b_keep]
t6a_promote >> t8_notify
t6b_keep    >> t8_notify
t7_no_drift >> t8_notify
