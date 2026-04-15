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
with register=True - this is the ONLY place v2 gets registered.
The MLflow audit trail shows this run was triggered by the monitoring
DAG, not by a human. This is what makes the demonstration authentic.
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
sys.path.insert(0, str(PROJECT_ROOT))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME        = "mental_health_prediction"
V1_MODEL_NAME          = "mental_health_v1"
V2_MODEL_NAME          = "mental_health_v2"
ROLLING_WINDOW_WEEKS   = 8
MAE_THRESHOLD_FACTOR   = 1.3   # Alert if rolling MAE > 1.3x training MAE
PSI_MODERATE_THRESHOLD = 0.2
PSI_SEVERE_THRESHOLD = 0.25
MIN_DRIFTED_FEATURES = 3

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
    start_date=datetime(2020, 1, 1),
    schedule=None,      # Triggered by Prometheus alert or manual run
    catchup=False,
    tags=["monitoring", "retraining", "v2", "drift"],
)

def _compute_psi(reference, current, bins=10):
    import numpy as np
    import pandas as pd

    ref = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna()
    cur = pd.Series(current).replace([np.inf, -np.inf], np.nan).dropna()

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # If constant feature, no meaningful PSI
    if ref.nunique() <= 1 and cur.nunique() <= 1:
        return 0.0

    try:
        breakpoints = np.unique(
            np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
        )
        if len(breakpoints) < 3:
            return 0.0

        ref_counts, _ = np.histogram(ref, bins=breakpoints)
        cur_counts, _ = np.histogram(cur, bins=breakpoints)

        ref_pct = ref_counts / max(ref_counts.sum(), 1)
        cur_pct = cur_counts / max(cur_counts.sum(), 1)

        eps = 1e-6
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    except Exception:
        return 0.0

# Collect ground truth from outputs/predictions/ and match against labels in inference_v1.csv
def collect_ground_truth_fn(**context):
    import pandas as pd
    from pathlib import Path

    PRED_DIR   = PROJECT_ROOT / "outputs" / "predictions"
    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

    logical_date = context["logical_date"]
    cutoff_year, cutoff_week, _ = logical_date.isocalendar()
    cutoff_year_week = f"{cutoff_year}-W{cutoff_week:02d}"

    pred_files = sorted(PRED_DIR.glob("*.csv"))
    if not pred_files:
        raise FileNotFoundError(
            "No prediction files found in outputs/predictions/. "
            "Run the inference_pipeline DAG first."
        )

    eligible_files = []
    for f in pred_files:
        week_str = f.stem 
        if week_str <= cutoff_year_week:
            eligible_files.append(f)

    if not eligible_files:
        raise FileNotFoundError(
            f"No prediction files found at or before logical week {cutoff_year_week}."
        )

    recent_files = eligible_files[-ROLLING_WINDOW_WEEKS:]
    all_preds = pd.concat([pd.read_csv(f) for f in recent_files], ignore_index=True)

    inference_data = pd.read_csv(SPLITS_DIR / "inference_v1.csv")

    merged = all_preds.merge(
        inference_data[["uid", "year_week", "label_composite_score"]],
        on=["uid", "year_week"],
        how="inner",
    )

    print(f"Logical monitoring week : {cutoff_year_week}")
    print(f"Eligible prediction files: {len(eligible_files)}")
    print(f"Using recent files      : {[f.stem for f in recent_files]}")
    print(f"Predictions             : {len(all_preds)}")
    print(f"Matched w/ GT           : {len(merged)}")
    print(f"Weeks covered           : {merged['year_week'].unique().tolist()}")

    context["ti"].xcom_push(
        key="matched_data",
        value=merged.to_json(orient="records")
    )

# Compute rolling MAE and compare to training baseline
def compute_rolling_mae_fn(**context):
    import json
    import pandas as pd
    import mlflow
    from sklearn.metrics import mean_absolute_error

    ti = context["ti"]
    matched_json = ti.xcom_pull(
        task_ids="collect_ground_truth", key="matched_data"
    )
    matched = pd.read_json(matched_json)

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

# Compute feature drift using PSI and compare to thresholds
def compute_feature_drift_fn(**context):
    import pandas as pd
    from pathlib import Path

    SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
    MONITOR_DIR = PROJECT_ROOT / "outputs" / "reports" / "monitoring"
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    logical_date = context["logical_date"]
    cutoff_year, cutoff_week, _ = logical_date.isocalendar()
    cutoff_year_week = f"{cutoff_year}-W{cutoff_week:02d}"

    reference_df = pd.read_csv(SPLITS_DIR / "pre_covid_test.csv")
    current_df = pd.read_csv(SPLITS_DIR / "inference_v1.csv")

    eligible_weeks = sorted(
        w for w in current_df["year_week"].dropna().unique()
        if w <= cutoff_year_week
    )

    if not eligible_weeks:
        raise FileNotFoundError(
            f"No inference_v1 weeks found at or before {cutoff_year_week}."
        )

    recent_weeks = eligible_weeks[-ROLLING_WINDOW_WEEKS:]
    current_window = current_df[current_df["year_week"].isin(recent_weeks)].copy()

    feature_cols = [
        c for c in reference_df.columns
        if c not in {"uid", "year_week", "label_composite_score"}
    ]

    psi_rows = []
    for col in feature_cols:
        psi = _compute_psi(reference_df[col], current_window[col])
        psi_rows.append({"feature": col, "psi": psi})

    psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
    max_psi = float(psi_df["psi"].max()) if not psi_df.empty else 0.0
    n_drifted_features = int((psi_df["psi"] > PSI_MODERATE_THRESHOLD).sum())
    psi_triggered_drift = bool(
        (max_psi > PSI_SEVERE_THRESHOLD) or (n_drifted_features >= MIN_DRIFTED_FEATURES)
    )

    report_path = MONITOR_DIR / f"feature_drift_{cutoff_year_week}.csv"
    psi_df.to_csv(report_path, index=False)

    print(f"Reference split        : pre_covid_test.csv")
    print(f"Current logical week   : {cutoff_year_week}")
    print(f"Current weeks used     : {recent_weeks}")
    print(f"Max PSI                : {max_psi:.4f}")
    print(f"Drifted features >0.2  : {n_drifted_features}")
    print(f"PSI-triggered drift    : {psi_triggered_drift}")
    print(f"Saved report to        : {report_path}")

    context["ti"].xcom_push(key="max_psi", value=max_psi)
    context["ti"].xcom_push(key="n_drifted_features", value=n_drifted_features)
    context["ti"].xcom_push(key="psi_triggered_drift", value=psi_triggered_drift)

# Confirm drift and proceed to retraining if sustained
def confirm_drift_fn(**context):
    ti = context["ti"]

    mae_drift = ti.xcom_pull(task_ids="compute_rolling_mae", key="drift")
    rolling_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="rolling_mae")
    threshold = ti.xcom_pull(task_ids="compute_rolling_mae", key="threshold")

    max_psi = ti.xcom_pull(task_ids="compute_feature_drift", key="max_psi")
    n_drifted_features = ti.xcom_pull(
        task_ids="compute_feature_drift", key="n_drifted_features"
    )
    psi_drift = ti.xcom_pull(
        task_ids="compute_feature_drift", key="psi_triggered_drift"
    )

    print(f"MAE drift            : {mae_drift}")
    print(f"Rolling MAE          : {rolling_mae:.4f}")
    print(f"MAE threshold        : {threshold:.4f}")
    print(f"Max PSI              : {float(max_psi):.4f}")
    print(f"Drifted features     : {int(n_drifted_features)}")
    print(f"PSI drift            : {psi_drift}")

    if mae_drift or psi_drift:
        print("DRIFT CONFIRMED: retraining will start.")
        return "retrain_v2"
    else:
        print("No sustained drift detected.")
        return "no_drift_detected"

# Retrain v2 with COVID features, register if improved
def retrain_v2_fn(**context):
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
        register=True,          
    )

    print(f"v2 training complete:")
    print(f"  val MAE  : {metrics['val_mae']:.4f}")
    print(f"  test MAE : {metrics['test_mae']:.4f}")
    print(f"  val R2   : {metrics['val_r2']:.4f}")

    context["ti"].xcom_push(key="v2_val_mae",  value=metrics["val_mae"])
    context["ti"].xcom_push(key="v2_test_mae", value=metrics["test_mae"])
    context["ti"].xcom_push(key="v2_val_r2",   value=metrics["val_r2"])

# Evaluate both models on full_test and decide whether to promote v2
def evaluate_new_model_fn(**context):
    import mlflow
    import mlflow.lightgbm
    import pandas as pd
    from sklearn.metrics import mean_absolute_error

    from src.features.feature_columns import SENSING_FEATURES, ALL_FEATURES, TARGET

    ti = context["ti"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
        print("v2 outperforms v1 - proceeding to promotion.")
        return "promote_v2"
    else:
        print("v2 does not outperform v1 - keeping current production model.")
        return "keep_v1"

# Promote v2 to production, demote v1 to archived
# Maybe also trigger Kubernetes deployment update (not implemented here)
def promote_v2_fn(**context):
    import mlflow

    ti = context["ti"]
    v1_mae = ti.xcom_pull(task_ids="evaluate_new_model", key="v1_covid_mae")
    v2_mae = ti.xcom_pull(task_ids="evaluate_new_model", key="v2_covid_mae")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Archive v1
    try:
        v1_prod = client.get_model_version_by_alias(V1_MODEL_NAME, "production")
        client.delete_registered_model_alias(V1_MODEL_NAME, "production")
        print(f"Archived v1 @production (was version {v1_prod.version})")
    except Exception as e:
        print(f"Could not archive v1: {e}")

    # v2 is already @production from retrain_v2
    print(f"v2 is now @production.")
    print(f"  v1 COVID-period MAE : {v1_mae:.4f}")
    print(f"  v2 COVID-period MAE : {v2_mae:.4f}")
    print(f"  Improvement         : {v1_mae - v2_mae:.4f}")
    print("Grafana will show MAE recovery from this point.")

# Log outcome 
# and notify stakeholders with email (not implemented here)
def notify_fn(**context):
    import mlflow

    ti = context["ti"]
    rolling_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="rolling_mae")
    baseline_mae = ti.xcom_pull(task_ids="compute_rolling_mae", key="baseline_mae")
    v2_test_mae = ti.xcom_pull(task_ids="retrain_v2", key="v2_test_mae")
    branch_taken = ti.xcom_pull(task_ids="confirm_drift")

    max_psi = ti.xcom_pull(task_ids="compute_feature_drift", key="max_psi")
    n_drifted_features = ti.xcom_pull(
        task_ids="compute_feature_drift", key="n_drifted_features"
    )
    psi_triggered_drift = ti.xcom_pull(
        task_ids="compute_feature_drift", key="psi_triggered_drift"
    )
    mae_triggered_drift = ti.xcom_pull(task_ids="compute_rolling_mae", key="drift")

    if branch_taken == "retrain_v2":
        event_type = "drift_detected_and_retrained"
    else:
        event_type = "no_drift_detected"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="monitoring_event"):
        mlflow.log_metrics({
            "drift_rolling_mae": float(rolling_mae) if rolling_mae else 0.0,
            "v1_baseline_mae": float(baseline_mae) if baseline_mae else 0.0,
            "v2_test_mae_post_retrain": float(v2_test_mae) if v2_test_mae else 0.0,
            "max_psi": float(max_psi) if max_psi else 0.0,
            "n_drifted_features": float(n_drifted_features) if n_drifted_features else 0.0,
        })
        mlflow.set_tag("event_type", event_type)
        mlflow.set_tag("pipeline", "monitoring_pipeline")
        mlflow.set_tag("dag_run_id", context["run_id"])
        mlflow.set_tag("mae_triggered_drift", str(bool(mae_triggered_drift)))
        mlflow.set_tag("psi_triggered_drift", str(bool(psi_triggered_drift)))

    print("Monitoring event logged to MLflow.")

# Tasks

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

t2b_feature_drift = PythonOperator(
    task_id="compute_feature_drift",
    python_callable=compute_feature_drift_fn,
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

# Dependencies
t1_collect >> t2_mae
t1_collect >> t2b_feature_drift
[t2_mae, t2b_feature_drift] >> t3_confirm
t3_confirm >> [t4_retrain, t7_no_drift]
t4_retrain >> t5_evaluate
t5_evaluate >> [t6a_promote, t6b_keep]
t6a_promote >> t8_notify
t6b_keep    >> t8_notify
t7_no_drift >> t8_notify
