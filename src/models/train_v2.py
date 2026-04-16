"""
v2 training pipeline - LightGBM with sensing + COVID features.

Key differences from train_v1:
  - Loads full_train/val/test (2017-W37 to 2021-W26, all data at retraining time)
  - Uses ALL_FEATURES: 22 sensing + 9 COVID items + covid_period = 32 features
  - LightGBM only — native NaN handles 57.9% missing COVID values
  - Does NOT register model by default — pass register=True for later retraining phase
"""

import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

from src.features.feature_columns import ALL_FEATURES, TARGET
from src.features.preprocessing import get_X_y

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLITS_DIR   = PROJECT_ROOT / "data" / "processed" / "splits"

IOS_ONLY_COLS = [
    "other_playing_duration_ep_0_mean",
    "other_playing_duration_ep_0_std",
]

def load_v2_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / "full_train.csv")
    val   = pd.read_csv(SPLITS_DIR / "full_val.csv")
    test  = pd.read_csv(SPLITS_DIR / "full_test.csv")
    return train, val, test

def fix_android_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "is_ios" not in X.columns:
        return X
    android_mask = X["is_ios"] == 0
    for col in IOS_ONLY_COLS:
        if col in X.columns:
            X.loc[android_mask, col] = X.loc[android_mask, col].fillna(0)
    return X

def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    return {
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "r2":  round(float(r2_score(y, preds)), 4),
    }

# Trains LightGBM on v2 sensing + COVID features
def train_lightgbm_v2(
    n_estimators:      int   = 1000,
    max_depth:         int   = 6,
    num_leaves:        int   = 31,
    learning_rate:     float = 0.05,
    min_child_samples: int   = 20,
    run_name:          str   = "v2_lgbm_experiment",
    register:          bool  = False,
) -> tuple:
    train, val, test = load_v2_splits()

    X_train, y_train = get_X_y(train, ALL_FEATURES)
    X_val,   y_val   = get_X_y(val,   ALL_FEATURES)
    X_test,  y_test  = get_X_y(test,  ALL_FEATURES)

    X_train = fix_android_nan(X_train)
    X_val   = fix_android_nan(X_val)
    X_test  = fix_android_nan(X_test)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    train_metrics = evaluate(model, X_train, y_train)
    val_metrics   = evaluate(model, X_val,   y_val)
    test_metrics  = evaluate(model, X_test,  y_test)

    params = {
        "model_type":         "lightgbm",
        "dataset_version":    "v2-full",
        "n_features":         len(ALL_FEATURES),
        "n_train":            len(X_train),
        "n_estimators":       n_estimators,
        "best_iteration":     model.best_iteration_,
        "max_depth":          max_depth,
        "num_leaves":         num_leaves,
        "learning_rate":      learning_rate,
        "min_child_samples":  min_child_samples,
        "covid_features":     True,
        "lag_feature":        True,
    }
    metrics = {
        "train_mae": train_metrics["mae"],
        "val_mae":   val_metrics["mae"],
        "test_mae":  test_metrics["mae"],
        "train_r2":  train_metrics["r2"],
        "val_r2":    val_metrics["r2"],
        "test_r2":   test_metrics["r2"],
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_version", "v2")
        mlflow.set_tag("registered", str(register))
        mlflow.lightgbm.log_model(lgb_model=model, name="model")

        import tempfile, os
        fi = pd.DataFrame({
            "feature":    ALL_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        with tempfile.TemporaryDirectory() as tmp:
            fi_path = os.path.join(tmp, "v2_feature_importance.csv")
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

        run_id = run.info.run_id

    print(f"LightGBM v2 (best_iteration={model.best_iteration_})")
    print(f"  train MAE: {train_metrics['mae']}  R2: {train_metrics['r2']}")
    print(f"  val   MAE: {val_metrics['mae']}   R2: {val_metrics['r2']}")
    print(f"  test  MAE: {test_metrics['mae']}   R2: {test_metrics['r2']}")

    # Registration — only when called by retraining phase
    if register:
        model_uri = f"runs:/{run_id}/model"
        registered = mlflow.register_model(model_uri, "mental_health_v2")
        print(f"Registered: mental_health_v2 v{registered.version}")

    return model, metrics