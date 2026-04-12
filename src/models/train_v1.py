"""
v1 training pipeline — Ridge baseline and LightGBM.
"""

from xml.parsers.expat import model

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.features.feature_columns import SENSING_FEATURES, TARGET
from src.features.preprocessing import get_X_y, fit_scaler, apply_scaler, impute_for_ridge

PROJECT_ROOT = Path(__file__).parent.parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

IOS_ONLY_COLS = [
    "other_playing_duration_ep_0_mean",
    "other_playing_duration_ep_0_std",
]

# Loads pre-COVID train/val/test splits.
def load_v1_splits():
    train = pd.read_csv(SPLITS_DIR / "pre_covid_train.csv")
    val   = pd.read_csv(SPLITS_DIR / "pre_covid_val.csv")
    test  = pd.read_csv(SPLITS_DIR / "pre_covid_test.csv")
    return train, val, test

# Replaces NaN with 0 for iOS-only features in Android rows
def fix_android_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "is_ios" not in X.columns:
        return X
    android_mask = X["is_ios"] == 0
    for col in IOS_ONLY_COLS:
        if col in X.columns:
            X.loc[android_mask, col] = X.loc[android_mask, col].fillna(0)
    return X

# Runs prediction and returns MAE and R2.
def evaluate(model, X, y, scaler=None):
    X_input = apply_scaler(X, scaler) if scaler else X
    preds = model.predict(X_input)
    return {
        "mae": round(float(mean_absolute_error(y, preds)), 4),
        "r2":  round(float(r2_score(y, preds)), 4),
    }

# Trains Ridge regression baseline on v1 sensing features.
def train_ridge(alpha: float = 1.0, run_name: str = "v1_ridge_baseline"):
    train, val, test = load_v1_splits()

    X_train, y_train = get_X_y(train, SENSING_FEATURES)
    X_val,   y_val   = get_X_y(val,   SENSING_FEATURES)
    X_test,  y_test  = get_X_y(test,  SENSING_FEATURES)

    # Impute NaN 
    X_train, medians = impute_for_ridge(X_train, SENSING_FEATURES)
    X_val,   _       = impute_for_ridge(X_val,   SENSING_FEATURES, medians)
    X_test,  _       = impute_for_ridge(X_test,  SENSING_FEATURES, medians)

    scaler  = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s   = apply_scaler(X_val,   scaler)
    X_test_s  = apply_scaler(X_test,  scaler)

    model = Ridge(alpha=alpha)
    model.fit(X_train_s, y_train)

    train_metrics = evaluate(model, X_train_s, y_train)
    val_metrics   = evaluate(model, X_val_s,   y_val)
    test_metrics  = evaluate(model, X_test_s,  y_test)

    params = {
        "model_type":       "ridge",
        "alpha":            alpha,
        "dataset_version":  "v1-pre-covid",
        "n_train":          len(X_train),
        "n_val":            len(X_val),
        "n_test":           len(X_test),
        "n_features":       len(SENSING_FEATURES),
    }
    metrics = {
        "train_mae": train_metrics["mae"],
        "val_mae":   val_metrics["mae"],
        "test_mae":  test_metrics["mae"],
        "train_r2":  train_metrics["r2"],
        "val_r2":    val_metrics["r2"],
        "test_r2":   test_metrics["r2"],
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model, name="model")
        import pickle, tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = os.path.join(tmp, "scaler.pkl")
            medians_path = os.path.join(tmp, "imputation_medians.pkl")
            with open(scaler_path,  "wb") as f: pickle.dump(scaler,  f)
            with open(medians_path, "wb") as f: pickle.dump(medians, f)
            mlflow.log_artifact(scaler_path)
            mlflow.log_artifact(medians_path)

    print(f"Ridge (alpha={alpha})")
    print(f"  train MAE: {train_metrics['mae']}  R2: {train_metrics['r2']}")
    print(f"  val   MAE: {val_metrics['mae']}   R2: {val_metrics['r2']}")
    print(f"  test  MAE: {test_metrics['mae']}   R2: {test_metrics['r2']}")

    return model, scaler, medians, metrics

# Trains LightGBM on v1 sensing features
def train_lightgbm(
    n_estimators:   int   = 1000,
    max_depth:      int   = 6,
    num_leaves:     int   = 31,
    learning_rate:  float = 0.05,
    min_child_samples: int = 20,
    run_name: str = "v1_lightgbm",
):
    train, val, test = load_v1_splits()

    X_train, y_train = get_X_y(train, SENSING_FEATURES)
    X_val,   y_val   = get_X_y(val,   SENSING_FEATURES)
    X_test,  y_test  = get_X_y(test,  SENSING_FEATURES)

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
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    train_metrics = evaluate(model, X_train, y_train)
    val_metrics   = evaluate(model, X_val,   y_val)
    test_metrics  = evaluate(model, X_test,  y_test)

    params = {
        "model_type":         "lightgbm",
        "n_estimators":       n_estimators,
        "best_iteration":     model.best_iteration_,
        "max_depth":          max_depth,
        "num_leaves":         num_leaves,
        "learning_rate":      learning_rate,
        "min_child_samples":  min_child_samples,
        "dataset_version":    "v1-pre-covid",
        "n_train":            len(X_train),
        "n_features":         len(SENSING_FEATURES),
    }
    metrics = {
        "train_mae": train_metrics["mae"],
        "val_mae":   val_metrics["mae"],
        "test_mae":  test_metrics["mae"],
        "train_r2":  train_metrics["r2"],
        "val_r2":    val_metrics["r2"],
        "test_r2":   test_metrics["r2"],
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(lgb_model=model, name="model")
        import tempfile, os

        fi = pd.DataFrame({
            "feature":    SENSING_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        with tempfile.TemporaryDirectory() as tmp:
            fi_path = os.path.join(tmp, "feature_importance.csv")
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path)

    print(f"LightGBM (best_iteration={model.best_iteration_})")
    print(f"  train MAE: {train_metrics['mae']}  R2: {train_metrics['r2']}")
    print(f"  val   MAE: {val_metrics['mae']}   R2: {val_metrics['r2']}")
    print(f"  test  MAE: {test_metrics['mae']}   R2: {test_metrics['r2']}")

    return model, metrics