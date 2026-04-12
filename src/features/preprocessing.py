import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.features.feature_columns import (
    SENSING_FEATURES, COVID_FEATURES, ALL_FEATURES, TARGET, META_COLS
)

# Separates features and target
def get_X_y(df: pd.DataFrame, features: list) -> tuple:
    X = df[features].copy()
    y = df[TARGET].copy()
    return X, y

# Training data only
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

# Applies a pre-fitted scaler. 
def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)

# Fills in missing values
def impute_for_ridge(df: pd.DataFrame, features: list,
    medians: dict = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if medians is None:
        medians = {col: df[col].median() for col in features if df[col].isna().any()}
    for col, median in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    return df, medians