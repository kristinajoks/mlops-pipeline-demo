"""
Preprocessor for inference requests.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.feature_columns import SENSING_FEATURES, ALL_FEATURES

IOS_ONLY_COLS = [
    "other_playing_duration_ep_0_mean",
    "other_playing_duration_ep_0_std",
]

# Converts request features dict to single-row DataFrame 
# with correct column order
def request_to_dataframe(features: dict, feature_list: list) -> pd.DataFrame:
    row = {col: features.get(col, np.nan) for col in feature_list}
    return pd.DataFrame([row], columns=feature_list)

def fix_android_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "is_ios" not in X.columns:
        return X
    android_mask = X["is_ios"] == 0
    for col in IOS_ONLY_COLS:
        if col in X.columns:
            X.loc[android_mask, col] = X.loc[android_mask, col].fillna(0.0)
    return X

def preprocess_for_inference(features: dict, feature_list: list) -> pd.DataFrame:
    X = request_to_dataframe(features, feature_list)
    X = fix_android_nan(X)
    return X