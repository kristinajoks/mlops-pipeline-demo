import pandas as pd
from pathlib import Path
from src.features.feature_columns import (
    SENSING_FEATURES, ALL_FEATURES, COVID_FEATURES, TARGET
)

SPLITS_DIR = Path("data/processed/splits")

def test_sensing_features_in_pre_covid_train():
    df = pd.read_csv(SPLITS_DIR / "pre_covid_train.csv")
    missing = [f for f in SENSING_FEATURES if f not in df.columns]
    assert not missing, f"Missing sensing features: {missing}"

def test_all_features_in_full_train():
    df = pd.read_csv(SPLITS_DIR / "full_train.csv")
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    assert not missing, f"Missing features in full_train: {missing}"

def test_no_covid_cols_in_pre_covid_train():
    df = pd.read_csv(SPLITS_DIR / "pre_covid_train.csv")
    present = [f for f in COVID_FEATURES if f in df.columns]
    assert not present, f"COVID columns found in pre_covid_train: {present}"

def test_target_in_all_splits():
    for split in ["pre_covid_train", "pre_covid_val", "pre_covid_test",
                  "full_train", "full_val", "full_test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        assert TARGET in df.columns, f"{TARGET} missing from {split}"

def test_no_duplicate_features():
    assert len(SENSING_FEATURES) == len(set(SENSING_FEATURES))
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

def test_sensing_features_subset_of_all():
    assert set(SENSING_FEATURES).issubset(set(ALL_FEATURES))