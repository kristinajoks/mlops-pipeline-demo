"""
Unit and integration tests for feature_columns.py
"""

import pytest
import pandas as pd
from pathlib import Path

from src.features.feature_columns import (
    SENSING_FEATURES, COVID_FEATURES, ALL_FEATURES, TARGET, META_COLS
)

PROJECT_ROOT = Path(__file__).parent.parent
SPLITS_DIR   = PROJECT_ROOT / "data" / "processed" / "splits"

requires_data = pytest.mark.skipif(
    not (SPLITS_DIR / "pre_covid_train.csv").exists(),
    reason="Processed data files not available - run locally with DVC pull"
)


#  Unit tests (run in CI) 
class TestFeatureColumnLists:

    def test_sensing_features_not_empty(self):
        assert len(SENSING_FEATURES) > 0

    def test_all_features_not_empty(self):
        assert len(ALL_FEATURES) > 0

    def test_sensing_features_count(self):
        # 10 features x mean+std + is_ios + label_lag1
        assert len(SENSING_FEATURES) == 22

    def test_covid_features_count(self):
        # 9 COVID items + covid_period 
        assert len(COVID_FEATURES) == 10

    def test_no_duplicate_sensing_features(self):
        assert len(SENSING_FEATURES) == len(set(SENSING_FEATURES))

    def test_no_duplicate_all_features(self):
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES))

    def test_sensing_features_subset_of_all(self):
        assert set(SENSING_FEATURES).issubset(set(ALL_FEATURES))

    def test_covid_features_subset_of_all(self):
        assert set(COVID_FEATURES).issubset(set(ALL_FEATURES))

    def test_target_is_string(self):
        assert isinstance(TARGET, str)
        assert TARGET == "label_composite_score"

    def test_target_not_in_features(self):
        assert TARGET not in SENSING_FEATURES
        assert TARGET not in ALL_FEATURES

    def test_uid_not_in_features(self):
        assert "uid" not in SENSING_FEATURES
        assert "uid" not in ALL_FEATURES

    def test_year_week_not_in_features(self):
        assert "year_week" not in SENSING_FEATURES
        assert "year_week" not in ALL_FEATURES

    def test_covid9_not_in_covid_features(self):
        assert "COVID-9" not in COVID_FEATURES

    def test_covid_period_in_covid_features(self):
        assert "covid_period" in COVID_FEATURES

    def test_is_ios_in_sensing_features(self):
        assert "is_ios" in SENSING_FEATURES

    def test_mean_std_pairs_exist(self):
        non_stat_features = {"is_ios"}
        base_features = set()
        for f in SENSING_FEATURES:
            if f in non_stat_features:
                continue
            if f.endswith("_mean"):
                base_features.add(f[:-5])
            elif f.endswith("_std"):
                base_features.add(f[:-4])
        for base in base_features:
            assert f"{base}_mean" in SENSING_FEATURES, \
                f"Missing {base}_mean"
            assert f"{base}_std" in SENSING_FEATURES, \
                f"Missing {base}_std"


# Integration tests (run locally)
class TestFeatureColumnsAgainstData:

    @requires_data
    def test_sensing_features_in_pre_covid_train(self):
        df = pd.read_csv(SPLITS_DIR / "pre_covid_train.csv")
        missing = [f for f in SENSING_FEATURES if f not in df.columns]
        assert not missing, f"Missing sensing features: {missing}"

    @requires_data
    def test_all_features_in_full_train(self):
        df = pd.read_csv(SPLITS_DIR / "full_train.csv")
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        assert not missing, f"Missing features in full_train: {missing}"

    @requires_data
    def test_no_covid_cols_in_pre_covid_train(self):
        df = pd.read_csv(SPLITS_DIR / "pre_covid_train.csv")
        present = [f for f in COVID_FEATURES if f in df.columns]
        assert not present, f"COVID columns in pre_covid_train: {present}"

    @requires_data
    def test_target_in_all_splits(self):
        for split in ["pre_covid_train", "pre_covid_val", "pre_covid_test",
                      "full_train", "full_val", "full_test"]:
            df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
            assert TARGET in df.columns, f"{TARGET} missing from {split}"

    @requires_data
    def test_pre_covid_splits_correct_col_count(self):
        for split in ["pre_covid_train", "pre_covid_val", "pre_covid_test"]:
            df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
            # uid, year_week, label, n_surveys + 21 features = 25
            assert df.shape[1] == 25, \
                f"{split} has {df.shape[1]} cols, expected 25"

    @requires_data
    def test_full_splits_correct_col_count(self):
        for split in ["full_train", "full_val", "full_test"]:
            df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
            # uid, year_week, label, n_surveys + 21 sensing + 10 covid 
            assert df.shape[1] == 35, \
                f"{split} has {df.shape[1]} cols, expected 35"