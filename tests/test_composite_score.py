"""
Unit tests for src/labels/composite_score.py
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels.composite_score import (
    cronbach_alpha,
    normalize_instrument,
    prepare_sse,
    compute_composite,
    apply_next_week_shift,
)

# Fixtures 
@pytest.fixture
def sample_ema_df():
    """Small DataFrame simulating processed general_ema rows."""
    return pd.DataFrame({
        "uid":          ["A", "A", "B"],
        "year_week":    ["2019-W01", "2019-W02", "2019-W01"],
        "day":          [20190101, 20190108, 20190101],
        "phq4_score":   [0.0, 12.0, 6.0],
        "stress":       [1.0, 5.0, 3.0],
        "social_level": [5.0, 1.0, 3.0],
        "sse3-1":       [5.0, 1.0, 3.0],
        "sse3-2":       [5.0, 1.0, 3.0],
        "sse3-3":       [5.0, 1.0, 3.0],
        "sse3-4":       [5.0, 1.0, 3.0],
        "has_response": [True, True, True],
        "composite_score": [np.nan, np.nan, np.nan],  # filled by compute_composite
    })


# cronbach_alpha 
class TestCronbachAlpha:

    def test_perfect_consistency(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
        })
        assert cronbach_alpha(df) == 1.0

    def test_returns_float(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4],
            "b": [2, 3, 4, 5],
        })
        result = cronbach_alpha(df)
        assert isinstance(result, float)

    def test_nan_for_single_item(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert np.isnan(cronbach_alpha(df))

    def test_nan_for_empty_df(self):
        df = pd.DataFrame({"a": [], "b": []})
        assert np.isnan(cronbach_alpha(df))

    def test_nan_rows_dropped(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4],
            "b": [2, 3, 4,      5],
        })
        result = cronbach_alpha(df)
        assert not np.isnan(result)

    def test_phq4_like_items_high_alpha(self):
        np.random.seed(42)
        base = np.random.normal(5, 2, 200)
        df = pd.DataFrame({
            "a": base + np.random.normal(0, 0.1, 200),
            "b": base + np.random.normal(0, 0.1, 200),
            "c": base + np.random.normal(0, 0.1, 200),
            "d": base + np.random.normal(0, 0.1, 200),
        })
        assert cronbach_alpha(df) > 0.9

    def test_uncorrelated_items_low_alpha(self):
        np.random.seed(0)
        df = pd.DataFrame(np.random.normal(0, 1, (200, 4)),
                          columns=["a", "b", "c", "d"])
        assert cronbach_alpha(df) < 0.3


# normalize_instrument 
class TestNormalizeInstrument:

    def test_min_value_maps_to_zero(self):
        s = pd.Series([0.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=False)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_max_value_maps_to_one(self):
        s = pd.Series([12.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=False)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_invert_flips_values(self):
        s = pd.Series([0.0, 12.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=True)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(0.0)

    def test_midpoint_maps_to_half(self):
        s = pd.Series([6.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=False)
        assert result.iloc[0] == pytest.approx(0.5)

    def test_out_of_range_clipped(self):
        s = pd.Series([-5.0, 20.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=False)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_nan_preserved(self):
        s = pd.Series([np.nan, 6.0])
        result = normalize_instrument(s, min_val=0, max_val=12, invert=False)
        assert np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[1])


# prepare_sse 
class TestPrepareSSE:

    def test_adds_sse3_1_r_column(self, sample_ema_df):
        result = prepare_sse(sample_ema_df)
        assert "sse3-1_r" in result.columns

    def test_adds_sse_score_column(self, sample_ema_df):
        result = prepare_sse(sample_ema_df)
        assert "sse_score" in result.columns

    def test_reverse_coding_correct(self, sample_ema_df):
        result = prepare_sse(sample_ema_df)
        assert result["sse3-1_r"].iloc[0] == pytest.approx(1.0)
        assert result["sse3-1_r"].iloc[1] == pytest.approx(5.0)

    def test_sse_score_range_max(self, sample_ema_df):
        result = prepare_sse(sample_ema_df)
        assert result["sse_score"].iloc[0] == pytest.approx(16.0)

    def test_sse_score_range_min(self, sample_ema_df):
        result = prepare_sse(sample_ema_df)
        assert result["sse_score"].iloc[1] == pytest.approx(8.0)

    def test_does_not_modify_original(self, sample_ema_df):
        cols_before = set(sample_ema_df.columns)
        prepare_sse(sample_ema_df)
        assert set(sample_ema_df.columns) == cols_before


# compute_composite 
class TestComputeComposite:

    def test_best_possible_score_is_100(self):
        df = pd.DataFrame({
            "phq4_score":   [0.0],
            "stress":       [1.0],
            "social_level": [5.0],
            "sse3-1":       [1.0],  
            "sse3-2":       [5.0],
            "sse3-3":       [5.0],
            "sse3-4":       [5.0],
        })
        df = prepare_sse(df)
        result = compute_composite(df)
        assert result.iloc[0] == pytest.approx(100.0)

    def test_worst_possible_score_is_zero(self):
        df = pd.DataFrame({
            "phq4_score":   [12.0],
            "stress":       [5.0],
            "social_level": [1.0],
            "sse3-1":       [5.0], 
            "sse3-2":       [1.0],
            "sse3-3":       [1.0],
            "sse3-4":       [1.0],
        })
        df = prepare_sse(df)
        result = compute_composite(df)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_midpoint_score_is_fifty(self):
        df = pd.DataFrame({
            "phq4_score":   [6.0],
            "stress":       [3.0],
            "social_level": [3.0],
            "sse3-1":       [3.0],  
            "sse3-2":       [3.0],
            "sse3-3":       [3.0],
            "sse3-4":       [3.0],
        })
        df = prepare_sse(df)
        result = compute_composite(df)
        assert result.iloc[0] == pytest.approx(50.0)

    def test_missing_instrument_produces_nan(self):
        df = pd.DataFrame({
            "phq4_score":   [np.nan],
            "stress":       [3.0],
            "social_level": [3.0],
            "sse3-1":       [3.0],
            "sse3-2":       [3.0],
            "sse3-3":       [3.0],
            "sse3-4":       [3.0],
        })
        df = prepare_sse(df)
        result = compute_composite(df)
        assert np.isnan(result.iloc[0])

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"phq4_score": [1.0]})
        with pytest.raises(ValueError):
            compute_composite(df)

    def test_result_in_valid_range(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "phq4_score":   np.random.uniform(0, 12, 50),
            "stress":       np.random.uniform(1, 5, 50),
            "social_level": np.random.uniform(1, 5, 50),
            "sse3-1":       np.random.uniform(1, 5, 50),
            "sse3-2":       np.random.uniform(1, 5, 50),
            "sse3-3":       np.random.uniform(1, 5, 50),
            "sse3-4":       np.random.uniform(1, 5, 50),
        })
        df = prepare_sse(df)
        result = compute_composite(df).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()


# apply_next_week_shift 
class TestApplyNextWeekShift:

    @pytest.fixture
    def weekly_df(self):
        return pd.DataFrame({
            "uid":              ["A", "A", "A", "B", "B"],
            "year_week":        ["2019-W01", "2019-W02", "2019-W03",
                                 "2019-W01", "2019-W03"],
            "composite_score":  [60.0, 70.0, 80.0, 55.0, 75.0],
            "n_surveys_in_week": [1, 1, 1, 1, 1],
        })

    def test_returns_dataframe(self, weekly_df):
        result = apply_next_week_shift(weekly_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, weekly_df):
        result = apply_next_week_shift(weekly_df)
        assert "uid" in result.columns
        assert "feature_week" in result.columns
        assert "label_week" in result.columns
        assert "label_composite_score" in result.columns

    def test_label_is_next_week_score(self, weekly_df):
        result = apply_next_week_shift(weekly_df)
        # Student A, feature_week=2019-W01, label should be 2019-W02 score=70
        row = result[
            (result["uid"] == "A") & (result["feature_week"] == "2019-W01")
        ]
        assert len(row) == 1
        assert row["label_composite_score"].iloc[0] == pytest.approx(70.0)
        assert row["label_week"].iloc[0] == "2019-W02"

    def test_drops_weeks_without_next_label(self, weekly_df):
        result = apply_next_week_shift(weekly_df)
        # Student B has 2019-W01 and 2019-W03 (gap at W02)
        # W01 has no next week label (W02 missing) 
        student_b = result[result["uid"] == "B"]
        feature_weeks = student_b["feature_week"].tolist()
        assert "2019-W01" not in feature_weeks

    def test_last_week_always_dropped(self, weekly_df):
        result = apply_next_week_shift(weekly_df)
        # Student A: W01->W02, W02->W03 should exist, W03 has no next -> dropped
        student_a = result[result["uid"] == "A"]
        assert "2019-W03" not in student_a["feature_week"].tolist()
        assert len(student_a) == 2
