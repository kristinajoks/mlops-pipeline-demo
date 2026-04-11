"""
Unit tests for src/data/loaders.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import (
    parse_dates,
    classify_ema_rows,
    temporal_coverage,
    weekly_survey_counts,
    per_student_survey_counts
)

# Fixtures
@pytest.fixture
def minimal_ema_df():
    """Minimal general_ema-like DataFrame with two students."""
    return pd.DataFrame({
        "uid":         ["student_A", "student_A", "student_B", "student_B"],
        "day":         [20190101, 20190108, 20190101, 20190108],
        "phq4_score":  [2.0, 4.0, 1.0, np.nan],
        "stress":      [2.0, 3.0, 1.0, np.nan],
        "social_level":[3.0, 2.0, 4.0, np.nan],
        "sse3-1":      [2.0, 3.0, 5.0, np.nan],
        "sse3-2":      [3.0, 4.0, 4.0, np.nan],
        "sse3-3":      [4.0, 3.0, 5.0, np.nan],
        "sse3-4":      [4.0, 5.0, 5.0, np.nan],
        "phq4-1":      [1.0, 2.0, 0.0, np.nan],
        "phq4-2":      [0.0, 1.0, 0.0, np.nan],
        "phq4-3":      [1.0, 1.0, 1.0, np.nan],
        "phq4-4":      [0.0, 0.0, 0.0, np.nan],
        "avg_ema_spent_time": [45.0, 60.0, np.nan, 30.0],
    })

@pytest.fixture
def parsed_df(minimal_ema_df):
    return parse_dates(minimal_ema_df)

@pytest.fixture
def classified_df(parsed_df):
    return classify_ema_rows(parsed_df)

# parse_dates
class TestParseDates:

    def test_adds_date_column(self, minimal_ema_df):
        df = parse_dates(minimal_ema_df)
        assert "date" in df.columns

    def test_adds_year_week_column(self, minimal_ema_df):
        df = parse_dates(minimal_ema_df)
        assert "year_week" in df.columns

    def test_date_format_correct(self, minimal_ema_df):
        df = parse_dates(minimal_ema_df)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_year_week_format(self, minimal_ema_df):
        df = parse_dates(minimal_ema_df)
        assert df["year_week"].str.match(r"^\d{4}-W\d{2}$").all()

    def test_does_not_modify_original(self, minimal_ema_df):
        original_cols = list(minimal_ema_df.columns)
        parse_dates(minimal_ema_df)
        assert list(minimal_ema_df.columns) == original_cols

    def test_known_date_value(self, minimal_ema_df):
        df = parse_dates(minimal_ema_df)
        first = df[df["day"] == 20190101].iloc[0]
        assert first["year_week"] == "2019-W01"

# classify_ema_rows 
class TestClassifyEmaRows:

    def test_adds_has_response_column(self, classified_df):
        assert "has_response" in classified_df.columns

    def test_adds_incomplete_column(self, classified_df):
        assert "incomplete" in classified_df.columns

    def test_adds_no_survey_column(self, classified_df):
        assert "no_survey" in classified_df.columns

    def test_row_with_responses_is_has_response(self, classified_df):
        # student_A row 1 has phq4_score=2.0
        row = classified_df[
            (classified_df["uid"] == "student_A") &
            (classified_df["day"] == 20190101)
        ].iloc[0]
        assert row["has_response"] is True or row["has_response"] == True

    def test_row_with_only_timing_is_incomplete(self, classified_df):
        # student_B row 2: all response cols NaN, avg_ema_spent_time=30.0
        row = classified_df[
            (classified_df["uid"] == "student_B") &
            (classified_df["day"] == 20190108)
        ].iloc[0]
        assert row["incomplete"] is True or row["incomplete"] == True
        assert not (row["has_response"] is True or row["has_response"] == True)

    def test_mutual_exclusivity(self, classified_df):
        bad = classified_df[classified_df["has_response"] & classified_df["no_survey"]]
        assert len(bad) == 0

    def test_does_not_modify_original(self, parsed_df):
        cols_before = set(parsed_df.columns)
        classify_ema_rows(parsed_df)
        assert set(parsed_df.columns) == cols_before


# temporal_coverage 
class TestTemporalCoverage:

    def test_returns_dict(self, parsed_df):
        result = temporal_coverage(parsed_df)
        assert isinstance(result, dict)

    def test_global_start_correct(self, parsed_df):
        result = temporal_coverage(parsed_df)
        assert result["global_start"] == date(2019, 1, 1)

    def test_global_end_correct(self, parsed_df):
        result = temporal_coverage(parsed_df)
        assert result["global_end"] == date(2019, 1, 8)

    def test_n_students_correct(self, parsed_df):
        result = temporal_coverage(parsed_df)
        assert result["n_students"] == 2

    def test_per_student_has_both_students(self, parsed_df):
        result = temporal_coverage(parsed_df)
        assert "student_A" in result["per_student"].index
        assert "student_B" in result["per_student"].index


# weekly_survey_counts 
class TestWeeklySurveyCounts:

    def test_returns_dataframe(self, classified_df):
        result = weekly_survey_counts(classified_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, classified_df):
        result = weekly_survey_counts(classified_df)
        assert "uid" in result.columns
        assert "year_week" in result.columns
        assert "surveys_in_week" in result.columns

    def test_excludes_non_response_rows(self, classified_df):
        # student_B row 2 is incomplete 
        result = weekly_survey_counts(classified_df)
        student_b = result[result["uid"] == "student_B"]
        # Only one week should appear for student_B (the completed survey)
        assert len(student_b) == 1

    def test_counts_are_positive(self, classified_df):
        result = weekly_survey_counts(classified_df)
        assert (result["surveys_in_week"] > 0).all()


# per_student_survey_counts 
class TestPerStudentSurveyCounts:
    def test_returns_dataframe(self, classified_df):
        result = per_student_survey_counts(classified_df)
        assert isinstance(result, pd.DataFrame)

    def test_student_a_has_two_surveys(self, classified_df):
        result = per_student_survey_counts(classified_df)
        count_a = result[result["uid"] == "student_A"]["total_surveys"].iloc[0]
        assert count_a == 2

    def test_student_b_has_one_survey(self, classified_df):
        result = per_student_survey_counts(classified_df)
        count_b = result[result["uid"] == "student_B"]["total_surveys"].iloc[0]
        assert count_b == 1

    def test_sorted_descending(self, classified_df):
        result = per_student_survey_counts(classified_df)
        counts = result["total_surveys"].tolist()
        assert counts == sorted(counts, reverse=True)
