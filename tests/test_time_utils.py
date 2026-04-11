"""
Unit tests for src/utils/time_utils.py
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.time_utils import (
    to_iso_year_week,
    next_iso_week,
    compute_gap_weeks,
    epoch_to_hour,
    epoch_series_to_hour,
    circular_mean_hours,
    circular_std_hours,
    aggregate_sleep_times,
)

# to_iso_year_week 
class TestToIsoYearWeek:

    def test_known_date(self):
        s = pd.to_datetime(pd.Series(["2019-01-07"]))
        result = to_iso_year_week(s)
        assert result.iloc[0] == "2019-W02"

    def test_year_boundary_late_december(self):
        s = pd.to_datetime(pd.Series(["2018-12-31"]))
        result = to_iso_year_week(s)
        assert result.iloc[0] == "2019-W01"

    def test_output_format(self):
        s = pd.to_datetime(pd.Series(["2020-03-18"]))
        result = to_iso_year_week(s)
        assert result.iloc[0].startswith("2020-W")
        assert len(result.iloc[0]) == 8 

    def test_week_zero_padded(self):
        s = pd.to_datetime(pd.Series(["2019-01-01"]))
        result = to_iso_year_week(s)
        parts = result.iloc[0].split("-W")
        assert len(parts[1]) == 2

# next_iso_week
class TestNextIsoWeek:

    def test_simple_increment(self):
        assert next_iso_week("2019-W01") == "2019-W02"

    def test_year_rollover(self):
        result = next_iso_week("2019-W52")
        assert result == "2020-W01"

    def test_output_format(self):
        result = next_iso_week("2020-W10")
        assert result.startswith("202")
        assert "-W" in result

    def test_week_50_to_51(self):
        assert next_iso_week("2019-W50") == "2019-W51"

# epoch_to_hour
class TestEpochToHour:

    def test_zero_is_8pm(self):
        assert epoch_to_hour(0) == pytest.approx(20.0)

    def test_eight_is_9pm(self):
        assert epoch_to_hour(8) == pytest.approx(21.0)

    def test_midnight_wrap(self):
        assert epoch_to_hour(32) == pytest.approx(0.0)

    def test_nan_returns_nan(self):
        assert np.isnan(epoch_to_hour(np.nan))

    def test_result_in_0_to_24(self):
        for val in range(0, 100, 5):
            result = epoch_to_hour(val)
            assert 0 <= result < 24

# epoch_series_to_hour
class TestEpochSeriesToHour:

    def test_returns_series(self):
        s = pd.Series([0, 8, 32])
        result = epoch_series_to_hour(s)
        assert isinstance(result, pd.Series)

    def test_zero_maps_to_20(self):
        s = pd.Series([0.0])
        result = epoch_series_to_hour(s)
        assert result.iloc[0] == pytest.approx(20.0)

    def test_all_values_in_valid_range(self):
        s = pd.Series(range(0, 200))
        result = epoch_series_to_hour(s)
        assert (result >= 0).all()
        assert (result < 24).all()

# compute_gap_weeks
class TestComputeGapWeeks:

    @pytest.fixture
    def df_with_gaps(self):
        return pd.DataFrame({
            "uid":          ["A", "A", "A"],
            "day":          [20190101, 20190108, 20190122],
            "year_week":    ["2019-W01", "2019-W02", "2019-W04"],
            "date":         pd.to_datetime(["2019-01-01", "2019-01-08", "2019-01-22"]),
            "has_response": [True, False, True],
        })

    def test_returns_dataframe(self, df_with_gaps):
        result = compute_gap_weeks(df_with_gaps)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, df_with_gaps):
        result = compute_gap_weeks(df_with_gaps)
        for col in ["uid", "total_weeks_in_span", "observed_weeks",
                    "gap_weeks", "pct_gap"]:
            assert col in result.columns

    def test_raises_without_has_response(self):
        df = pd.DataFrame({
            "uid":       ["A", "A"],
            "year_week": ["2019-W01", "2019-W02"],
            "date":      pd.to_datetime(["2019-01-01", "2019-01-08"]),
        })
        with pytest.raises(ValueError):
            compute_gap_weeks(df)

    def test_no_gaps_when_consecutive(self):
        df = pd.DataFrame({
            "uid":          ["A", "A"],
            "day":          [20190101, 20190107],
            "year_week":    ["2019-W01", "2019-W02"],
            "date":         pd.to_datetime(["2019-01-01", "2019-01-07"]),
            "has_response": [True, True],
        })
        result = compute_gap_weeks(df)
        assert result[result["uid"] == "A"]["gap_weeks"].iloc[0] == 0

    def test_pct_gap_between_0_and_100(self, df_with_gaps):
        result = compute_gap_weeks(df_with_gaps)
        assert (result["pct_gap"] >= 0).all()
        assert (result["pct_gap"] <= 100).all()

    def test_sorted_descending_by_pct_gap(self, df_with_gaps):
        df2 = pd.DataFrame({
            "uid":          ["B", "B"],
            "day":          [20190101, 20190107],
            "year_week":    ["2019-W01", "2019-W02"],
            "date":         pd.to_datetime(["2019-01-01", "2019-01-07"]),
            "has_response": [True, True],
        })
        df_combined = pd.concat([df_with_gaps, df2], ignore_index=True)
        result = compute_gap_weeks(df_combined)
        pcts = result["pct_gap"].tolist()
        assert pcts == sorted(pcts, reverse=True)

# circular_mean_hours
class TestCircularMeanHours:
    def test_symmetric_around_midnight(self):
        s = pd.Series([23.75, 0.25])
        result = circular_mean_hours(s)
        assert result == pytest.approx(0.0, abs=0.1) or result == pytest.approx(24.0, abs=0.1)

    def test_arithmetic_would_give_wrong_answer(self):
        s = pd.Series([23.75, 0.25])
        arithmetic = s.mean()
        circular = circular_mean_hours(s)
        assert arithmetic == pytest.approx(12.0)       
        assert circular != pytest.approx(12.0, abs=1.0)

    def test_all_same_value(self):
        s = pd.Series([22.0, 22.0, 22.0])
        assert circular_mean_hours(s) == pytest.approx(22.0, abs=0.01)

    def test_midday_values(self):
        s = pd.Series([11.0, 13.0])
        result = circular_mean_hours(s)
        assert result == pytest.approx(12.0, abs=0.1)

    def test_empty_returns_nan(self):
        assert np.isnan(circular_mean_hours(pd.Series([], dtype=float)))

    def test_single_value_returns_itself(self):
        s = pd.Series([21.5])
        assert circular_mean_hours(s) == pytest.approx(21.5, abs=0.01)

# circular_std_hours
class TestCircularStdHours:

    def test_identical_values_give_zero(self):
        s = pd.Series([22.0, 22.0, 22.0])
        assert circular_std_hours(s) == pytest.approx(0.0, abs=0.01)

    def test_spread_is_positive(self):
        s = pd.Series([20.0, 22.0, 0.0, 2.0])
        assert circular_std_hours(s) > 0

    def test_midnight_crossing_gives_low_std(self):
        s = pd.Series([23.5, 23.75, 0.0, 0.25, 0.5])
        std = circular_std_hours(s)
        assert std < 1.0

    def test_arithmetic_std_would_be_wrong(self):
        s = pd.Series([23.5, 23.75, 0.0, 0.25, 0.5])
        arithmetic_std = s.std()
        circular_std = circular_std_hours(s)
        assert arithmetic_std > 10.0   # huge - wrong
        assert circular_std < 1.0      # small - correct

    def test_empty_returns_nan(self):
        assert np.isnan(circular_std_hours(pd.Series([], dtype=float)))

# aggregate_sleep_times
class TestAggregateSleepTimes:

    def test_returns_dict_with_mean_and_std(self):
        s = pd.Series([32.0, 34.0])
        result = aggregate_sleep_times(s)
        assert "mean" in result
        assert "std" in result

    def test_mean_near_midnight_for_midnight_epochs(self):
        s = pd.Series([30.0, 34.0])
        result = aggregate_sleep_times(s)
        mean = result["mean"]
        assert mean < 1.0 or mean > 23.0  

    def test_std_small_for_consistent_sleep(self):
        s = pd.Series([30.0, 31.0, 32.0, 33.0, 34.0])
        result = aggregate_sleep_times(s)
        assert result["std"] < 1.0