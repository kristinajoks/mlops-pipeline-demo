"""
Time and date utilities.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ISO week utilities 
# Format: YYYY-Www (e.g. 2019-W34)
def to_iso_year_week(date_series: pd.Series) -> pd.Series:
    iso = date_series.dt.isocalendar()
    return iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)

# Next week computation for given ISO week label
def next_iso_week(year_week: str) -> str:
    dt = pd.to_datetime(year_week + "-1", format="%G-W%V-%u")
    next_dt = dt + pd.Timedelta(weeks=1)
    iso = next_dt.isocalendar()
    return f"{iso[0]}-W{str(iso[1]).zfill(2)}"


# Gap week analysis 
# total weeks, weeks with and without responses, percentage gap 
def compute_gap_weeks(df: pd.DataFrame) -> pd.DataFrame:
    if "has_response" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'has_response' column. "
            "Call classify_ema_rows() first."
        )

    completed = df[df["has_response"]].copy()
    observed_weeks = completed.groupby("uid")["year_week"].apply(set)
    date_range = completed.groupby("uid")["date"].agg(start="min", end="max")

    results = []
    for uid, row in date_range.iterrows():
        all_dates = pd.date_range(row["start"], row["end"], freq="W-MON")
        all_weeks = set(
            str(d.isocalendar()[0]) + "-W" + str(d.isocalendar()[1]).zfill(2)
            for d in all_dates
        )
        observed = observed_weeks.get(uid, set())
        gap_count = len(all_weeks - observed)
        total = len(all_weeks)
        results.append({
            "uid":                uid,
            "total_weeks_in_span": total,
            "observed_weeks":     len(observed),
            "gap_weeks":          gap_count,
            "pct_gap":            round(gap_count / max(total, 1) * 100, 2),
        })

    return (
        pd.DataFrame(results)
        .sort_values("pct_gap", ascending=False)
        .reset_index(drop=True)
    )


# Sleep encoding 
# The epoch column encodes sleep time as a fraction of the 7.5 hours between 8pm and 3:30am
def epoch_to_hour(epoch_val: float) -> float:
    if pd.isna(epoch_val):
        return np.nan
    minutes_after_8pm = epoch_val * 7.5
    hour_of_day = (20 + minutes_after_8pm / 60) % 24
    return round(hour_of_day, 3)

# Vectorized version for Series input
def epoch_series_to_hour(series: pd.Series) -> pd.Series:
    minutes_after_8pm = series * 7.5
    return (20 + minutes_after_8pm / 60) % 24

# Circular statistics for sleep time
# Circular mean of clock hours on 0-24 scale
def circular_mean_hours(hours: pd.Series) -> float:
    hours_clean = hours.dropna()
    if len(hours_clean) == 0:
        return np.nan

    angles = hours_clean.values * 2 * np.pi / 24
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    mean_angle = np.arctan2(mean_sin, mean_cos)
    mean_hour = mean_angle * 24 / (2 * np.pi)
    return float(mean_hour % 24)
 
# Circular standard deviation of clock hours on 0-24 scale
def circular_std_hours(hours: pd.Series) -> float:
    hours_clean = hours.dropna()
    if len(hours_clean) == 0:
        return np.nan
    angles = hours_clean.values * 2 * np.pi / 24
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    if R >= 1.0:
        return 0.0
    circular_std_radians = np.sqrt(-2 * np.log(R))
    return float(circular_std_radians * 24 / (2 * np.pi))
 
# Main function to aggregate sleep times for a Series of epoch values
def aggregate_sleep_times(series: pd.Series) -> dict:
    hours = epoch_series_to_hour(series)
    return {
        "mean": circular_mean_hours(hours),
        "std":  circular_std_hours(hours),
    }