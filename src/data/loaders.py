"""
Data loading utilities.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Column groups (general_ema)
PHQ4_ITEMS = ["phq4-1", "phq4-2", "phq4-3", "phq4-4"]
SSE3_ITEMS = ["sse3-1", "sse3-2", "sse3-3", "sse3-4"]
SCORE_COLS = ["phq4_score", "pam", "stress", "social_level"]
TIMING_COLS = [
    "phq4_resp_mean", "phq4_resp_median",
    "sse3_resp_mean", "sse3_resp_median",
    "avg_ema_spent_time",
]
ALL_EMA_COLS = PHQ4_ITEMS + SSE3_ITEMS + SCORE_COLS + TIMING_COLS

# Columns that indicate a completed EMA response
RESPONSE_COLS = PHQ4_ITEMS + SSE3_ITEMS + ["phq4_score", "stress", "social_level"]

# Columns that indicate an issued but empty survey 
TIMING_ONLY_COLS = ["avg_ema_spent_time"]

# COVID EMA item columns
COVID_ITEMS = [f"COVID-{i}" for i in range(1, 11)]

# Students excluded from analysis (identified in Phase 1)
EXCLUDE_UIDS = [
    "df5e798581def8d477316520953b9171",  # 0 surveys - paper exclusion 
    "e6d71fe4a3c10b075ae1cf51a2fe6cfd",  # 0 surveys - paper exclusion 
    "ea716dd032aaa0dcf8bfa36b1811917f",  # 9 surveys, 50 days - early dropout, not in demographics
    "ad15fc229da933fbf1fc0f92fc9b55a3",  # 1 survey, 2 days - withdrawal
]

# Shared utilities

# ISO year is used so that dates near boundaries 
# are assigned to the correct week.
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["day"].astype(str), format="%Y%m%d")
    df["year_week"] = (
        df["date"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    return df

# Three classes of rows: 
# has_response- student filled in EMA questions
# incomplete- timing data present but no response
# no_survey- entirely blank day row
def classify_ema_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() 
    existing_response = [c for c in RESPONSE_COLS if c in df.columns]
    existing_timing   = [c for c in TIMING_ONLY_COLS if c in df.columns]

    df["has_response"] = df[existing_response].notna().any(axis=1)
    df["was_issued"]   = df[existing_timing].notna().any(axis=1)
    df["incomplete"]   = df["was_issued"] & ~df["has_response"]
    df["no_survey"]    = ~df["was_issued"] & ~df["has_response"]
    return df

# General EMA loader
def load_general_ema(path: str,
    exclude_uids: Optional[list] = None,
    classify: bool = True) -> pd.DataFrame:
    if exclude_uids is None:
        exclude_uids = EXCLUDE_UIDS

    df = pd.read_csv(path)
    df = parse_dates(df)

    if exclude_uids:
        df = df[~df["uid"].isin(exclude_uids)].copy()

    if classify:
        df = classify_ema_rows(df)

    return df

# COVID EMA loader
def load_covid_ema(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = parse_dates(df)
    existing_covid = [c for c in COVID_ITEMS if c in df.columns]
    df["has_response"] = df[existing_covid].notna().any(axis=1)
    return df


# Sensing loader
def load_sensing(path: str, clean_uids: Optional[list] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()  # defragment to avoid PerformanceWarning when adding new columns later
    df = parse_dates(df)

    if clean_uids is not None:
        df = df[df["uid"].isin(clean_uids)].copy()

    return df


# EDA utility functions 
# Exploratory Data Analysis - analyzing and summarizing datasets 
# to understand their main characteristics

# Global and per-student date range statistics
def temporal_coverage(df: pd.DataFrame) -> dict: 
    global_min = df["date"].min()
    global_max = df["date"].max()
    total_days = (global_max - global_min).days + 1

    per_student = (
        df.groupby("uid")["date"]
        .agg(start="min", end="max")
        .assign(span_days=lambda x: (x["end"] - x["start"]).dt.days + 1)
    )

    return {
        "global_start":        global_min.date(),
        "global_end":          global_max.date(),
        "total_calendar_days": total_days,
        "total_calendar_weeks": round(total_days / 7, 2),
        "n_students":          df["uid"].nunique(),
        "per_student":         per_student,
    }


# Weekly survey counts per student
def weekly_survey_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "has_response" not in df.columns:
        df = classify_ema_rows(df)

    weekly = (
        df[df["has_response"]]
        .groupby(["uid", "year_week"])
        .size()
        .reset_index(name="surveys_in_week")
    )
    return weekly

# Total survey counts per student
def per_student_survey_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "has_response" not in df.columns:
        df = classify_ema_rows(df)

    counts = (
        df[df["has_response"]]
        .groupby("uid")
        .size()
        .reset_index(name="total_surveys")
        .sort_values("total_surveys", ascending=False)
        .reset_index(drop=True)
    )
    return counts

# Missing value analysis per column
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in ALL_EMA_COLS if c in df.columns]
    report = pd.DataFrame({
        "column":    existing,
        "missing_n": [df[c].isna().sum() for c in existing],
        "missing_%": [round(df[c].isna().mean() * 100, 2) for c in existing],
    })
    return report.sort_values("missing_%", ascending=False).reset_index(drop=True)

# Missing value analysis per student
def per_student_missing(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in ALL_EMA_COLS if c in df.columns]
    result = []
    for uid, grp in df.groupby("uid"):
        total       = len(grp)
        any_data    = grp[existing].notna().any(axis=1).sum()
        all_missing = total - any_data
        result.append({
            "uid":            uid,
            "total_rows":     total,
            "rows_with_data": any_data,
            "fully_missing":  all_missing,
            "pct_missing":    round(all_missing / total * 100, 2),
        })
    return (
        pd.DataFrame(result)
        .sort_values("pct_missing", ascending=False)
        .reset_index(drop=True)
    )
