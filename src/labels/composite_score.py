"""
Composite mental health score computation and weekly label engineering.

Composite formula (decided in Phase 1):
  1. Reverse code sse3-1: reversed = 6 - sse3-1
  2. Sum SSE items: sse3-1_r + sse3-2 + sse3-3 + sse3-4  (range 4-20)
  3. Normalize each instrument to [0, 1] using known valid ranges
  4. Invert negative indicators (PHQ4, stress): normalized = 1 - normalized
  5. Average the four normalized values
  6. Multiply by 100 to express as percentage

Instrument ranges and directions (decided in Phase 1):
  phq4_score   : 0-12,  negative 
  stress       : 1-5,   negative 
  social_level : 1-5,   positive 
  sse_score    : 4-20,  positive 
"""

import pandas as pd
import numpy as np

# Instrument configuration 

INSTRUMENT_CONFIG = {
    "phq4_score":   {"min": 0,  "max": 12, "invert": True},
    "stress":       {"min": 1,  "max": 5,  "invert": True},
    "social_level": {"min": 1,  "max": 5,  "invert": False},
    "sse_score":    {"min": 4,  "max": 20, "invert": False},
}

SSE3_ITEMS_SCORED = ["sse3-1_r", "sse3-2", "sse3-3", "sse3-4"]


# Cronbach's alpha 
# Checks whether items measure the same construct
def cronbach_alpha(df_items: pd.DataFrame) -> float:
    df_clean = df_items.dropna()
    n_items = df_clean.shape[1]
    if n_items < 2 or len(df_clean) == 0:
        return float("nan")
    item_variances = df_clean.var(axis=0, ddof=1).sum()
    total_variance = df_clean.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
    return round(float(alpha), 4)


# Normalization
def normalize_instrument(series: pd.Series, min_val: float,
    max_val: float, invert: bool) -> pd.Series:
    normalized = (series - min_val) / (max_val - min_val)
    normalized = normalized.clip(0, 1)
    if invert:
        normalized = 1 - normalized
    return normalized


# SSE preparation 
# Reverse code sse3-1 and compute the SSE sum score
def prepare_sse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sse3-1_r"] = 6 - df["sse3-1"]
    existing = [c for c in SSE3_ITEMS_SCORED if c in df.columns]
    df["sse_score"] = df[existing].sum(axis=1, min_count=1)
    return df


# Composite score 
def compute_composite(df: pd.DataFrame) -> pd.Series:
    norm_cols = []
    df_out = df.copy()

    for col, cfg in INSTRUMENT_CONFIG.items():
        if col not in df_out.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        nc = f"norm_{col}"
        df_out[nc] = normalize_instrument(
            df_out[col], cfg["min"], cfg["max"], cfg["invert"]
        )
        norm_cols.append(nc)

    composite = df_out[norm_cols].mean(axis=1, skipna=False) * 100
    return composite


# Weekly label engineering
# Weeks with multiple surveys are averaged 
def build_weekly_labels(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["has_response"]].copy()

    weekly = (
        completed.groupby(["uid", "year_week"])
        .agg(
            composite_score=("composite_score", "mean"),
            n_surveys_in_week=("composite_score", "count"),
        )
        .reset_index()
    )
    weekly["composite_score"] = weekly["composite_score"].round(4)
    return weekly

# The label decided on this week's data predicts the next week's composite score
def apply_next_week_shift(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["week_dt"] = pd.to_datetime(
        weekly["year_week"] + "-1", format="%G-W%V-%u"
    )
    weekly["next_week_dt"] = weekly["week_dt"] + pd.Timedelta(weeks=1)
    weekly["next_year_week"] = (
        weekly["next_week_dt"].dt.isocalendar().year.astype(str)
        + "-W"
        + weekly["next_week_dt"].dt.isocalendar().week.astype(str).str.zfill(2)
    )

    # Match each week W with the same student's week W+1 label
    label_lookup = weekly[["uid", "year_week", "composite_score"]].rename(
        columns={
            "year_week":       "next_year_week",
            "composite_score": "label_composite_score",
        }
    )

    shifted = weekly.merge(label_lookup, on=["uid", "next_year_week"], how="inner")

    return (
        shifted[["uid", "year_week", "next_year_week",
                 "label_composite_score", "n_surveys_in_week"]]
        .rename(columns={
            "year_week":      "feature_week",
            "next_year_week": "label_week",
        })
        .reset_index(drop=True)
    )
