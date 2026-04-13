"""
Request and response schemas using SENSING_FEATURES from src/features/feature_columns.
"""

from pydantic import BaseModel, Field
from typing import Optional

# One student week of sensing features
class SensingFeatures(BaseModel):
    # Autoregression
    label_lag1: Optional[float] = Field(None, description="Previous week composite score")

    # Sleep
    sleep_duration_mean: Optional[float] = Field(None, ge=0, le=24)
    sleep_duration_std:  Optional[float] = Field(None, ge=0)

    # Phone usage
    unlock_num_ep_0_mean:       Optional[float] = Field(None, ge=0)
    unlock_num_ep_0_std:        Optional[float] = Field(None, ge=0)
    unlock_duration_ep_0_mean:  Optional[float] = Field(None, ge=0)
    unlock_duration_ep_0_std:   Optional[float] = Field(None, ge=0)

    # Activity
    act_still_ep_0_mean:       Optional[float] = Field(None, ge=0)
    act_still_ep_0_std:        Optional[float] = Field(None, ge=0)
    act_in_vehicle_ep_0_mean:  Optional[float] = Field(None, ge=0)
    act_in_vehicle_ep_0_std:   Optional[float] = Field(None, ge=0)
    act_on_bike_ep_0_mean:     Optional[float] = Field(None, ge=0)
    act_on_bike_ep_0_std:      Optional[float] = Field(None, ge=0)

    # Location
    loc_self_dorm_dur_mean: Optional[float] = Field(None, ge=0)
    loc_self_dorm_dur_std:  Optional[float] = Field(None, ge=0)
    loc_social_dur_mean:    Optional[float] = Field(None, ge=0)
    loc_social_dur_std:     Optional[float] = Field(None, ge=0)
    loc_study_dur_mean:     Optional[float] = Field(None, ge=0)
    loc_study_dur_std:      Optional[float] = Field(None, ge=0)

    # Media (iOS only)
    other_playing_duration_ep_0_mean: Optional[float] = Field(None, ge=0)
    other_playing_duration_ep_0_std:  Optional[float] = Field(None, ge=0)

    # Platform
    is_ios: int = Field(..., ge=0, le=1, description="1=iOS, 0=Android")

class PredictionRequest(BaseModel):
    uid:       str  = Field(..., description="Student identifier")
    year_week: str  = Field(..., description="ISO week e.g. 2020-W15")
    features:  SensingFeatures

class PredictionResponse(BaseModel):
    uid:                  str
    year_week:            str
    predicted_score:      float = Field(..., ge=0, le=100)
    model_name:           str
    model_version:        str
    model_alias:          str

class HealthResponse(BaseModel):
    status:        str
    model_name:    str
    model_alias:   str
    mlflow_uri:    str