SENSING_FEATURES = [
    "sleep_duration_mean",             "sleep_duration_std",
    "unlock_num_ep_0_mean",            "unlock_num_ep_0_std",
    "unlock_duration_ep_0_mean",       "unlock_duration_ep_0_std",
    "act_still_ep_0_mean",             "act_still_ep_0_std",
    "act_in_vehicle_ep_0_mean",        "act_in_vehicle_ep_0_std",
    "act_on_bike_ep_0_mean",           "act_on_bike_ep_0_std",
    "loc_self_dorm_dur_mean",          "loc_self_dorm_dur_std",
    "loc_social_dur_mean",             "loc_social_dur_std",
    "loc_study_dur_mean",              "loc_study_dur_std",
    "other_playing_duration_ep_0_mean","other_playing_duration_ep_0_std",
    "is_ios",
]

COVID_FEATURES = [
    "COVID-1", "COVID-2", "COVID-3", "COVID-4",
    "COVID-5", "COVID-6", "COVID-7", "COVID-8",
    "COVID-10", "covid_period",
]

ALL_FEATURES = SENSING_FEATURES + COVID_FEATURES

TARGET = "label_composite_score"

META_COLS = ["uid", "year_week", "n_surveys_in_week"]