# Code of the public notebook "XGBoost v2 for Predicting Smartphone Addiction" (version 1), cells 1-5:
# https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction
# Importing this module builds the notebook's label-free features of train and test (train_df, test_df) and the
# list of columns it target / frequency encodes (cat_cols: the 12 raw predictors and 8 decile bins). The notebook's
# own fold encoder (compute_fold_encodings) is not used: train.py encodes with inner out-of-fold encodings.
# Only the data loading was changed (data come from common.data).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import read_test, read_train  # noqa: E402

# notebook cell 1
import gc
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
SEED = 42

def seed_everything(seed=42):
    np.random.seed(seed)

seed_everything(SEED) 

# notebook cell 3
train_df = read_train()
test_df = read_test()

TARGET = "addicted_label"
ID_COL = "id" if "id" in train_df.columns else None

initial_features = [c for c in train_df.columns if c not in [TARGET, ID_COL]]

# notebook cell 5
def engineer_features(df):
    
    df = df.copy()

    df["missing_count"] = df[initial_features].astype(str).isin(["nan", "None", "<NA>", "np.nan"]).sum(axis=1).astype(float)
    
    def to_float(col_name):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name].astype(str), errors="coerce")
        return pd.Series(np.nan, index=df.index)
    
    s_media = to_float("social_media_hours")
    gaming = to_float("gaming_hours")
    work = to_float("work_study_hours")
    daily_screen = to_float("daily_screen_time_hours")
    sleep = to_float("sleep_hours")
    weekend_screen = to_float("weekend_screen_time")
    app_opens = to_float("app_opens_per_day")
    notifications = to_float("notifications_per_day")

    df["total_breakdown_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0) + work.fillna(0.0)
    df["social_ratio"] = s_media / (daily_screen + 1e-5)
    df["gaming_ratio"] = gaming / (daily_screen + 1e-5)
    df["work_ratio"] = work / (daily_screen + 1e-5)
    df["unaccounted_screen_time"] = daily_screen - df["total_breakdown_hours"]
    df["unaccounted_screen_ratio"] = df["unaccounted_screen_time"] / (daily_screen + 1e-5)
    df["screen_to_sleep_ratio"] = daily_screen / (sleep + 1e-5)
    df["weekend_vs_daily_ratio"] = weekend_screen / (daily_screen + 1e-5)
    df["app_opens_per_hour"] = app_opens / (daily_screen + 1e-5)
    df["notifications_per_hour"] = notifications / (daily_screen + 1e-5)
    
    df["leisure_hours"] = s_media.fillna(0.0) + gaming.fillna(0.0)
    df["productivity_ratio"] = work / (df["leisure_hours"] + 1e-5)
    df["entertainment_disruption_index"] = df["leisure_hours"] / (sleep + 1e-5)
    df["screen_intensity_per_unlock"] = daily_screen / (app_opens + 1e-5)
    df["notification_distraction_index"] = notifications / (daily_screen + 1e-5)
    df["opens_per_notification"] = app_opens / (notifications + 1e-5)
    df["sleep_deprivation_index"] = (daily_screen + 1e-5) * (24.0 - sleep.clip(0, 24) + 1e-5)
    df["weekly_screen_estimate"] = (daily_screen * 5.0) + (weekend_screen * 2.0)

    for col_name, col_series in [
        ("daily_screen_time_hours", daily_screen),
        ("sleep_hours", sleep),
        ("work_study_hours", work),
        ("social_media_hours", s_media),
        ("gaming_hours", gaming),
        ("weekend_screen_time", weekend_screen),
    ]:
        norm_val = (col_series.fillna(0.0) % 24.0) * (2 * np.pi / 24.0)
        df[f"{col_name}_sin"] = np.sin(norm_val)
        df[f"{col_name}_cos"] = np.cos(norm_val)

    for ratio_col in ["social_ratio", "gaming_ratio", "work_ratio"]:
        norm_ratio = df[ratio_col].clip(0.0, 1.0) * (2 * np.pi)
        df[f"{ratio_col}_sin"] = np.sin(norm_ratio)
        df[f"{ratio_col}_cos"] = np.cos(norm_ratio)

    df["log_notifications"] = np.log1p(notifications.clip(lower=0).fillna(0.0))
    df["log_app_opens"] = np.log1p(app_opens.clip(lower=0).fillna(0.0))
    df["log_daily_screen"] = np.log1p(daily_screen.clip(lower=0).fillna(0.0))
    df["log_leisure"] = np.log1p(df["leisure_hours"].clip(lower=0))

    engineered_cont_cols = [
        "total_breakdown_hours", "social_ratio", "gaming_ratio", "work_ratio",
        "unaccounted_screen_time", "unaccounted_screen_ratio", "screen_to_sleep_ratio", 
        "weekend_vs_daily_ratio", "app_opens_per_hour", "notifications_per_hour", 
        "productivity_ratio", "entertainment_disruption_index", "screen_intensity_per_unlock",
        "notification_distraction_index", "opens_per_notification", "sleep_deprivation_index",
        "weekly_screen_estimate", "leisure_hours", "log_notifications",
        "log_app_opens", "log_daily_screen", "log_leisure",
        "daily_screen_time_hours_sin", "daily_screen_time_hours_cos",
        "sleep_hours_sin", "sleep_hours_cos",
        "work_study_hours_sin", "work_study_hours_cos",
        "social_media_hours_sin", "social_media_hours_cos",
        "gaming_hours_sin", "gaming_hours_cos",
        "weekend_screen_time_sin", "weekend_screen_time_cos",
        "social_ratio_sin", "social_ratio_cos",
        "gaming_ratio_sin", "gaming_ratio_cos",
        "work_ratio_sin", "work_ratio_cos",
    ]
    
    df[engineered_cont_cols] = df[engineered_cont_cols].fillna(0.0)
    return df

train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

cols_to_bin = [
    "social_media_hours", "gaming_hours", "work_study_hours", 
    "daily_screen_time_hours", "sleep_hours", "app_opens_per_day", 
    "notifications_per_day", "weekend_screen_time"
]

binned_cols = []
for col in cols_to_bin:
    if col in train_df.columns:
        train_vals = pd.to_numeric(train_df[col].astype(str), errors='coerce')
        binned, bins = pd.qcut(train_vals.dropna(), q=10, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        bin_col_name = f"{col}_bin"
        train_df[bin_col_name] = pd.cut(pd.to_numeric(train_df[col].astype(str), errors='coerce'), bins=bins, labels=False).fillna(-1).astype(np.float32)
        test_df[bin_col_name] = pd.cut(pd.to_numeric(test_df[col].astype(str), errors='coerce'), bins=bins, labels=False).fillna(-1).astype(np.float32)
        binned_cols.append(bin_col_name)

        bin_rad_tr = (train_df[bin_col_name].clip(lower=0) / 10.0) * (2 * np.pi)
        bin_rad_te = (test_df[bin_col_name].clip(lower=0) / 10.0) * (2 * np.pi)
        
        train_df[f"{bin_col_name}_sin"] = np.sin(bin_rad_tr).astype(np.float32)
        train_df[f"{bin_col_name}_cos"] = np.cos(bin_rad_tr).astype(np.float32)
        test_df[f"{bin_col_name}_sin"] = np.sin(bin_rad_te).astype(np.float32)
        test_df[f"{bin_col_name}_cos"] = np.cos(bin_rad_te).astype(np.float32)

def compute_group_aggregations(train, test):
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    if "app_opens_per_day_bin" in combined.columns and "daily_screen_time_hours" in combined.columns:
        app_screen_val = pd.to_numeric(combined["daily_screen_time_hours"].astype(str), errors='coerce').fillna(0.0)
        group_mean = combined.groupby("app_opens_per_day_bin")["daily_screen_time_hours"].transform("mean")
        combined["diff_screen_from_app_open_bin_mean"] = app_screen_val - group_mean
        combined["ratio_screen_to_app_open_bin_mean"] = app_screen_val / (group_mean + 1e-5)

    if "daily_screen_time_hours_bin" in combined.columns and "sleep_hours" in combined.columns:
        sleep_val = pd.to_numeric(combined["sleep_hours"].astype(str), errors='coerce').fillna(0.0)
        group_mean = combined.groupby("daily_screen_time_hours_bin")["sleep_hours"].transform("mean")
        combined["diff_sleep_from_screen_bin_mean"] = sleep_val - group_mean
        combined["ratio_sleep_to_screen_bin_mean"] = sleep_val / (group_mean + 1e-5)

    if "app_opens_per_day_bin" in combined.columns and "notifications_per_day" in combined.columns:
        notif_val = pd.to_numeric(combined["notifications_per_day"].astype(str), errors='coerce').fillna(0.0)
        group_mean = combined.groupby("app_opens_per_day_bin")["notifications_per_day"].transform("mean")
        combined["diff_notif_from_app_open_bin_mean"] = notif_val - group_mean

    tr_out = combined.iloc[:len(train)].copy()
    te_out = combined.iloc[len(train):].reset_index(drop=True).copy()
    return tr_out, te_out

train_df, test_df = compute_group_aggregations(train_df, test_df)

cat_cols = [c for c in initial_features if c in train_df.columns] + binned_cols
