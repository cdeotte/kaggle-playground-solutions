"""Column lists, helpers and loaders of the base feature blocks.

Cache layout (cache_dir('forensic_features')):
  source_channel_train.npy, source_channel_test.npy   float32 (rows, 583), train.csv / test.csv order
  source_channel_manifest.json                        feature_names, groups (univariate, latent_univariate, pair,
                                                      geometry, aggregate, missing, compact), shapes
  screen_response/fold_{k}_train.npy                  outer-training rows of fold k (in train.csv order)
  screen_response/fold_{k}_valid.npy                  rows of fold k
  screen_response/fold_{k}_test.npy                   all test rows
  screen_response/fold_{k}_{train,valid}_indices.npy  the train.csv row indices of the two row sets
  screen_response/fold_{k}_manifest.json              feature_names
"""
import json
from pathlib import Path

import numpy as np

from common.data import cache_dir

NUMERIC = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours",
           "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
PREDICTORS = NUMERIC + CATEGORICAL
SCREEN = ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "weekend_screen_time"]


def cache():
    return cache_dir("forensic_features")


def safe_ratio(numerator, denominator):
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def source_channel_manifest():
    return json.loads((cache() / "source_channel_manifest.json").read_text())


def source_channel(role):
    """role 'train' or 'test': memory-mapped float32 matrix (all rows) and the feature names."""
    names = list(source_channel_manifest()["feature_names"])
    return np.load(cache() / f"source_channel_{role}.npy", mmap_mode="r"), names


def screen_response(fold):
    """(train, valid, test, names) of fold `fold`; train = outer-training rows, valid = rows of the fold."""
    prefix = cache() / "screen_response" / f"fold_{fold}"
    names = list(json.loads(Path(f"{prefix}_manifest.json").read_text())["feature_names"])
    return (np.load(f"{prefix}_train.npy", mmap_mode="r"), np.load(f"{prefix}_valid.npy", mmap_mode="r"),
            np.load(f"{prefix}_test.npy", mmap_mode="r"), names)
