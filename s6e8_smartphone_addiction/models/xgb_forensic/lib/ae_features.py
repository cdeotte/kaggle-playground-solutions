"""The 25 masked-autoencoder 'fill' features read from the reconstruction cache of ../realmlp_repair.

cache_dir('realmlp_repair') / 'reconstruction-production-v1' / 'fold_{k}' holds, per outer fold, a masked neural
autoencoder fitted on predictors only (outer-training rows reconstructed by 3 row-excluded states, validation / test
rows by the outer-training state): train_features.npy (outer-training rows), valid_features.npy, test_features.npy,
train_indices.npy, valid_indices.npy and manifest.json (feature_names, 189 columns).
Fill view (25 columns, in this order):
  aefull__completed__<numeric>             the 9 numerics completed by the autoencoder (observed value kept)
  ae__cat_probability_if_missing__<c>__<l> the 8 categorical level probabilities where the categorical is missing
  aefullgeom__*                            8 screen-geometry values of the completed row (parts sum, slack, slack
                                           share, social / gaming / work share, weekend - daily, original-rule
                                           probability)"""
import json

import numpy as np

from common.data import cache_dir

NUMERIC_FIELDS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                  "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
CATEGORY_LEVELS = {"gender": ("Female", "Male", "Other"), "stress_level": ("High", "Low", "Medium"),
                   "academic_work_impact": ("No", "Yes")}
GEOMETRY_NAMES = ("aefullgeom__screen_parts_sum", "aefullgeom__screen_slack", "aefullgeom__screen_slack_share",
                  "aefullgeom__screen_social_share", "aefullgeom__screen_gaming_share", "aefullgeom__screen_work_share",
                  "aefullgeom__weekend_minus_daily", "aefullgeom__source_rule_probability")
FILL_NAMES = ([f"aefull__completed__{field}" for field in NUMERIC_FIELDS]
              + [f"ae__cat_probability_if_missing__{field}__{level}"
                 for field, levels in CATEGORY_LEVELS.items() for level in levels]
              + list(GEOMETRY_NAMES))


def reconstruction_root(fold):
    return cache_dir("realmlp_repair") / "reconstruction-production-v1" / f"fold_{fold}"


def ae_fill_parts(fold, train_idx, valid_idx):
    """extra_parts hook of lib/strong.build_strong_fold: one block (names, train, valid, test) inserted right after
    the screen-response block, giving the 127-column order the model was selected with."""
    root = reconstruction_root(fold)
    names = list(json.loads((root / "manifest.json").read_text())["feature_names"])
    columns = [names.index(name) for name in FILL_NAMES]
    assert np.array_equal(np.load(root / "train_indices.npy"), train_idx)
    assert np.array_equal(np.load(root / "valid_indices.npy"), valid_idx)
    blocks = [np.asarray(np.load(root / f"{role}_features.npy", mmap_mode="r")[:, columns], np.float32)
              for role in ("train", "valid", "test")]
    return [(list(FILL_NAMES), *blocks)]
