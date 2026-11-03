"""Inputs of the imputed TabM: the masked-autoencoder / XGBoost reconstruction cache and the density-completion
response cache of ../realmlp_repair, the completed predictors and the dual row-local view.

cache_dir('realmlp_repair') / reconstruction-production-v1 / fold_k / {train,valid,test}_features.npy (189 columns,
  outer-training rows reconstructed 3-way row-excluded, the fold's rows and test rows by the outer-training state),
  train_indices.npy, valid_indices.npy, manifest.json (feature_names)
cache_dir('realmlp_repair') / density_completion_response_production_features / fold_k /
  {train,valid,test}_features.npy (30 density-completion response columns; same layout and manifest.json)"""
import json

import numpy as np
import pandas as pd

from common.data import cache_dir, folds

from .row_features import build_rowwise_features

NUMERIC_FIELDS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                  "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
CATEGORICAL_LEVELS = {"gender": ("Female", "Male", "Other"), "stress_level": ("High", "Low", "Medium"),
                      "academic_work_impact": ("No", "Yes")}
RECONSTRUCTION_CACHE = "reconstruction-production-v1"
COMPLETION_CACHE = "density_completion_response_production_features"


def _load_fold_cache(cache_name, fold, n_train, selected=None):
    """(names, all train rows matrix, test matrix): the outer-training and fold rows scattered back to train.csv
    order."""
    root = cache_dir("realmlp_repair") / cache_name / f"fold_{fold}"
    names = list(json.loads((root / "manifest.json").read_text())["feature_names"])
    selected = list(range(len(names))) if selected is None else selected
    train_indices, valid_indices = np.load(root / "train_indices.npy"), np.load(root / "valid_indices.npy")
    full = np.empty((n_train, len(selected)), dtype=np.float32)
    full[train_indices] = np.asarray(np.load(root / "train_features.npy", mmap_mode="r")[:, selected], np.float32)
    full[valid_indices] = np.asarray(np.load(root / "valid_features.npy", mmap_mode="r")[:, selected], np.float32)
    test = np.asarray(np.load(root / "test_features.npy", mmap_mode="r")[:, selected], np.float32)
    return [names[i] for i in selected], full, test


def load_reconstruction(fold, n_train):
    return _load_fold_cache(RECONSTRUCTION_CACHE, fold, n_train)


def load_density(fold, n_train):
    """'full30': all 30 density-completion response columns (names, all train rows matrix, test matrix)."""
    return _load_fold_cache(COMPLETION_CACHE, fold, n_train)


def completed_predictors(frame, reconstruction, names, fill_source="hybrid"):
    """Missing numerics <- '<source>full__completed__<field>'; missing categoricals <- argmax of the mean of the
    autoencoder and XGBoost class probabilities ('hybrid')."""
    index = {name: column for column, name in enumerate(names)}
    completed = frame.copy()
    for field in NUMERIC_FIELDS:
        values = np.asarray(reconstruction[:, index[f"{fill_source}full__completed__{field}"]], dtype=np.float64)
        missing = pd.to_numeric(completed[field], errors="coerce").isna().to_numpy()
        completed.loc[missing, field] = values[missing]
    for field, levels in CATEGORICAL_LEVELS.items():
        missing = completed[field].isna().to_numpy()
        if not missing.any():
            continue
        ae = np.column_stack([reconstruction[:, index[f"ae__cat_probability_if_missing__{field}__{level}"]]
                              for level in levels])
        if fill_source == "ae":
            probabilities = ae
        else:
            xgb = np.column_stack([reconstruction[:, index[f"xgb__cat_probability_if_missing__{field}__{level}"]]
                                   for level in levels])
            probabilities = 0.5 * (ae + xgb)
        labels = np.asarray(levels, dtype=object)[np.argmax(probabilities, axis=1)]
        completed.loc[missing, field] = labels[missing]
    return completed


def missing_flags(frame):
    fields = list(NUMERIC_FIELDS) + list(CATEGORICAL_LEVELS)
    return pd.DataFrame({f"nnmissing__{field}": frame[field].isna().to_numpy(dtype=np.float32) for field in fields})


def dual_rowwise_features(frame, reconstruction, names, blocks, completion_blocks, fill_source="hybrid"):
    """The row-local blocks of the raw rows, the completion blocks of the completed rows ('nncomplete__' prefix) and
    12 missing flags ('nnmissing__')."""
    output, categories = build_rowwise_features(frame, blocks)
    completed = completed_predictors(frame, reconstruction, names, fill_source)
    completed_output, completed_categories = build_rowwise_features(completed, completion_blocks)
    output = pd.concat([output.reset_index(drop=True),
                        completed_output.add_prefix("nncomplete__").reset_index(drop=True)], axis=1)
    categories = list(categories) + [f"nncomplete__{name}" for name in completed_categories]
    return pd.concat([output, missing_flags(frame.reset_index(drop=True))], axis=1), categories
