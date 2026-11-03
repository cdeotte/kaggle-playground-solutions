"""Reconstruction / density-completion inputs of the reconstruction CatBoost (train_reconstruction_catboost.py).

Both caches are made by ../realmlp_repair (cache_dir('realmlp_repair')):
  reconstruction-production-v1/fold_{k}   masked autoencoder + XGBoost reconstructions of every predictor, fitted on
                                          predictors only (189 columns; outer-training rows from 3 row-excluded
                                          states): train_features.npy (outer-training rows), valid_features.npy,
                                          test_features.npy, train_indices.npy, valid_indices.npy, manifest.json
                                          (feature_names)
  density_completion_response_production_features/fold_{k}
                                          per-fold density-completion response channels (30 columns; the model uses
                                          the 10 '__aggregate__' ones), same file layout and manifest.json

parent_frames(fold)          the forensic CatBoost matrix of train_forensic_catboost.py (forensic float32 columns +
                             margin columns, 114 fixed-original source channels, native fragment categories, semantic
                             categories) for the outer-training, validation and test rows
reconstruction_categories()  'core' categorical view: missing-pattern codes (all 12 fields, 5 screen fields, count),
                             per numeric the q32 code of the hybrid completion and the q8 code of the XGBoost / AE
                             disagreement where missing (edges from the outer-training rows), per categorical the
                             completed level, the predicted level where missing and the XGBoost / AE agreement"""
import json

import numpy as np
import pandas as pd

from common.data import cache_dir
from models.catboost_forensic.lib.strong_cat import (build_strong_application, cat_semantic_categories,
                                                     fixed_source_features, native_categories, source_test_features)

NUMERIC_FIELDS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                  "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
CATEGORICAL_LEVELS = {"gender": ("Female", "Male", "Other"), "stress_level": ("High", "Low", "Medium"),
                      "academic_work_impact": ("No", "Yes")}
SCREEN_FIELDS = ("daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                 "weekend_screen_time")
# the forensic CatBoost parent (train_forensic_catboost.py)
PARENT_CAT_GROUPS = ["raw_cat", "mask", "exact_value", "semantic_state"]
PARENT_SOURCE_NUMERIC_GROUPS = ["univariate_llr_logcount"]
PARENT_SEMANTIC_GROUPS = ["source_qbins", "source_q16_pairs", "imputed_screen", "source_rule"]


def _load_cache(name, fold, columns=None):
    """(names, train, valid, test) of one realmlp_repair cache; rows = outer-training rows, rows of the fold, test."""
    root = cache_dir("realmlp_repair") / name / f"fold_{fold}"
    names = list(json.loads((root / "manifest.json").read_text())["feature_names"])
    columns = list(range(len(names))) if columns is None else columns(names)
    arrays = [np.asarray(np.load(root / f"{role}_features.npy", mmap_mode="r")[:, columns], np.float32)
              for role in ("train", "valid", "test")]
    return [names[i] for i in columns], *arrays


def load_reconstruction(fold):
    return _load_cache("reconstruction-production-v1", fold)


def load_density(fold, view="compact10"):
    """(names, train, valid, test) of the density-completion channels ('compact10' = the 10 aggregates)."""
    pick = (lambda names: [i for i, name in enumerate(names) if "__aggregate__" in name]) if view == "compact10" else None
    return _load_cache("density_completion_response_production_features", fold, pick)


def reconstruction_view_indices(names, view):
    """Numeric reconstruction views: the first 71 / 123 / 189 cache columns."""
    return {"none": [], "compact71": list(range(71)), "detailed123": list(range(123)),
            "full189": list(range(189))}[view]


def parent_frames(fold, encoder_class=None):
    """Frames (train, valid, test), categorical names, target, train_idx, valid_idx and the 3 base margins."""
    (strong_train, strong_valid, strong_test, target, train_idx, valid_idx,
     margin_train, margin_valid, margin_test) = build_strong_application(fold, encoder_class=encoder_class)
    test_rows = len(target) + np.arange(len(strong_test), dtype=np.int64)
    native = [native_categories(rows, PARENT_CAT_GROUPS) for rows in (train_idx, valid_idx, test_rows)]
    semantic_train, semantic_apply, semantic_names = cat_semantic_categories(
        strong_train, pd.concat([strong_valid, strong_test], ignore_index=True), PARENT_SEMANTIC_GROUPS)
    semantic = [semantic_train, semantic_apply.iloc[:len(valid_idx)].reset_index(drop=True),
                semantic_apply.iloc[len(valid_idx):].reset_index(drop=True)]
    source_train, source_cat_train, source_cat_names = fixed_source_features(train_idx, PARENT_SOURCE_NUMERIC_GROUPS)
    source_valid, source_cat_valid, _ = fixed_source_features(valid_idx, PARENT_SOURCE_NUMERIC_GROUPS)
    source_test, source_cat_test = source_test_features(source_train.columns.tolist(), source_cat_names)
    cat_names = list(native[0][1]) + list(semantic_names) + list(source_cat_names)
    frames = [pd.concat([strong, source, nat[0], source_cat, sem], axis=1)
              for strong, source, nat, source_cat, sem in zip(
                  (strong_train, strong_valid, strong_test), (source_train, source_valid, source_test), native,
                  (source_cat_train, source_cat_valid, source_cat_test), semantic)]
    return frames, cat_names, target, train_idx, valid_idx, (margin_train, margin_valid, margin_test)


def quantile_edges(values, bins):
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    return np.unique(np.quantile(finite, np.arange(1, bins) / bins)).astype(np.float64)


def apply_bins(values, edges):
    values = np.asarray(values, np.float64)
    output = np.zeros(len(values), dtype=np.int32)
    finite = np.isfinite(values)
    output[finite] = np.searchsorted(edges, values[finite], side="right").astype(np.int32) + 1
    return output


def raw_missing_views(frame):
    fields = list(NUMERIC_FIELDS) + list(CATEGORICAL_LEVELS)
    missing = np.column_stack([frame[field].isna().to_numpy() for field in fields])
    all_code = np.sum(missing * (1 << np.arange(len(fields), dtype=np.int64))[None, :], axis=1).astype(np.int32)
    screen = [fields.index(name) for name in SCREEN_FIELDS]
    screen_code = np.sum(missing[:, screen] * (1 << np.arange(len(screen), dtype=np.int64))[None, :],
                         axis=1).astype(np.int32)
    return all_code, screen_code, missing.sum(axis=1).astype(np.int32)


def reconstruction_categories(train_raw, apply_raw, train_recon, apply_recon, names):
    """'core' view for the outer-training rows and a list of application row sets (raw rows + reconstruction rows)."""
    lookup = {name: index for index, name in enumerate(names)}
    train_parts, apply_parts = {}, [{} for _ in apply_raw]

    def add(name, train_values, values):
        train_parts[name] = np.asarray(train_values, np.int32)
        for destination, local in zip(apply_parts, values):
            destination[name] = np.asarray(local, np.int32)

    train_masks = raw_missing_views(train_raw)
    apply_masks = [raw_missing_views(frame) for frame in apply_raw]
    for index, label in enumerate(("all12", "screen5", "count")):
        add(f"reconcat__missing__{label}", train_masks[index], [values[index] for values in apply_masks])
    for field in NUMERIC_FIELDS:
        for name, source_name, bins in ((f"reconcat__hybrid_q32__{field}", f"hybridfull__completed__{field}", 32),
                                        (f"reconcat__disagreement_q8__{field}",
                                         f"hybrid__disagreement_norm_if_missing__{field}", 8)):
            column = lookup[source_name]
            edges = quantile_edges(train_recon[:, column], bins)
            add(name, apply_bins(train_recon[:, column], edges),
                [apply_bins(matrix[:, column], edges) for matrix in apply_recon])
    for field, levels in CATEGORICAL_LEVELS.items():
        level_lookup = {str(level): index + 1 for index, level in enumerate(levels)}
        xgb_columns = [lookup[f"xgb__cat_probability_if_missing__{field}__{level}"] for level in levels]
        ae_columns = [lookup[f"ae__cat_probability_if_missing__{field}__{level}"] for level in levels]

        def bundle(raw, matrix):
            missing = raw[field].isna().to_numpy()
            xgb = np.asarray(matrix[:, xgb_columns], np.float64)
            ae = np.asarray(matrix[:, ae_columns], np.float64)
            xgb_code = np.argmax(xgb, axis=1).astype(np.int32) + 1
            ae_code = np.argmax(ae, axis=1).astype(np.int32) + 1
            hybrid_code = np.argmax(0.5 * (xgb + ae), axis=1).astype(np.int32) + 1
            observed = raw[field].map(level_lookup).fillna(0).to_numpy(np.int32)
            return (np.where(missing, hybrid_code, observed).astype(np.int32),
                    np.where(missing, hybrid_code, 0).astype(np.int32),
                    np.where(missing, 1 + (xgb_code != ae_code), 0).astype(np.int32))

        train_bundle = bundle(train_raw, train_recon)
        apply_bundles = [bundle(raw, matrix) for raw, matrix in zip(apply_raw, apply_recon)]
        for local_index, label in enumerate(("completed", "hybrid_prediction", "agreement")):
            add(f"reconcat__{label}__{field}", train_bundle[local_index],
                [values[local_index] for values in apply_bundles])
    return pd.DataFrame(train_parts), [pd.DataFrame(parts) for parts in apply_parts], list(train_parts)


# Features removed after the frame is assembled (the recipe's pruning list; categorical features are never pruned).
DROP_FEATURE_NAMES = (
    "fixed_source__binary_llr__weekend_screen_time__q16", "strong__imputer_v2_l2_if_missing__work_study_hours",
    "fixed_source__binary_llr__weekend_screen_time__q8", "fixed_source__source_logcount__academic_work_impact__exact",
    "fixed_source__binary_llr__social_media_hours__q8", "fixed_source__source_logcount__work_study_hours__q8",
    "strong__academic_work_impact", "fixed_source__source_logcount__age__q32",
    "fixed_source__binary_llr__academic_work_impact__exact", "fixed_source__binary_llr__daily_screen_time_hours__q16",
    "fixed_source__source_logcount__stress_level__exact", "fixed_source__source_logcount__age__q64",
    "fixed_source__source_logcount__social_media_hours__q8", "fixed_source__binary_llr__work_study_hours__q8",
    "fixed_source__source_logcount__gender__exact", "fixed_source__source_logcount__age__q128",
    "fixed_source__binary_llr__daily_screen_time_hours__q32", "strong__stress_level",
    "fixed_source__source_logcount__daily_screen_time_hours__q8", "strong__gender",
    "fixed_source__binary_llr__daily_screen_time_hours__q64", "fixed_source__binary_llr__social_media_hours__q16",
    "fixed_source__binary_llr__social_media_hours__q32", "fixed_source__binary_llr__gender__exact",
    "strong__imputer_v2_l2_if_missing__weekend_screen_time", "fixed_source__source_logcount__daily_screen_time_hours__q16",
    "fixed_source__binary_llr__age__q64", "fixed_source__source_logcount__social_media_hours__q16",
    "fixed_source__binary_llr__gaming_hours__q8", "fixed_source__source_logcount__gaming_hours__q8",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__sleep_hours",
    "fixed_source__source_logcount__age__q8", "fixed_source__source_logcount__sleep_hours__q8",
    "fixed_source__source_logcount__work_study_hours__q16", "fixed_source__source_logcount__weekend_screen_time__q8",
    "fixed_source__source_logcount__notifications_per_day__q8", "fixed_source__binary_llr__sleep_hours__q16",
    "fixed_source__source_logcount__age__q16", "fixed_source__binary_llr__daily_screen_time_hours__q128",
    "strong__screen_response_logit__strict_slack__q16",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__social_media_hours",
    "fixed_source__binary_llr__sleep_hours__q8",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__work_study_hours",
    "strong__imputer_v2_l2_if_missing__daily_screen_time_hours",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__gaming_hours",
    "fixed_source__binary_llr__weekend_screen_time__q32", "fixed_source__source_logcount__daily_screen_time_hours__q64",
    "fixed_source__source_logcount__age__round2", "fixed_source__binary_llr__age__round2",
    "fixed_source__binary_llr__work_study_hours__q16", "fixed_source__binary_llr__age__q8",
    "fixed_source__source_logcount__social_media_hours__q32", "fixed_source__binary_llr__stress_level__exact",
    "strong__exact_copula_density_ratio18_unsupported__sleep_hours__weekend_screen_time",
    "strong__exact_copula_density_ratio18_unsupported__social_media_hours__notifications_per_day",
    "fixed_source__binary_llr__age__q128",
    "strong__exact_copula_density_ratio18_unsupported__social_media_hours__weekend_screen_time",
    "fixed_source__source_logcount__gaming_hours__q32", "strong__age__frequency",
    "fixed_source__source_logcount__weekend_screen_time__q16", "fixed_source__source_logcount__gaming_hours__q16",
    "fixed_source__binary_llr__daily_screen_time_hours__q8", "fixed_source__binary_llr__weekend_screen_time__q64",
    "fixed_source__source_logcount__social_media_hours__q64", "fixed_source__source_logcount__sleep_hours__q16",
    "fixed_source__binary_llr__notifications_per_day__q8", "strong__gaming_hours",
    "fixed_source__source_logcount__daily_screen_time_hours__q32", "fixed_source__binary_llr__age__q32",
    "strong__imputer_v2_l2_if_missing__gaming_hours", "fixed_source__source_logcount__app_opens_per_day__q16",
    "fixed_source__binary_llr__gaming_hours__q16", "strong__imputer_v2_l2_if_missing__social_media_hours",
    "fixed_source__source_logcount__social_media_hours__round2", "fixed_source__source_logcount__app_opens_per_day__q8",
    "fixed_source__source_logcount__weekend_screen_time__q32", "fixed_source__source_logcount__gaming_hours__q64",
    "fixed_source__binary_llr__app_opens_per_day__q8",
    "strong__exact_copula_density_ratio18_unsupported__gaming_hours__work_study_hours",
    "fixed_source__source_logcount__weekend_screen_time__q64", "strong__age", "fixed_source__binary_llr__age__q16",
    "fixed_source__source_logcount__weekend_screen_time__round2",
    "fixed_source__source_logcount__daily_screen_time_hours__q128",
    "strong__exact_copula_density_ratio18_unsupported__work_study_hours__sleep_hours",
    "fixed_source__binary_llr__social_media_hours__q64", "fixed_source__source_logcount__work_study_hours__q32",
    "strong__exact_copula_density_ratio18_unsupported__gaming_hours__sleep_hours", "strong__work_study_hours",
    "strong__imputed_screen_response_logit__v2_l2__strict_slack__q16", "fixed_source__binary_llr__app_opens_per_day__q16",
    "strong__exact_copula_density_ratio18_unsupported__social_media_hours__work_study_hours",
    "strong__screen_response_logit__strict_slack__q64", "fixed_source__source_logcount__social_media_hours__q128",
    "fixed_source__source_logcount__weekend_screen_time__q128", "fixed_source__binary_llr__weekend_screen_time__round2",
    "strong__imputer_v2_l2_imputed__work_study_hours", "strong__scaled_complete_budget_remaining",
    "fixed_source__binary_llr__sleep_hours__q32",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__notifications_per_day",
    "strong__screen_response_logit__partial_slack__q16", "fixed_source__binary_llr__social_media_hours__q128",
    "fixed_source__binary_llr__notifications_per_day__q16", "fixed_source__source_logcount__gaming_hours__q128",
    "strong__weekend_minus_daily", "fixed_source__source_logcount__app_opens_per_day__q64",
    "fixed_source__source_logcount__sleep_hours__q64", "fixed_source__source_logcount__notifications_per_day__round2",
    "fixed_source__source_logcount__notifications_per_day__q16", "fixed_source__binary_llr__notifications_per_day__q32",
    "strong__exact_copula_density_ratio18_unsupported__notifications_per_day__weekend_screen_time",
    "strong__exact_copula_density_ratio18_unsupported__daily_screen_time_hours__app_opens_per_day",
    "fixed_source__binary_llr__gaming_hours__q64", "fixed_source__binary_llr__notifications_per_day__q64",
    "strong__imputer_v2_l2_imputed__gaming_hours", "strong__imputed_screen_response_logit__v2_l2__strict_slack__q64",
    "strong__screen_response_logit__component_share_daily__q16", "fixed_source__source_logcount__sleep_hours__q32",
    "strong__observed_joint_pmi__pmi_top18_shrink5__daily_screen_time_hours__social_media_hours__count_shrink_5",
    "fixed_source__binary_llr__weekend_screen_time__q128",
)
