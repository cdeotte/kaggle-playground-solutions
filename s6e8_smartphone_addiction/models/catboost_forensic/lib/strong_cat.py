"""CatBoost views of the forensic evidence.

build_strong_application(fold)  the 102-column forensic matrix of ../xgb_forensic (lib/strong.py) for the
                                outer-training, validation and test rows as float32 'strong__' columns (categoricals as
                                their codes, NaN kept), plus strong__selected_base_margin / _probability and the margins
native_categories(rows, groups)  int32 columns of cache cat_source_fragments_v1 (build_source_fragments.py); rows index
                                train rows 0..n_train-1 and test rows n_train..; recurrence_pmi18_c2 / source_pair_pmi18
                                select the PMI_PAIRS18 pairs
fixed_source_features(rows, groups, view)
                                columns of ../forensic_features source_channel_train.npy ('fixed_source__' prefix)
source_test_features(names, cat_names)
                                the same columns for the test rows
cat_semantic_categories(strong_train, strong_apply, groups)
                                interaction categories: original q16 / q64 codes of each numeric (source_qbins), q16 x
                                q16 codes of the PMI_PAIRS18 pairs (source_q16_pairs), original q16 codes of the l2-imputed
                                screen fields and their pairs (imputed_screen), the original rule region of the raw and
                                the imputed daily / social values (source_rule)"""
import json

import numpy as np
import pandas as pd

from common.data import cache_dir, read_original
from models.forensic_features.lib.blocks import source_channel, source_channel_manifest
from models.xgb_forensic.lib.pair_evidence import PMI_PAIRS18
from models.xgb_forensic.lib.strong import MARGIN_VARIANT, base_margin, build_strong_fold

NUMERIC_COLUMNS = ("age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                   "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time")
SCREEN_COLUMNS = ("daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours",
                  "weekend_screen_time")


def fragment_root():
    return cache_dir("catboost_forensic") / "cat_source_fragments_v1"


def fragment_manifest():
    return json.loads((fragment_root() / "manifest.json").read_text())


def numeric_copy(frame):
    output = {}
    for name in frame.columns:
        series = frame[name]
        if isinstance(series.dtype, pd.CategoricalDtype):
            values = series.cat.codes.to_numpy(np.float32)
            values[values < 0] = np.nan
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(np.float32)
        output[f"strong__{name}"] = values
    return pd.DataFrame(output)


def build_strong_application(fold, encoder_class=None, margin_variant=MARGIN_VARIANT, extra_parts=None):
    """strong_train, strong_valid, strong_test (float32 'strong__' frames incl. the margin columns), target (all train
    rows), train_idx, valid_idx, margin_train, margin_valid, margin_test."""
    s = build_strong_fold(fold, encoder_class=encoder_class, extra_parts=extra_parts)
    prior = s.prior
    frames, margins = [], []
    for frame in (s.train, s.valid, s.test):
        margin = base_margin(frame, prior, margin_variant)
        strong = numeric_copy(frame)
        strong["strong__selected_base_margin"] = margin.astype(np.float32)
        strong["strong__selected_base_probability"] = (1.0 / (1.0 + np.exp(-margin))).astype(np.float32)
        frames.append(strong)
        margins.append(margin)
    return (frames[0], frames[1], frames[2], s.target, s.train_idx, s.valid_idx, margins[0], margins[1], margins[2])


def native_categories(row_indices, groups):
    manifest = fragment_manifest()
    names = list(manifest["categorical_names"])
    lookup = {name: index for index, name in enumerate(names)}
    selected_names = []
    for group in groups:
        if group == "recurrence_pmi18_c2":
            local = [f"recurrence_c2__{left}__{right}" for left, right in PMI_PAIRS18]
        elif group == "source_pair_pmi18":
            local = [f"source_pair_row__{left}__{right}" for left, right in PMI_PAIRS18]
        else:
            local = list(manifest["groups"][group])
        for name in local:
            if name in lookup and name not in selected_names:
                selected_names.append(name)
    cache = np.load(fragment_root() / "categorical.npy", mmap_mode="r")
    values = np.asarray(cache[row_indices][:, [lookup[name] for name in selected_names]], dtype=np.int32)
    frame = pd.DataFrame(values, columns=[f"native__{name}" for name in selected_names])
    return frame, frame.columns.tolist()


def _fixed_source_indices(names, groups, numeric_groups):
    selected = []
    for group in numeric_groups:
        if group == "univariate_llr_logcount":
            local = [int(i) for i in groups["univariate"] if names[int(i)].startswith(("binary_llr__", "source_logcount__"))]
        elif group == "univariate_llr":
            local = [int(i) for i in groups["univariate"] if names[int(i)].startswith("binary_llr__")]
        else:
            local = [int(i) for i in groups[group]]
        selected += [i for i in local if i not in selected]
    return selected


def _fixed_source_categorical(names, view):
    if view == "none":
        return []
    if view == "llr_q16_q64":
        return [i for i, n in enumerate(names) if n.startswith("binary_llr__") and n.endswith(("__q16", "__q64"))]
    if view == "llr_all":
        return [i for i, n in enumerate(names) if n.startswith("binary_llr__")]
    if view == "latent_q16":
        return [i for i, n in enumerate(names) if n.startswith("latent_logrel__")]
    raise ValueError(view)


def fixed_source_features(row_indices, numeric_groups, categorical_view="none"):
    """Fixed-original source channels of train rows: numeric frame, categorical frame (float32 bits read as int32),
    categorical names."""
    manifest = source_channel_manifest()
    names = list(manifest["feature_names"])
    numeric_idx = _fixed_source_indices(names, manifest["groups"], numeric_groups)
    categorical_idx = _fixed_source_categorical(names, categorical_view)
    matrix, _ = source_channel("train")
    numeric = np.asarray(matrix[np.ix_(row_indices, np.asarray(numeric_idx, np.int64))], dtype=np.float32)
    categorical = np.asarray(matrix[np.ix_(row_indices, np.asarray(categorical_idx, np.int64))], dtype=np.float32)
    categorical_names = [f"fixed_source_cat__{names[i]}" for i in categorical_idx]
    return (pd.DataFrame(numeric, columns=[f"fixed_source__{names[i]}" for i in numeric_idx]),
            pd.DataFrame(np.ascontiguousarray(categorical).view(np.int32), columns=categorical_names),
            categorical_names)


def source_test_features(expected_numeric, expected_categorical):
    """The test rows of the fixed-source columns named `expected_numeric` ('fixed_source__' or 'cat_cache_numeric__'
    prefix) and `expected_categorical` ('fixed_source_cat__')."""
    matrix, names = source_channel("test")
    lookup = {name: index for index, name in enumerate(names)}
    cache_manifest = fragment_manifest()
    cache_lookup = {name: index for index, name in enumerate(cache_manifest["numeric_names"])}
    cache_matrix = np.load(fragment_root() / "numeric.npy", mmap_mode="r")
    test_rows = np.arange(int(cache_manifest["train_rows"]), int(cache_manifest["rows"]), dtype=np.int64)
    numeric_parts = []
    for name in expected_numeric:
        if name.startswith("fixed_source__"):
            numeric_parts.append(np.asarray(matrix[:, lookup[name.removeprefix("fixed_source__")]], np.float32))
        else:
            numeric_parts.append(np.asarray(
                cache_matrix[test_rows, cache_lookup[name.removeprefix("cat_cache_numeric__")]], np.float32))
    categorical_parts = [
        np.ascontiguousarray(np.asarray(matrix[:, lookup[name.removeprefix("fixed_source_cat__")]], np.float32)).view(np.int32)
        for name in expected_categorical]
    numeric = np.column_stack(numeric_parts) if numeric_parts else np.empty((len(matrix), 0), np.float32)
    categorical = np.column_stack(categorical_parts) if categorical_parts else np.empty((len(matrix), 0), np.int32)
    return (pd.DataFrame(numeric.astype(np.float32), columns=list(expected_numeric)),
            pd.DataFrame(categorical.astype(np.int32), columns=list(expected_categorical)))


def _source_bins(values, edges):
    output = np.zeros(len(values), dtype=np.int32)
    observed = np.isfinite(values)
    output[observed] = (np.searchsorted(edges[1:-1], values[observed], side="right") + 1).astype(np.int32)
    return output


def _rule(daily, social):
    output = np.ones(len(daily), dtype=np.int32)
    output[(daily <= 6.0) & (social <= 4.0)] = 2
    output[(daily > 8.0) | (social > 4.0)] = 3
    output[~np.isfinite(daily) | ~np.isfinite(social)] = 0
    return output


def cat_semantic_categories(strong_train, strong_apply, groups):
    """Interaction categories for the outer-training rows and the application rows (edges from the original only)."""
    source = read_original()[list(NUMERIC_COLUMNS)]
    train_parts, apply_parts = {}, {}

    def add(name, train_values, apply_values):
        train_parts[name] = np.asarray(train_values, np.int32)
        apply_parts[name] = np.asarray(apply_values, np.int32)

    q16_train, q16_apply = {}, {}
    for column in NUMERIC_COLUMNS:
        train_raw = strong_train[f"strong__{column}"].to_numpy(np.float64)
        apply_raw = strong_apply[f"strong__{column}"].to_numpy(np.float64)
        source_values = source[column].to_numpy(np.float64)
        q16_edges = np.unique(np.quantile(source_values, np.linspace(0.0, 1.0, 17)))
        q16_train[column] = _source_bins(train_raw, q16_edges)
        q16_apply[column] = _source_bins(apply_raw, q16_edges)
        if "source_qbins" in groups:
            add(f"semantic__source_q16__{column}", q16_train[column], q16_apply[column])
            q64_edges = np.unique(np.quantile(source_values, np.linspace(0.0, 1.0, 65)))
            add(f"semantic__source_q64__{column}", _source_bins(train_raw, q64_edges), _source_bins(apply_raw, q64_edges))
    if "source_q16_pairs" in groups:
        for left, right in PMI_PAIRS18:
            add(f"semantic__source_q16_pair__{left}__{right}", q16_train[left] * 32 + q16_train[right],
                q16_apply[left] * 32 + q16_apply[right])
    if "imputed_screen" in groups:
        imputed_train, imputed_apply = {}, {}
        for column in SCREEN_COLUMNS:
            feature = f"strong__imputer_v2_l2_imputed__{column}"
            edges = np.unique(np.quantile(source[column].to_numpy(np.float64), np.linspace(0.0, 1.0, 17)))
            imputed_train[column] = np.searchsorted(edges[1:-1], strong_train[feature].to_numpy(np.float64),
                                                    side="right").astype(np.int32) + 1
            imputed_apply[column] = np.searchsorted(edges[1:-1], strong_apply[feature].to_numpy(np.float64),
                                                    side="right").astype(np.int32) + 1
            add(f"semantic__imputed_source_q16__{column}", imputed_train[column], imputed_apply[column])
        for left_index, left in enumerate(SCREEN_COLUMNS):
            for right in SCREEN_COLUMNS[left_index + 1:]:
                add(f"semantic__imputed_q16_pair__{left}__{right}", imputed_train[left] * 32 + imputed_train[right],
                    imputed_apply[left] * 32 + imputed_apply[right])
    if "source_rule" in groups:
        add("semantic__fixed_source_rule",
            _rule(strong_train["strong__daily_screen_time_hours"].to_numpy(np.float64),
                  strong_train["strong__social_media_hours"].to_numpy(np.float64)),
            _rule(strong_apply["strong__daily_screen_time_hours"].to_numpy(np.float64),
                  strong_apply["strong__social_media_hours"].to_numpy(np.float64)))
        daily, social = ("strong__imputer_v2_l2_imputed__daily_screen_time_hours",
                         "strong__imputer_v2_l2_imputed__social_media_hours")
        add("semantic__imputed_source_rule",
            _rule(strong_train[daily].to_numpy(np.float64), strong_train[social].to_numpy(np.float64)),
            _rule(strong_apply[daily].to_numpy(np.float64), strong_apply[social].to_numpy(np.float64)))
    train_frame, apply_frame = pd.DataFrame(train_parts), pd.DataFrame(apply_parts)
    return train_frame, apply_frame, train_frame.columns.tolist()
