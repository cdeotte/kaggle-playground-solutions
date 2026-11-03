"""Per-fold feature cache of the forensic LightGBM (written by ../build_fold_features.py).

cache_dir('lgbm_margin') / features / fold_{k} /
  {train,valid,test}_numeric.npy             float32 (rows, 333): strong (104 = the 102 forensic columns of
                                             ../xgb_forensic + strong__selected_base_margin / _probability),
                                             fixed_source (114 fixed-original univariate LLR / log-count channels),
                                             cat_numeric (115 numeric columns of the source-fragment cache)
  {train,valid,test}_categories.npy          int32 (rows, 129): native source-fragment categories (76) and semantic
                                             interaction categories (53), recoded 0..k-1 over the fold's
                                             outer-training values (-1 = unseen)
  {train,valid,test}_category_frequency.npy  float32 (rows, 129): log1p(outer-training count of the code)
  {train,valid,test}_category_summary.npy    float32 (rows, 6): # known / unseen codes, mean / min / max / std of the
                                             known log-frequencies
  train_indices.npy, valid_indices.npy       int64 train.csv rows of the outer-training part and of the fold
  manifest.json                              base_numeric_names, numeric_groups, category_names, category_groups,
                                             category_cardinalities, category_frequency_names, category_summary_names
train = the fold's outer-training rows, valid = the rows of the fold, test = all test rows."""
import json

import numpy as np

from common.data import cache_dir

KINDS = ("numeric", "categories", "category_frequency", "category_summary")


def fold_root(fold):
    return cache_dir("lgbm_margin") / "features" / f"fold_{fold}"


def load_manifest(fold):
    return json.loads((fold_root(fold) / "manifest.json").read_text())


def load(fold, role, kind):
    """role in train / valid / test, kind in KINDS; memory-mapped."""
    return np.load(fold_root(fold) / f"{role}_{kind}.npy", mmap_mode="r")


def contiguous_categories(train, valid, test):
    """Recode each column over its outer-training vocabulary; unseen application values -> -1."""
    outputs = [np.empty(values.shape, dtype=np.int32) for values in (train, valid, test)]
    cardinalities = []
    for column in range(train.shape[1]):
        vocabulary = np.unique(train[:, column])
        cardinalities.append(int(len(vocabulary)))
        for destination, values in zip(outputs, (train, valid, test)):
            positions = np.searchsorted(vocabulary, values[:, column])
            clipped = np.minimum(positions, len(vocabulary) - 1)
            matched = (positions < len(vocabulary)) & (vocabulary[clipped] == values[:, column])
            destination[:, column] = np.where(matched, positions, -1).astype(np.int32)
    return outputs[0], outputs[1], outputs[2], cardinalities


def category_frequency(train, valid, test, cardinalities):
    outputs = [np.empty(values.shape, dtype=np.float32) for values in (train, valid, test)]
    for column, cardinality in enumerate(cardinalities):
        observed = train[:, column] >= 0
        counts = np.bincount(train[observed, column], minlength=cardinality).astype(np.float64)
        for destination, values in zip(outputs, (train, valid, test)):
            code = values[:, column]
            known = code >= 0
            frequency = np.zeros(len(code), dtype=np.float64)
            frequency[known] = counts[code[known]]
            destination[:, column] = np.log1p(frequency).astype(np.float32)
    return outputs[0], outputs[1], outputs[2]


SUMMARY_NAMES = ["category_summary__known_count", "category_summary__unseen_count",
                 "category_summary__mean_logcount", "category_summary__min_logcount",
                 "category_summary__max_logcount", "category_summary__std_logcount"]


def category_summaries(codes, frequency):
    known = codes >= 0
    safe = np.where(known, frequency, np.nan)
    with np.errstate(invalid="ignore"):
        mean, minimum = np.nanmean(safe, axis=1), np.nanmin(safe, axis=1)
        maximum, std = np.nanmax(safe, axis=1), np.nanstd(safe, axis=1)
    return np.column_stack([known.sum(axis=1), (~known).sum(axis=1), np.nan_to_num(mean), np.nan_to_num(minimum),
                            np.nan_to_num(maximum), np.nan_to_num(std)]).astype(np.float32)
