"""Reads the per-fold forensic cache of ../lgbm_margin (cache_dir('lgbm_margin') / features / fold_k:
{train,valid,test}_numeric.npy, train_indices.npy, valid_indices.npy, manifest.json with base_numeric_names)."""
import json
import re

import numpy as np
import pandas as pd

from common.data import cache_dir


def fold_root(fold):
    return cache_dir("lgbm_margin") / "features" / f"fold_{fold}"


def load_fold(fold):
    """manifest, outer-training rows, rows of the fold (train.csv indices)."""
    root = fold_root(fold)
    manifest = json.loads((root / "manifest.json").read_text())
    return manifest, np.load(root / "train_indices.npy"), np.load(root / "valid_indices.npy")


def numeric_matrix(fold, role):
    return np.load(fold_root(fold) / f"{role}_numeric.npy", mmap_mode="r")


def resolve_cache_columns(all_names, include_regex):
    compiled = [re.compile(pattern) for pattern in include_regex]
    selected = [i for i, name in enumerate(all_names) if any(p.search(name) is not None for p in compiled)]
    return selected, [all_names[i] for i in selected]


def append_cache_features(frame, matrix, selected_indices, selected_names):
    selected = np.asarray(matrix[:, selected_indices], dtype=np.float32)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(selected, columns=selected_names)], axis=1)
