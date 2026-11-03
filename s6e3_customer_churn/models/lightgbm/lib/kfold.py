"""Fold ids of StratifiedKFold(n_splits, shuffle=True, random_state=seed); n_splits=5, seed=42 gives common.data.folds."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_fold_ids(y, n_splits, seed=42):
    f = np.full(len(y), -1, np.int16)
    for k, (_, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        f[va] = k
    return f
