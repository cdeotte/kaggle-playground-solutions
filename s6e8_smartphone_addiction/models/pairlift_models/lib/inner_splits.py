"""Repeated stratified inner holdouts for the TabM learner (pytabkit's val_idxs).

Inside outer fold k: StratifiedShuffleSplit(n_splits, test_size=validation_fraction, random_state=20260824 + k) over
the outer-training labels. Each of the n_splits TabM members early-stops on its own holdout (indices relative to the
outer-training rows); the outer-validation rows are never used."""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

INNER_SPLIT_BASE_SEED = 20260824


def repeated_stratified_holdouts(outer_train_target, n_splits, validation_fraction, random_state):
    """(n_splits, n_holdout) array of sorted holdout positions relative to the outer-training rows."""
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=float(validation_fraction),
                                      random_state=int(random_state))
    rows = np.asarray(outer_train_target)
    holdouts = [np.sort(np.asarray(valid, dtype=np.int64))
                for _, valid in splitter.split(np.zeros(len(rows), dtype=np.int8), rows)]
    return np.stack(holdouts)
