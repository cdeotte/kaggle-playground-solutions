"""Repeated stratified inner holdouts of the outer-training rows (TabM's n_cv internal members)."""
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def repeated_stratified_holdouts(outer_train_index, y, *, n_splits, validation_fraction, random_state):
    """(n_splits, n_valid) validation indices relative to the outer-training rows."""
    outer_train_index = np.asarray(outer_train_index, dtype=np.int64)
    local_y = np.asarray(y)[outer_train_index]
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=float(validation_fraction),
                                      random_state=int(random_state))
    valid_relative = [np.sort(np.asarray(valid, dtype=np.int64))
                      for _, valid in splitter.split(np.zeros(len(outer_train_index), dtype=np.int8), local_y)]
    return np.stack(valid_relative)
