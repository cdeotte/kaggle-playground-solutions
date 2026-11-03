"""CPU replacement for cuML's exact-key target encoder (same constructor and fit_transform / transform interface).

The parent pipeline calls it once per numeric key with explicit five-way inner fold ids. Every training row is encoded
with the totals of the other four inner folds: (sum + smooth * prior) / (count + smooth), prior = mean of all training
rows; application rows use the map fitted on all training rows (unseen keys -> prior)."""
import numpy as np
import pandas as pd


class CPUExactTargetEncoder:
    def __init__(self, *, n_folds, smooth, seed, split_method, output_type):
        if n_folds != 5 or split_method != "customize" or output_type != "numpy":
            raise ValueError("unexpected exact target-encoder contract")
        self.n_folds = int(n_folds)
        self.smooth = float(smooth)
        self.seed = int(seed)
        self.keys = self.mapping = self.prior = None

    @staticmethod
    def _values(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(np.float64)

    def fit_transform(self, series, target, *, fold_ids):
        values = self._values(series)
        y = np.asarray(target, dtype=np.float64)
        folds = np.asarray(fold_ids, dtype=np.int32)
        keys, inverse = np.unique(values, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
        sums = np.bincount(inverse, weights=y, minlength=len(keys)).astype(np.float64)
        prior = float(y.mean())
        encoded = np.empty(len(values), dtype=np.float64)
        for fold in range(self.n_folds):
            held = folds == fold
            fold_count = np.bincount(inverse[held], minlength=len(keys)).astype(np.float64)
            fold_sum = np.bincount(inverse[held], weights=y[held], minlength=len(keys)).astype(np.float64)
            encoded[held] = ((sums - fold_sum)[inverse[held]] + self.smooth * prior) / (
                (counts - fold_count)[inverse[held]] + self.smooth)
        self.keys = keys
        self.mapping = (sums + self.smooth * prior) / (counts + self.smooth)
        self.prior = prior
        return encoded.astype(np.float32)

    def transform(self, series):
        values = self._values(series)
        positions = np.searchsorted(self.keys, values)
        clipped = np.minimum(positions, len(self.keys) - 1)
        matched = (positions < len(self.keys)) & (self.keys[clipped] == values)
        output = np.full(len(values), self.prior, dtype=np.float64)
        output[matched] = self.mapping[positions[matched]]
        return output.astype(np.float32)
