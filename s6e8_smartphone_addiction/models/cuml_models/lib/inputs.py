"""Matrices made by other folders of this repo (run those first).

../xgb_forensic     cache/source_knn_v1/{train,test}.npy (+ manifest.json feature_names): 17 original-neighbor features
                    (label means of the k = 4 / 16 / 64 nearest original rows, plain and inverse-distance weighted,
                    nearest label, distances to the nearest rows / nearest positive / nearest negative, their contrast
                    and log ratio, entropy, observed field count)
                    cache/source_knn_impute_v1/{train,test}.npy (+ manifest.json feature_names): 15 original-neighbor
                    imputations of the 5 screen fields (imputed value, prediction if missing, uncertainty if missing)
                    cache/observed_joint_cache_v1/{train,test}_features.npy (+ manifest.json columns): 72 columns, log
                    joint count and PMI of the observed values of each of the 36 numeric field pairs
../pairlift_models  cache/pairlift/fold_<k>/{train,valid,test}_features.npy: the 10 pair-lift columns, 5-fold inner
                    cross-fitted within the outer-train rows of fold k (rows: outer-train rows, held-out rows, test
                    rows)."""
import json

import numpy as np

from common.data import cache_dir


def _names(path, key, count, prefix):
    if path.exists():
        names = list(json.loads(path.read_text())[key])
        if len(names) == count:
            return names
    return [f'{prefix}{j:02d}' for j in range(count)]


def source_knn():
    """(train, test, names) of the 17 original-neighbor features."""
    d = cache_dir('xgb_forensic') / 'source_knn_v1'
    train, test = np.load(d / 'train.npy', mmap_mode='r'), np.load(d / 'test.npy', mmap_mode='r')
    return train, test, _names(d / 'manifest.json', 'feature_names', train.shape[1], 'source_knn_')


def source_knn_impute():
    """(train, test, names) of the 15 original-neighbor imputation features."""
    d = cache_dir('xgb_forensic') / 'source_knn_impute_v1'
    train, test = np.load(d / 'train.npy', mmap_mode='r'), np.load(d / 'test.npy', mmap_mode='r')
    return train, test, _names(d / 'manifest.json', 'feature_names', train.shape[1], 'source_knn_impute_')


def observed_joint():
    """(train, test, names) of the 72 observed-joint log-count / PMI columns."""
    d = cache_dir('xgb_forensic') / 'observed_joint_cache_v1'
    train = np.load(d / 'train_features.npy', mmap_mode='r')
    test = np.load(d / 'test_features.npy', mmap_mode='r')
    if train.shape[1] != 72 or test.shape[1] != 72:
        raise RuntimeError('observed-joint matrices must have 72 columns')
    return train, test, _names(d / 'manifest.json', 'columns', 72, 'observed_joint_')


def pairlift(fold, role):
    """The 10 pair-lift columns of fold `fold` for role 'train' (outer-train rows), 'valid' or 'test'."""
    d = cache_dir('pairlift_models') / 'pairlift' / f'fold_{fold}'
    return np.asarray(np.load(d / f'{role}_features.npy'), dtype=np.float32)
