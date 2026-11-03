"""Fold loop shared by the training scripts of this folder."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common.data import FOLD_SEED, N_FOLDS, folds


def fold_splits(y, n_folds=N_FOLDS, seed=FOLD_SEED):
    """(train_idx, valid_idx) of every fold: the shared folds of common.data.folds for 5 folds with seed 42, otherwise
    StratifiedKFold(n_folds, shuffle=True, random_state=seed) on the target."""
    y = np.asarray(y)
    if n_folds == N_FOLDS and seed == FOLD_SEED:
        fold_id = folds(y)
        return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(N_FOLDS)]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def run_folds(fit_fold, y, n_test, n_folds=N_FOLDS, seed=FOLD_SEED, name=''):
    """fit_fold(fold, train_idx, valid_idx, seed) -> (valid_pred, test_pred), with fold counted from 1.
    Returns the out-of-fold predictions and the mean of the fold models' test predictions."""
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(n_test, dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(fold_splits(y, n_folds, seed), start=1):
        valid_pred, test_pred = fit_fold(fold, tr_idx, va_idx, seed)
        oof[va_idx] = valid_pred
        test += np.asarray(test_pred, dtype=np.float64) / n_folds
        print(f'{name} fold {fold}/{n_folds}: AUC {roc_auc_score(y[va_idx], valid_pred):.6f}', flush=True)
    return oof, test
