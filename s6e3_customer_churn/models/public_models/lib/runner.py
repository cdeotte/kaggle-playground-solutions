"""Shared fold loops of the public-notebook models.

Every public model is trained on our 5 folds (common.data.folds: StratifiedKFold(5, shuffle=True, random_state=42) on
the target). kfold(y, n_folds) gives the fold id of every train row for StratifiedKFold(n_folds, shuffle=True,
random_state=42); split(fold_ids, k) gives the (sorted) train.csv row positions of fold k's training and validation
rows.

run_folds: for notebooks whose folds are independent (every model is seeded on its own). A model script supplies
    fit_fold(fold, train_index, valid_index, n_folds) -> (valid_pred, test_pred): probabilities for the validation rows
    and for all test rows. Finished folds are kept in <folder>/parts/ and skipped, so a stopped run continues where it
    stopped.
run_all_folds: for notebooks that seed once at the start and let the random state run on from fold to fold. A model
    script supplies cross_validate(n_folds) -> (oof, test), which seeds, trains the folds in order in one process and
    returns the held-out predictions and the mean of the fold models' test predictions.
Both write the pair <folder>/preds/<pair>_oof.npy and <pair>_test.npy (probabilities, train.csv / test.csv row
order)."""
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PUBLIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_DIR.parents[1]))
from common.data import FOLD_SEED, N_FOLDS, cache_dir, folds, target  # noqa: E402
from common.pairs import have_part, save_pair, save_part  # noqa: E402


def model_dir(folder):
    """models/public_models/<folder>/"""
    return PUBLIC_DIR / folder


def work_dir(folder):
    """models/public_models/<folder>/cache/: the working directory of libraries that write logs (catboost_info, ...)."""
    return cache_dir(f'public_models/{folder}')


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def kfold(y, n_folds=N_FOLDS):
    """Fold id of every train row: StratifiedKFold(n_folds, shuffle=True, random_state=42) on the target."""
    if n_folds == N_FOLDS:
        return folds(y)
    f = np.full(len(y), -1, np.int16)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=FOLD_SEED)
    for k, (_, valid_index) in enumerate(skf.split(np.zeros(len(y)), y)):
        f[valid_index] = k
    return f


def split(fold_ids, k):
    """(train_index, valid_index) of fold k: sorted train.csv row positions."""
    return np.flatnonzero(fold_ids != k), np.flatnonzero(fold_ids == k)


def _part_file(mdir, name, fold, kind):
    return Path(mdir) / 'parts' / f'{name}_f{fold}_{kind}.npy'


def _fold_loop(folder, name, fit_fold, y, n_folds):
    mdir, f = model_dir(folder), kfold(y, n_folds)
    for k in range(n_folds):
        if have_part(mdir, name, k):
            print(f'{name} fold {k}: already done', flush=True)
            continue
        train_index, valid_index = split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index, n_folds)
        valid_pred = np.asarray(valid_pred, np.float64).reshape(-1)
        test_pred = np.asarray(test_pred, np.float64).reshape(-1)
        print(f'{name} fold {k}: AUC {roc_auc_score(y[valid_index], valid_pred):.6f}', flush=True)
        save_part(mdir, name, k, valid_pred, test_pred)
    oof, test = np.zeros(len(y)), 0.0
    for k in range(n_folds):
        oof[f == k] = np.load(_part_file(mdir, name, k, 'valid'))
        test = test + np.load(_part_file(mdir, name, k, 'test')).astype(np.float64) / n_folds
    return oof, test


def write_pair(folder, pair, oof, test, y):
    save_pair(model_dir(folder), pair, oof, test)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof


def run_folds(folder, pair, fit_fold, test_folds=N_FOLDS, y=None):
    y = target() if y is None else np.asarray(y)
    oof, test = _fold_loop(folder, pair, fit_fold, y, N_FOLDS)
    if test_folds != N_FOLDS:
        _, test = _fold_loop(folder, f'{pair}_{test_folds}folds', fit_fold, y, test_folds)
    return write_pair(folder, pair, oof, test, y)


def run_all_folds(folder, pair, cross_validate, test_folds=N_FOLDS, y=None):
    y = target() if y is None else np.asarray(y)
    oof, test = cross_validate(N_FOLDS)
    if test_folds != N_FOLDS:
        _, test = cross_validate(test_folds)
    return write_pair(folder, pair, oof, test, y)
