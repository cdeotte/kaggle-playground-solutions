"""Shared fold loop of the public-notebook models.

Every public model is trained on our 5 folds (common.data.folds: StratifiedKFold(5, shuffle=True, random_state=42)
on the target). A model script supplies fit_fold(fold, train_index, valid_index) -> (valid_pred, test_pred), where the
indices are train.csv row positions and the predictions are probabilities for the validation rows and for all test
rows.

run_folds: each fold is independent (the model is seeded per fold); finished folds are kept in <model>/parts/ and
    skipped, so folds can run one at a time or in parallel processes. Once all 5 are done the pair is written:
    <model>/preds/<pair>_oof.npy (held-out predictions) and <pair>_test.npy (mean of the 5 fold models).
run_all_folds: the 5 folds in order in one process, for notebooks that seed once at the start and let the random state
    run on from fold to fold (as in our run); writes the pair directly."""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PUBLIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_DIR.parents[1]))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402


def model_dir(name):
    """models/public_models/<name>/"""
    return PUBLIC_DIR / name


def work_dir(name):
    """models/public_models/<name>/cache/: working directory for libraries that write logs or checkpoints to the current
    directory."""
    return cache_dir(f'public_models/{name}')


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def fold_args():
    """--folds 0,1,2,3,4 (default: all five)."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4', help='comma-separated folds to train; finished folds are skipped')
    return fold_list(ap.parse_args().folds)


def _split(f, k):
    return np.flatnonzero(f != k), np.flatnonzero(f == k)


def run_folds(name, fit_fold, fold_ids=None, y=None):
    y = target() if y is None else np.asarray(y)
    f = folds(y)
    for k in (range(5) if fold_ids is None else fold_ids):
        if have_part(model_dir(name), name, k):
            print(f'{name} fold {k}: already done', flush=True)
            continue
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        valid_pred = np.asarray(valid_pred, np.float64).reshape(-1)
        test_pred = np.asarray(test_pred, np.float64).reshape(-1)
        print(f'{name} fold {k}: AUC {roc_auc_score(y[valid_index], valid_pred):.6f}', flush=True)
        save_part(model_dir(name), name, k, valid_pred, test_pred)
    return assemble(model_dir(name), name, f, y)


def run_all_folds(name, fit_fold, y=None):
    y = target() if y is None else np.asarray(y)
    f = folds(y)
    oof, test = np.zeros(len(y)), 0.0
    for k in range(5):
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        oof[valid_index] = np.asarray(valid_pred, np.float64).reshape(-1)
        test = test + np.asarray(test_pred, np.float64).reshape(-1) / 5
        print(f'{name} fold {k}: AUC {roc_auc_score(y[valid_index], oof[valid_index]):.6f}', flush=True)
    save_pair(model_dir(name), name, oof, test)
    print(f'{name}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof
