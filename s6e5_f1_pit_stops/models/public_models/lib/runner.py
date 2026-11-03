"""Shared fold loop of the public-notebook models.

Every public model is trained on our 5 folds (common.data.folds: StratifiedKFold(5, shuffle=True, random_state=42) on
the target). A model script supplies fit_fold(fold, train_index, valid_index) -> (valid_pred, test_pred), where the
indices are train.csv row positions (sorted) and the predictions are probabilities for the validation rows and for all
test rows. Notebooks that append the original dataset take its fold-matched subset from orig_train_index(fold).

run_folds: each fold is independent (the notebook seeds its model, or the global random state, per fold); finished
    folds are kept in <folder>/parts/ and skipped, so folds can run one at a time or in parallel processes. Once all 5
    are done the pair is written: <folder>/preds/<pair>_oof.npy (held-out predictions) and <pair>_test.npy (mean of the
    5 fold models).
run_all_folds: the 5 folds in order in one process, for notebooks that seed once at the start and let the random state
    run on from fold to fold; writes the pair directly.
clip=(low, high): the final OOF / test predictions are clipped, as our run of the notebook did."""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PUBLIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_DIR.parents[1]))
from common.data import cache_dir, folds, orig_folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402


def model_dir(folder):
    """models/public_models/<folder>/"""
    return PUBLIC_DIR / folder


def work_dir(folder):
    """models/public_models/<folder>/cache/: checkpoints, and the working directory of libraries that write logs."""
    return cache_dir(f'public_models/{folder}')


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def fold_args(parser=None):
    """Parses --folds 0,1,2,3,4 (default: all five) plus the arguments already added to `parser`; returns the namespace
    with .folds as a list of ints."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument('--folds', default='0,1,2,3,4',
                        help='comma-separated folds to train; finished folds are skipped')
    args = parser.parse_args()
    args.folds = fold_list(args.folds)
    return args


def orig_train_index(fold, y_orig=None):
    """Positions of the original rows whose own fold (the same StratifiedKFold on the original target) is not `fold`:
    the fold-matched 4/5 of the original dataset that a notebook appends to the training rows of outer fold `fold`."""
    return np.flatnonzero(orig_folds(y_orig) != fold)


def _split(f, k):
    return np.flatnonzero(f != k), np.flatnonzero(f == k)


def run_folds(folder, pair, fit_fold, fold_ids=None, y=None, clip=None):
    y = target() if y is None else np.asarray(y)
    f, mdir = folds(y), model_dir(folder)
    for k in (range(5) if fold_ids is None else fold_ids):
        if have_part(mdir, pair, k):
            print(f'{pair} fold {k}: already done', flush=True)
            continue
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        valid_pred = np.asarray(valid_pred, np.float64).reshape(-1)
        test_pred = np.asarray(test_pred, np.float64).reshape(-1)
        print(f'{pair} fold {k}: AUC {roc_auc_score(y[valid_index], valid_pred):.6f}', flush=True)
        save_part(mdir, pair, k, valid_pred, test_pred)
    oof = assemble(mdir, pair, f)
    if oof is None:
        return None
    if clip is not None:
        test = np.load(mdir / 'preds' / f'{pair}_test.npy')
        oof = np.clip(oof, *clip)
        save_pair(mdir, pair, oof, np.clip(test, *clip))
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof


def run_all_folds(folder, pair, fit_fold, y=None, clip=None):
    y = target() if y is None else np.asarray(y)
    f = folds(y)
    oof, test = np.zeros(len(y)), 0.0
    for k in range(5):
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        oof[valid_index] = np.asarray(valid_pred, np.float64).reshape(-1)
        test = test + np.asarray(test_pred, np.float64).reshape(-1) / 5
        print(f'{pair} fold {k}: AUC {roc_auc_score(y[valid_index], oof[valid_index]):.6f}', flush=True)
    if clip is not None:
        oof, test = np.clip(oof, *clip), np.clip(test, *clip)
    save_pair(model_dir(folder), pair, oof, test)
    print(f'{pair}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof
