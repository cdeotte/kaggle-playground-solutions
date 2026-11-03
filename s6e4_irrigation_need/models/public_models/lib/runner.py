"""Shared fold loops of the public-notebook models.

Every public model is trained on our 5 folds (common.data.folds: StratifiedKFold(5, shuffle=True, random_state=42) on
the target). The pair of a model is named after its folder and written to <folder>/preds/: <folder>_oof.npy and
<folder>_test.npy, probabilities in class order Low, Medium, High (train.csv / test.csv row order).

run_folds: a model script supplies fit_fold(fold, train_index, valid_index) -> (valid_pred, test_pred), where the
    indices are train.csv row positions (sorted) and the predictions are (rows, 3) probabilities for the validation rows
    and for all test rows. Each fold is independent (the notebook seeds its models per fold); finished folds are kept
    in <folder>/parts/ and skipped, so folds can run one at a time or in parallel processes. Once all 5 are done the
    pair is written: OOF (n_train, 3) from the held-out predictions, test = mean of the 5 fold models.
run_all_folds: the same, but the 5 folds run in order in one process, for notebooks that seed once at the start and
    let the random state run on from fold to fold; writes the pair directly.
run_nested_folds: nested 5 x 5 cross-validation. fit_outer_fold(fold, train_index, valid_index) -> (full_pred,
    test_pred): full_pred (n_train, 3) holds the inner out-of-fold predictions for the outer fold's training rows and
    the mean of the 5 inner models for its held-out rows; test_pred is the mean of the 5 inner models on test. The pair's
    OOF is the (5, n_train, 3) stack of the outer folds' full_pred (the stacker averages the slices), test = mean over
    the 5 outer folds (the mean of all 25 inner models). Finished outer folds are kept in <folder>/parts/."""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

PUBLIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_DIR.parents[1]))
from common.data import cache_dir, folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_pair, save_part  # noqa: E402


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


def fold_args(parser=None):
    """Parses --folds 0,1,2,3,4 (default: all five) plus the arguments already added to `parser`; returns the namespace
    with .folds as a list of ints."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument('--folds', default='0,1,2,3,4',
                        help='comma-separated folds to train; finished folds are skipped')
    args = parser.parse_args()
    args.folds = fold_list(args.folds)
    return args


def bac(y, prob):
    return balanced_accuracy_score(y, np.asarray(prob).argmax(1))


def _split(f, k):
    return np.flatnonzero(f != k), np.flatnonzero(f == k)


def run_folds(folder, fit_fold, fold_ids=None, y=None):
    y = target() if y is None else np.asarray(y)
    f, mdir = folds(y), model_dir(folder)
    for k in (range(5) if fold_ids is None else fold_ids):
        if have_part(mdir, folder, k):
            print(f'{folder} fold {k}: already done', flush=True)
            continue
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        valid_pred, test_pred = np.asarray(valid_pred, np.float32), np.asarray(test_pred, np.float32)
        print(f'{folder} fold {k}: balanced accuracy {bac(y[valid_index], valid_pred):.6f}', flush=True)
        save_part(mdir, folder, k, valid_pred, test_pred)
    return assemble(mdir, folder, f, y)


def run_all_folds(folder, fit_fold, y=None):
    y = target() if y is None else np.asarray(y)
    f = folds(y)
    oof, test = np.zeros((len(y), 3)), 0.0
    for k in range(5):
        train_index, valid_index = _split(f, k)
        valid_pred, test_pred = fit_fold(k, train_index, valid_index)
        oof[valid_index] = np.asarray(valid_pred, np.float64)
        test = test + np.asarray(test_pred, np.float64) / 5
        print(f'{folder} fold {k}: balanced accuracy {bac(y[valid_index], oof[valid_index]):.6f}', flush=True)
    save_pair(model_dir(folder), folder, oof, test)
    print(f'{folder}: OOF balanced accuracy {bac(y, oof):.6f}', flush=True)
    return oof


def _nested_part(mdir, folder, k, split):
    return Path(mdir) / 'parts' / f'{folder}_outer{k}_{split}.npy'


def run_nested_folds(folder, fit_outer_fold, fold_ids=None, y=None):
    y = target() if y is None else np.asarray(y)
    f, mdir = folds(y), model_dir(folder)
    for k in (range(5) if fold_ids is None else fold_ids):
        if all(_nested_part(mdir, folder, k, s).exists() for s in ('full', 'test')):
            print(f'{folder} outer fold {k}: already done', flush=True)
            continue
        train_index, valid_index = _split(f, k)
        full_pred, test_pred = fit_outer_fold(k, train_index, valid_index)
        full_pred, test_pred = np.asarray(full_pred, np.float32), np.asarray(test_pred, np.float32)
        print(f'{folder} outer fold {k}: balanced accuracy held-out rows '
              f'{bac(y[valid_index], full_pred[valid_index]):.6f}, all rows {bac(y, full_pred):.6f}', flush=True)
        p = _nested_part(mdir, folder, k, 'full')
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, full_pred)
        np.save(_nested_part(mdir, folder, k, 'test'), test_pred)
    if not all(_nested_part(mdir, folder, k, s).exists() for k in range(5) for s in ('full', 'test')):
        print(f'{folder}: not all outer folds are done yet', flush=True)
        return None
    oof = np.stack([np.load(_nested_part(mdir, folder, k, 'full')) for k in range(5)])
    test = np.mean([np.load(_nested_part(mdir, folder, k, 'test')).astype(np.float64) for k in range(5)], axis=0)
    save_pair(mdir, folder, oof, test)
    print(f'{folder}: OOF {oof.shape}, balanced accuracy of the mean of the 5 slices {bac(y, oof.mean(0)):.6f}',
          flush=True)
    return oof
