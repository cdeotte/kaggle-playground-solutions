#!/usr/bin/env python
"""Level-2 stacker: one L2 logistic regression on the logits of every OOF / test pair found under models/*/preds/ and
models/public_models/**/preds/ (pairs: <pair>_oof.npy (n_train) + <pair>_test.npy (n_test), probabilities).

1. Fold-wise calibration of each OOF (test predictions are left as they are): within every validation fold the OOF
   probabilities are ranked onto a shared integer grid 0..(smallest fold size - 1), the rows with the same rank are pooled
   across the 5 folds, and every OOF probability is replaced by the mean probability of its pooled rank. This puts the 5
   fold models of a level-1 model on one common scale.
2. Inputs: logit(p), p clipped to [1e-15, 1 - 1e-15], logits clipped to +-30.
3. Model: cuML LogisticRegression (L2, C = 0.01, solver qn, tol 1e-4, max_iter 10,000); scikit-learn when cuML is missing.
   CV: the same 5 folds as the level-1 models. Test prediction: the mean of 10 fits on all train rows (the GPU solver is not
   exactly deterministic).
Usage: python stack.py [--models-dir ../models] [--C 0.01] [--full-fits 10] [--out submission.csv]"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.data import ID, TARGET, folds, labels, read_test, read_train  # noqa: E402

EPS, LOGIT_CLIP = 1e-15, 30.0


def oof_files(models_dir):
    found = glob.glob(os.path.join(models_dir, '*', 'preds', '*_oof.npy'))
    found += glob.glob(os.path.join(models_dir, 'public_models', '**', 'preds', '*_oof.npy'), recursive=True)
    return sorted(set(found))


def common_fold_ranks(values, common_size):
    """Integer ranks 0..common_size-1 within one fold (stable sort, positions mapped onto the shared grid)."""
    n = len(values)
    order = np.argsort(values, kind='mergesort')
    ranks_sorted = np.clip((np.arange(n, dtype=np.int64) * common_size) // n, 0, common_size - 1).astype(np.int32)
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = ranks_sorted
    return ranks


def fold_calibrate(oof, fold_ids):
    sizes = [int((fold_ids == k).sum()) for k in range(5)]
    common = min(sizes)
    ranks = np.empty(len(oof), dtype=np.int32)
    for k in range(5):
        idx = np.where(fold_ids == k)[0]
        ranks[idx] = common_fold_ranks(oof[idx], common)
    means = pd.DataFrame({'rank': ranks, 'p': oof}).groupby('rank', sort=True)['p'].mean()
    return np.clip(means.iloc[ranks].to_numpy(np.float64), EPS, 1 - EPS)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.clip(np.log(p / (1 - p)), -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)


def make_model(C, backend):
    if backend == 'cuml':
        from cuml.linear_model import LogisticRegression
        return LogisticRegression(penalty='l2', C=C, fit_intercept=True, max_iter=10_000, tol=1e-4, solver='qn',
                                  output_type='numpy', verbose=False)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=10_000, tol=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default=str(HERE.parent / 'models'))
    ap.add_argument('--C', type=float, default=0.01)
    ap.add_argument('--full-fits', type=int, default=10)
    ap.add_argument('--backend', default='auto', choices=['auto', 'cuml', 'sklearn'])
    ap.add_argument('--out', default=str(HERE / 'submission.csv'))
    a = ap.parse_args()
    backend = a.backend
    if backend == 'auto':
        try:
            import cupy
            cupy.zeros(1)  # initialize the GPU before the large arrays are loaded
            backend = 'cuml'
        except Exception:
            backend = 'sklearn'

    train, test = read_train(), read_test()
    y = labels(train[TARGET]).astype(np.int32)
    f = folds(y)
    names, X, X_test = [], [], []
    for oof_path in oof_files(a.models_dir):
        test_path = oof_path[:-len('_oof.npy')] + '_test.npy'
        name = f'{os.path.relpath(Path(oof_path).parents[1], a.models_dir)}/{Path(oof_path).name[:-len("_oof.npy")]}'
        if not os.path.exists(test_path):
            print(f'skipping {name}: no test file', flush=True)
            continue
        o = np.load(oof_path).astype(np.float64).ravel()[:len(y)]
        t = np.load(test_path).astype(np.float64).ravel()
        assert len(o) == len(y) and len(t) == len(test), f'{name}: wrong length'
        names.append(name)
        X.append(logit(fold_calibrate(np.clip(o, EPS, 1 - EPS), f)))
        X_test.append(logit(t))
        print(f'  {name:70s} OOF AUC {roc_auc_score(y, o):.6f}')
    X, X_test = np.stack(X, 1), np.stack(X_test, 1)
    print(f'{len(names)} pairs')

    oof = np.zeros(len(y))
    for k in range(5):
        tr, va = f != k, f == k
        m = make_model(a.C, backend)
        m.fit(X[tr], y[tr])
        oof[va] = np.asarray(m.predict_proba(X[va]))[:, 1]
        print(f'fold {k}: AUC {roc_auc_score(y[va], oof[va]):.6f}', flush=True)
    print(f'stack ({backend}): CV AUC {roc_auc_score(y, oof):.6f}')

    pred = np.zeros(len(test))
    coef = 0.0
    for _ in range(a.full_fits):
        m = make_model(a.C, backend)
        m.fit(X, y)
        pred += np.asarray(m.predict_proba(X_test))[:, 1] / a.full_fits
        coef = coef + np.ravel(np.asarray(m.coef_)) / a.full_fits
    print('largest coefficients:')
    for c, n in sorted(zip(coef, names), key=lambda z: -abs(z[0]))[:15]:
        print(f'  {c:+.4f}  {n}')
    pd.DataFrame({ID: test[ID], TARGET: pred}).to_csv(a.out, index=False)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
