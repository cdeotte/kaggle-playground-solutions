#!/usr/bin/env python
"""Level-2 stacker: one L2 logistic regression on the logits of every OOF / test pair found under models/*/preds/.

Pairs:
  models/<model>/preds/<pair>_oof.npy + <pair>_test.npy   (our models, and public pairs dropped into models/public_models/preds/)
  models/<model>/preds/<pair>_oof.csv + <pair>_test.csv   (an `id` column + one prediction column; aligned by id)
Every pair becomes logits: if all its values lie in [0, 1] they are probabilities (clipped to [1e-15, 1 - 1e-15], then logit);
otherwise the pair already holds logits / log-likelihood ratios. Logits are clipped to +-30 and stored as float32.

The stacker uses the same 5 folds as the level-1 models: fit on 4 folds, predict the 5th (OOF) and the test rows; test = mean of
the 5 fold models. cuML (GPU) is used when installed: LogisticRegression(L2, C=1, solver qn) on the raw logits; its solver is not
exactly deterministic, so the whole fit is repeated --repeats times and averaged (it also prints harmless 'L-BFGS line search
failed' warnings: on these highly correlated inputs qn stops at its last valid step). Otherwise scikit-learn (CPU,
standardized logits) is used.
Usage: python stack.py [--models-dir ../models] [--repeats 5] [--C 1.0] [--backend auto|cuml|sklearn] [--out submission.csv]"""
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
from common.data import TARGET, folds, labels, read_test, read_train  # noqa: E402


def read_csv_pred(path, ids):
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != 'id'][0]
    return df.set_index('id').loc[ids, col].to_numpy(np.float64)


def find_pairs(models_dir, train_ids, test_ids):
    pairs = []
    for oof_path in sorted(glob.glob(os.path.join(models_dir, '*', 'preds', '*_oof.*'))):
        ext = oof_path.rsplit('.', 1)[-1]
        if ext not in ('npy', 'csv'):
            continue
        test_path = oof_path[:-len(f'_oof.{ext}')] + f'_test.{ext}'
        model = Path(oof_path).parents[1].name
        name = f'{model}/{Path(oof_path).name[:-len(f"_oof.{ext}")]}'
        if not os.path.exists(test_path):
            print(f'skipping {name}: no test file', flush=True)
            continue
        if ext == 'npy':
            o, t = np.load(oof_path).astype(np.float64).ravel(), np.load(test_path).astype(np.float64).ravel()
        else:
            o, t = read_csv_pred(oof_path, train_ids), read_csv_pred(test_path, test_ids)
        assert len(o) == len(train_ids) and len(t) == len(test_ids), f'{name}: wrong length'
        assert np.isfinite(o).all() and np.isfinite(t).all(), f'{name}: non-finite values'
        pairs.append((name, o, t))
    return pairs


def to_logits(oof, test):
    if oof.min() >= 0 and oof.max() <= 1 and test.min() >= 0 and test.max() <= 1:
        f = lambda p: np.log(np.clip(p, 1e-15, 1 - 1e-15) / (1 - np.clip(p, 1e-15, 1 - 1e-15)))  # noqa: E731
        oof, test = f(oof), f(test)
    return np.clip(oof, -30, 30).astype(np.float32), np.clip(test, -30, 30).astype(np.float32)


def fit_cuml(A, ya, B, T, C):
    from cuml.linear_model import LogisticRegression
    m = LogisticRegression(penalty='l2', C=C, fit_intercept=True, max_iter=2000, tol=1e-7, solver='qn', output_type='numpy',
                           verbose=False)
    m.fit(A, ya)
    return m.predict_proba(B)[:, 1], m.predict_proba(T)[:, 1], np.ravel(m.coef_)


def fit_sklearn(A, ya, B, T, C):
    from sklearn.linear_model import LogisticRegression
    A, B, T = (x.astype(np.float64) for x in (A, B, T))
    mu, sd = A.mean(0), A.std(0) + 1e-12
    m = LogisticRegression(C=C, solver='lbfgs', max_iter=10000, tol=1e-8)
    m.fit((A - mu) / sd, ya)
    return m.predict_proba((B - mu) / sd)[:, 1], m.predict_proba((T - mu) / sd)[:, 1], m.coef_.ravel() / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default=str(HERE.parent / 'models'))
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--C', type=float, default=1.0)
    ap.add_argument('--backend', default='auto', choices=['auto', 'cuml', 'sklearn'])
    ap.add_argument('--out', default=str(HERE / 'submission.csv'))
    a = ap.parse_args()
    backend = a.backend
    if backend == 'auto':
        try:
            import cuml  # noqa: F401
            backend = 'cuml'
        except ImportError:
            backend = 'sklearn'
    fit = fit_cuml if backend == 'cuml' else fit_sklearn
    repeats = a.repeats if backend == 'cuml' else 1

    tr, te = read_train(), read_test()
    y = labels(tr[TARGET])
    f = folds(y)
    pairs = find_pairs(a.models_dir, tr['id'].to_numpy(), te['id'].to_numpy())
    assert pairs, f'no pairs found under {a.models_dir}/*/preds/'
    print(f'{len(pairs)} pairs:')
    Xo, Xt = [], []
    for name, o, t in pairs:
        lo, lt = to_logits(o, t)
        Xo.append(lo)
        Xt.append(lt)
        print(f'  {name:60s} OOF AUC {roc_auc_score(y, lo):.6f}')
    Xo, Xt = np.stack(Xo, 1), np.stack(Xt, 1)

    oof, test, coefs = np.zeros(len(y)), np.zeros(len(Xt)), []
    for r in range(repeats):
        for k in range(5):
            trn, val = f != k, f == k
            pv, pt, c = fit(Xo[trn], y[trn], Xo[val], Xt, a.C)
            oof[val] += pv / repeats
            test += pt / (5 * repeats)
            if r == 0:
                coefs.append(c)
    fold_auc = [roc_auc_score(y[f == k], oof[f == k]) for k in range(5)]
    print(f'\nstack ({backend}, {repeats} fit(s)): CV AUC {roc_auc_score(y, oof):.6f} | folds ' + ' '.join(f'{v:.6f}' for v in fold_auc))
    print('mean coefficient per pair (logit inputs):')
    for (name, _, _), c in sorted(zip(pairs, np.mean(coefs, 0)), key=lambda z: -abs(z[1])):
        print(f'  {c:+.4f}  {name}')
    pd.DataFrame({'id': te['id'], TARGET: test}).to_csv(a.out, index=False)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
