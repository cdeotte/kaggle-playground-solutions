#!/usr/bin/env python
"""Nested class-conditional density-ratio experts (the density columns of the copula XGB C).

For a label pool `fit` and each class c: an XGB separates the real class-c rows of the pool (raw one-hot predictors)
from a copy in which every raw field (all its one-hot columns together) is permuted independently across those rows,
i.e. the class-conditional joint density vs the product of its field marginals. Params: depth 4, eta 0.04, 2 parallel
trees, min_child_weight 50, lambda 20, alpha 1, subsample 0.8, colsample_bytree 0.9, max_bin 512, seed 42; permutation
seed 42 + 101 c. Both class experts score every apply row (margins after 150, 350 and 700 rounds); the apply rows'
labels are never read.
Pools per outer fold k: pool 0 = (outer-train rows -> fold k + test rows), pools 1-5 = inner folds of the outer-train rows.
Writes cache/contrast/copula/densities/raw/fold_<k>/pool_<j>/features.npy, columns CHANNELS.
Needs lib.py of ../income_mixture_xgb.
Usage: python copula_density.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.append(str(HERE.parent / 'income_mixture_xgb'))
from contrast_data import CONTRAST, ints  # noqa: E402
from lib import RAW, Data, splits  # noqa: E402

CHECKPOINTS = [150, 350, 700]
CHANNELS = [f'raw__class{cls}__r{rounds}' for cls in (0, 1) for rounds in CHECKPOINTS]
PARAMS = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=4, eta=.04, num_parallel_tree=2,
              min_child_weight=50, reg_lambda=20, reg_alpha=1, subsample=.8, colsample_bytree=.9, max_bin=512, seed=42)


def design(d):
    """Raw one-hot predictors of all rows and the column groups of each raw field."""
    names = list(d.raw_names)
    groups = [[j for j, n in enumerate(names) if n == key or n.startswith(key + '_')] for key in RAW]
    assert sorted(j for g in groups for j in g) == list(range(len(names)))
    return d.raw.copy().astype(np.float32), groups


def contrast(d, fit, X, groups, cls):
    owner = fit[d.y[fit] == cls]
    real = X[owner]
    fake = np.empty_like(real)
    rng = np.random.default_rng(42 + 101 * cls)
    for g in groups:
        perm = rng.permutation(len(owner))
        fake[:, g] = real[perm][:, g]
    return np.concatenate([real, fake]), np.r_[np.ones(len(real), np.float32), np.zeros(len(fake), np.float32)]


def run_pool(d, X, groups, k, j, fit, apply):
    import xgboost as xgb
    out = CONTRAST / 'copula' / 'densities' / 'raw' / f'fold_{k}' / f'pool_{j}'
    if (out / 'features.npy').exists():
        return
    start = time.time()
    dq = xgb.DMatrix(X[apply])
    predictions = []
    for cls in (0, 1):
        TX, TY = contrast(d, fit, X, groups, cls)
        dt = xgb.DMatrix(TX, label=TY)
        model, previous = None, 0
        for rounds in CHECKPOINTS:
            model = xgb.train(PARAMS, dt, num_boost_round=rounds - previous, xgb_model=model)
            predictions.append(model.predict(dq, output_margin=True).astype(np.float32))
            previous = rounds
        del dt, TX, TY, model
    P = np.column_stack(predictions)
    assert np.isfinite(P).all()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'features.npy', P)
    print(f'density experts fold {k} pool {j}: {P.shape} ({time.time() - start:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    d = Data()
    X, groups = design(d)
    for k in ints(args.folds):
        a, b = np.flatnonzero(d.folds != k), np.flatnonzero(d.folds == k)
        for j, (fit, apply) in enumerate([(a, np.r_[b, d.test])] + splits(a, d.y)):
            assert not np.intersect1d(fit, apply).size
            run_pool(d, X, groups, k, j, fit, apply)


if __name__ == '__main__':
    main()
