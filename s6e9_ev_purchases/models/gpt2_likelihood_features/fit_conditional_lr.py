#!/usr/bin/env python
"""Pair gpt2_conditional_llr: a nested L2 logistic regression (cuML) on the 306 features of build_conditional_features.py.

Per outer fold: 12 candidates (feature family forward / inverse / combined / full x C in 0.01, 0.1, 1) are scored by inner CV over
the other 4 folds (features standardized with the inner-train moments and clipped to +-12); the candidate with the best inner OOF
AUC on the outer-train rows is refit on all of them and predicts the held-out fold and the test rows.
OOF = held-out probabilities; test = mean of the 5 fold models; both clipped to [1e-7, 1 - 1e-7].
Usage: python fit_conditional_lr.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.gpt_rows import assemble_pair  # noqa: E402
from common.pairs import fold_list, have_part, save_part  # noqa: E402

NAME = 'gpt2_conditional_llr'
CANDIDATES = [(family, C) for family in ['forward', 'inverse', 'combined', 'full'] for C in [0.01, 0.1, 1.0]]


def moments(x, rows):
    mu = x[rows].mean(axis=0, dtype=np.float64)
    sd = x[rows].std(axis=0, dtype=np.float64)
    return mu, np.where(sd > 1e-7, sd, 1.0)


def transform(x, rows, mu, sd):
    return np.ascontiguousarray(np.clip((x[rows].astype(np.float64) - mu) / sd, -12.0, 12.0), dtype=np.float32)


def fit_predict(x_fit, y_fit, predicts, columns, C):
    from cuml.linear_model import LogisticRegression
    use = np.asarray(columns, np.int32)
    model = LogisticRegression(penalty='l2', C=C, fit_intercept=True, class_weight=None, solver='qn', tol=1e-7, max_iter=500,
                               output_type='numpy', verbose=False)
    model.fit(np.ascontiguousarray(x_fit[:, use]), y_fit.astype(np.int32))
    return [np.asarray(model.predict_proba(np.ascontiguousarray(v[:, use])))[:, 1].astype(np.float64) for v in predicts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    n_train = len(y)
    x_all = np.load(HERE / 'features' / 'conditional_features.npy', mmap_mode='r')
    families = json.loads((HERE / 'features' / 'conditional_feature_families.json').read_text())
    x = x_all[:n_train]
    test_rows = np.arange(n_train, len(x_all))
    for outer in fold_list(a.folds):
        if have_part(HERE, NAME, outer):
            continue
        outer_train = f != outer
        train_rows, valid_rows = np.flatnonzero(outer_train), np.flatnonzero(f == outer)
        local = np.full(n_train, -1, np.int64)
        local[train_rows] = np.arange(len(train_rows))
        inner_pred = {c: np.full(len(train_rows), np.nan) for c in CANDIDATES}
        for inner in [v for v in range(5) if v != outer]:
            fit_rows, inner_valid = np.flatnonzero(outer_train & (f != inner)), np.flatnonzero(f == inner)
            mu, sd = moments(x, fit_rows)
            x_fit, x_valid = transform(x, fit_rows, mu, sd), transform(x, inner_valid, mu, sd)
            for family, C in CANDIDATES:
                inner_pred[family, C][local[inner_valid]] = fit_predict(x_fit, y[fit_rows], [x_valid], families[family], C)[0]
            del x_fit, x_valid
        auc = {c: roc_auc_score(y[train_rows], p) for c, p in inner_pred.items()}
        family, C = max(CANDIDATES, key=lambda c: auc[c])
        print(f'fold {outer}: selected {family} C={C} (inner OOF AUC {auc[family, C]:.6f})', flush=True)
        mu, sd = moments(x, train_rows)
        valid_pred, test_pred = fit_predict(transform(x, train_rows, mu, sd), y[train_rows],
                                            [transform(x, valid_rows, mu, sd), transform(x_all, test_rows, mu, sd)], families[family], C)
        save_part(HERE, NAME, outer, valid_pred, test_pred)
        print(f'fold {outer}: AUC {roc_auc_score(y[valid_rows], valid_pred):.6f}', flush=True)
    assemble_pair(HERE, NAME, f, y, clip=1e-7)


if __name__ == '__main__':
    main()
