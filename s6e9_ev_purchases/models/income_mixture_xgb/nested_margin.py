#!/usr/bin/env python
"""Doubly nested ridge logistic margins: the XGB base margin of the stage-1, stage-2 and stage-8 models.

Outer fold k: on each of 5 inner splits (StratifiedKFold(5, seed 42)) of the outer-train rows, the stage-1 features are
rebuilt with encodings fitted on the inner-train rows only (cross-fitted inside them, noise sd 0.04) and a ridge
logistic regression (sum loss, lambda 10 = C 0.1, free intercept, inputs standardized on the fit rows and clipped to
+-15, categoricals one-hot, Newton steps in float64 on GPU) predicts the inner-held rows. The query rows (fold k + test)
get the margin of the same regression fitted on all outer-train rows.
Writes cache/stage1/nested_margin_f{k}.npz (fit, query, fit_margin, query_margin).
Usage: python nested_margin.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.pairs import fold_list  # noqa: E402
from features import Builder  # noqa: E402
from layout import STAGE1 as CACHE  # noqa: E402


def dense_lr_input(X, types):
    cols = []
    for j, t in enumerate(types):
        if t == 'c':
            # raw categories have at most 4 levels; level 0 is the reference
            for val in range(1, 4):
                cols.append((X[:, j] == val).astype('float32'))
        else:
            cols.append(X[:, j])
    return np.column_stack(cols).astype('float32')


def fit_lr_predict(A, B, ya):
    import torch
    mu = A.mean(axis=0, dtype=np.float64)
    sd = np.maximum(A.std(axis=0, dtype=np.float64), 1e-3)
    A = np.column_stack([np.ones(len(A)), np.clip((A-mu)/sd, -15, 15)])
    B = np.column_stack([np.ones(len(B)), np.clip((B-mu)/sd, -15, 15)])
    x = torch.tensor(A, dtype=torch.float64, device='cuda')
    y = torch.tensor(ya, dtype=torch.float64, device='cuda')
    beta = torch.zeros(x.shape[1], dtype=torch.float64, device='cuda')
    lam = torch.full_like(beta, 10.)
    lam[0] = 0
    for _ in range(16):
        p = torch.sigmoid(x@beta)
        w = (p*(1-p)).clamp(min=1e-8)
        g = x.T@(y-p)-lam*beta
        H = x.T@(x*w[:, None])+torch.diag(lam+1e-8)
        step = torch.linalg.solve(H, g)
        if step.abs().max() > 5:
            step *= 5/step.abs().max()
        beta += step
        if float(step.abs().max()) < 1e-7:
            break
    coef = beta.cpu().numpy()
    return (B@coef).astype('float32')


def margins(builder, k):
    path = CACHE/f'nested_margin_f{k}.npz'
    folds = np.load(CACHE/'fold_id.npy')
    fit = np.flatnonzero(folds != k)
    valid = np.flatnonzero(folds == k)
    query = np.r_[valid, np.arange(builder.n, len(builder.base))]
    if path.exists():
        d = np.load(path)
        assert np.array_equal(d['fit'], fit) and np.array_equal(d['query'], query)
        return d['fit_margin'], d['query_margin'], query
    result = np.empty(len(fit), dtype='float32')
    for j, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(fit, builder.y[fit])):
        X, V, _, types = builder.build(fit[a], fit[b], noise=.04)
        result[b] = fit_lr_predict(dense_lr_input(X, types), dense_lr_input(V, types), builder.y[fit[a]])
        print('nested margin', k, j, flush=True)
    X, V, _, types = builder.build(fit, query, noise=.04)
    pred = fit_lr_predict(dense_lr_input(X, types), dense_lr_input(V, types), builder.y[fit])
    np.savez(path, fit=fit, query=query, fit_margin=result, query_margin=pred)
    return result, pred, query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    b = Builder()
    for k in fold_list(a.folds):
        margins(b, k)


if __name__ == '__main__':
    main()
