#!/usr/bin/env python
"""TabPFN-3.5, zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 estimators, package defaults (automatic mixed precision), fit_mode='fit_with_cache', ignore_pretraining_limits=True.
The 4 categoricals are passed as categorical features; NaN (Arrival Delay) is kept.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'tabpfn35'
MODEL_PATH = 'tabpfn-v3.5-20260909.safetensors'  # TabPFN-3.5 classifier checkpoint (see the tabpfn package docs)
N_ESTIMATORS = 8
QUERY_CHUNK = 50_000


def predict(Xc, yc, Xq, meta, fold):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(model_path=MODEL_PATH, device='cuda', categorical_features_indices=meta['cat_idx'],
                           ignore_pretraining_limits=True, random_state=42 + fold, n_estimators=N_ESTIMATORS,
                           fit_mode='fit_with_cache')
    clf.fit(Xc, yc)
    out = [np.asarray(clf.predict_proba(Xq[s:s + QUERY_CHUNK]))[:, 1] for s in range(0, len(Xq), QUERY_CHUNK)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    import torch
    from sklearn.metrics import roc_auc_score
    torch.backends.cuda.matmul.allow_tf32 = True
    X, Xt, meta = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(X[tr], y[tr], np.concatenate([X[va], Xt]), meta, k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
