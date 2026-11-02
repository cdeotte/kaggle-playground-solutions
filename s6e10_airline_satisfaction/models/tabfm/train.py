#!/usr/bin/env python
"""TabFM 1.0 (1.64B parameters, bf16), zero-shot, full context, on the 21 raw features.

Context = all outer-train rows of the fold (559,708 rows) with their labels; queries = the held-out fold + all test rows.
8 estimators. Stock TabFMClassifier (preprocessing, ensemble members, class shifts, temperature, logit averaging) with only
_batch_forward replaced by the exact context cache in tabfm_cache.py, so all queries go through ONE predict_proba call.
The 4 categoricals are passed as pandas category columns; NaN kept.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

NAME = 'tabfm'
N_ESTIMATORS = 8


def frame(X, meta):
    df = pd.DataFrame(X, columns=meta['features'])
    for j in meta['cat_idx']:
        c = meta['features'][j]
        df[c] = pd.Categorical(df[c].round().astype(int).map(dict(enumerate(meta['cat_codes'][c]))))
    return df


def predict(model, Xc, yc, Xq, meta, fold):
    import tabfm_cache
    from tabfm import TabFMClassifier
    clf = TabFMClassifier(model=model, n_estimators=N_ESTIMATORS, random_state=42 + fold)
    clf.fit(frame(Xc, meta), yc)
    tabfm_cache.install(clf, q_chunk=32768, icl_chunk=65536)
    return clf.predict_proba(frame(Xq, meta))[:, 1].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    model = tabfm_v1_0_0.load(model_type='classification', device='cuda')
    X, Xt, meta = raw_arrays()
    y = target()
    f = folds(y)
    for k in fold_list(a.folds):
        if have_part(HERE, NAME, k):
            continue
        tr, va = f != k, f == k
        p = predict(model, X[tr], y[tr], np.concatenate([X[va], Xt]), meta, k)
        nv = int(va.sum())
        save_part(HERE, NAME, k, p[:nv], p[nv:])
        print(f'fold {k}: AUC {roc_auc_score(y[va], p[:nv]):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
