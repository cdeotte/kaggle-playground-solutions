#!/usr/bin/env python
"""Nested cuML sub-model margins on the 282-column matrix of build_rf_matrix.py (5 columns per fold):
  m_lr     LogisticRegression(C 1, max_iter 3000) decision function
  m_svc    SVC(rbf, C 0.5, gamma 0.002) decision function
  m_knn*   KNeighborsClassifier probability at k = 50, 200, 1000
The matrix is standardised with outer-train mean / sd. Outer-train rows get inner out-of-fold values
(StratifiedKFold(5, shuffle=True, random_state=777)); validation and test rows get the value of a fit on all outer-train
rows. Writes cache/margins_f<k>_{tr,te}.parquet.
Usage: python build_rf_margins.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import cache_dir, folds as fold_ids, target  # noqa: E402
from common.pairs import fold_list  # noqa: E402

CACHE = cache_dir('cuml_density_models')
t0 = time.time()


def log(m):
    print(f'[{time.time() - t0:7.1f}s] {m}', flush=True)


def build(fold, y, folds):
    import cupy as cp
    from cuml.linear_model import LogisticRegression
    from cuml.neighbors import KNeighborsClassifier
    from cuml.svm import SVC

    trn_idx, val_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
    n_tr = len(y)
    X = np.load(CACHE / f'M_f{fold}.npy')
    log(f'fold {fold}: matrix {X.shape}')
    mu = X[:n_tr][trn_idx].mean(0)
    sd = X[:n_tr][trn_idx].std(0)
    Z = ((X - mu) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)
    Z[~np.isfinite(Z)] = 0.0
    GZ_tr = cp.asarray(Z[:n_tr][trn_idx])
    GY_tr = cp.asarray(y[trn_idx].astype(np.float32))
    GZ_va = cp.asarray(Z[:n_tr][val_idx])
    GZ_te = cp.asarray(Z[n_tr:])
    OUT_TR, OUT_TE, NAMES = [], [], []

    def nested(name, fit_predict, ncol):
        t = time.time()
        parts = np.full((n_tr, ncol), np.nan, np.float32)
        for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=777).split(np.zeros(len(trn_idx)), y[trn_idx]):
            p = fit_predict(GZ_tr[cp.asarray(i_fit)], GY_tr[cp.asarray(i_fit)])
            parts[trn_idx[i_app]] = p(GZ_tr[cp.asarray(i_app)])
            cp.get_default_memory_pool().free_all_blocks()
        pf = fit_predict(GZ_tr, GY_tr)
        parts[val_idx] = pf(GZ_va)
        te = pf(GZ_te)
        assert not np.isnan(parts).any()
        OUT_TR.append(parts)
        OUT_TE.append(np.asarray(te, np.float32))
        names = [name] if ncol == 1 else [f'{name}{i}' for i in range(ncol)]
        NAMES.extend(names)
        aucs = ' '.join(f'{roc_auc_score(y[val_idx], parts[val_idx, j]):.5f}' for j in range(ncol))
        log(f'  {name}: {time.time() - t:.0f}s, validation AUC {aucs}')

    def f_lr(Xf, yf):
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(Xf, yf)
        return lambda G: cp.asnumpy(m.decision_function(G)).reshape(-1, 1).astype(np.float32)

    def f_svc(Xf, yf):
        m = SVC(kernel='rbf', C=0.5, gamma=0.002, cache_size=4000, tol=1e-3)
        m.fit(Xf, yf)
        return lambda G: cp.asnumpy(m.decision_function(G)).reshape(-1, 1).astype(np.float32)

    def f_knn(Xf, yf):
        ms = []
        for k in (50, 200, 1000):
            mm = KNeighborsClassifier(n_neighbors=k)
            mm.fit(Xf, yf.astype(cp.int32))
            ms.append(mm)
        return lambda G: np.column_stack([cp.asnumpy(mm.predict_proba(G))[:, 1] for mm in ms]).astype(np.float32)

    nested('m_lr', f_lr, 1)
    nested('m_svc', f_svc, 1)
    nested('m_knn', f_knn, 3)
    pd.DataFrame(np.hstack(OUT_TR), columns=NAMES).to_parquet(CACHE / f'margins_f{fold}_tr.parquet')
    pd.DataFrame(np.hstack(OUT_TE), columns=NAMES).to_parquet(CACHE / f'margins_f{fold}_te.parquet')
    log(f'fold {fold}: {len(NAMES)} margin columns -> {CACHE}/margins_f{fold}_{{tr,te}}.parquet')
    del GZ_tr, GY_tr, GZ_va, GZ_te
    cp.get_default_memory_pool().free_all_blocks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    folds = fold_ids(y)
    for k in fold_list(a.folds):
        build(k, y, folds)


if __name__ == '__main__':
    main()
