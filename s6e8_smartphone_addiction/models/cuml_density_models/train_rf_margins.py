#!/usr/bin/env python
"""cuML random forest on the 282-column matrix + the 5 nested margins, pruned per fold to the top 40 columns
(pair cuml_rf_nested_margins).

Per outer fold:
  matrix    cache/M_f<k>.npy (build_rf_matrix.py) + cache/margins_f<k>_*.parquet (build_rf_margins.py) = 287 columns
  ranking   XGBoost gain of a 600-round model fitted on the outer-train rows (hist on CUDA, depth 6, eta 0.05, subsample 0.8,
            colsample_bytree 0.7, seed 42); the top 40 columns are kept (cache/sel_f<k>.json)
  model     cuml RandomForestClassifier(1000 trees, max_depth 20, max_features 0.3, min_samples_leaf 20, n_bins 128,
            bootstrap), random_state 43, 44, 45; the 3 forests are averaged in logit space
Test = mean of the 5 fold models.
Usage: python train_rf_margins.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import cache_dir, folds as fold_ids, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'cuml_rf_nested_margins'
TOP_K = 40
CACHE = cache_dir('cuml_density_models')
t0 = time.time()


def lg(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def run_fold(fold, y, folds):
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier

    trn_idx, val_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
    n_tr = len(y)
    X = np.load(CACHE / f'M_f{fold}.npy')
    COLS = json.load(open(CACHE / f'M_f{fold}.cols.json'))
    Mtr = pd.read_parquet(CACHE / f'margins_f{fold}_tr.parquet')
    Mte = pd.read_parquet(CACHE / f'margins_f{fold}_te.parquet')
    X = np.hstack([X, np.vstack([Mtr.values, Mte.values]).astype(np.float32)])
    COLS = COLS + list(Mtr.columns)
    print(f'[{time.time() - t0:.0f}s] fold {fold}: {X.shape}', flush=True)

    # per-fold feature ranking from the outer-train rows
    d = xgb.DMatrix(X[:n_tr][trn_idx], label=y[trn_idx], feature_names=COLS)
    b = xgb.train(dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=6,
                       eta=0.05, subsample=0.8, colsample_bytree=0.7, seed=42), d, 600)
    g = b.get_score(importance_type='gain')
    sel = sorted(COLS, key=lambda c: -g.get(c, 0.0))[:TOP_K]
    json.dump(sel, open(CACHE / f'sel_f{fold}.json', 'w'))
    idx = [COLS.index(c) for c in sel]
    Xs = np.ascontiguousarray(X[:, idx])
    print(f'[{time.time() - t0:.0f}s] pruned to the top {TOP_K} columns', flush=True)

    Gtr, Gva, Gte = cp.asarray(Xs[:n_tr][trn_idx]), cp.asarray(Xs[:n_tr][val_idx]), cp.asarray(Xs[n_tr:])
    Ytr = cp.asarray(y[trn_idx].astype(np.int32))
    pv_l, pt_l = [], []
    for s in (1, 2, 3):
        m = RandomForestClassifier(n_estimators=1000, max_depth=20, max_features=0.3, min_samples_leaf=20, n_bins=128,
                                   bootstrap=True, random_state=42 + s, n_streams=1)
        m.fit(Gtr, Ytr)
        pv_l.append(lg(cp.asnumpy(m.predict_proba(Gva))[:, 1]))
        pt_l.append(lg(cp.asnumpy(m.predict_proba(Gte))[:, 1]))
        del m
        cp.get_default_memory_pool().free_all_blocks()
        print(f'[{time.time() - t0:.0f}s]   seed {s} done', flush=True)
    pv = 1 / (1 + np.exp(-np.mean(pv_l, 0)))
    pt = 1 / (1 + np.exp(-np.mean(pt_l, 0)))
    print(f'[{time.time() - t0:.0f}s] {PAIR} fold {fold} VAL_AUC={roc_auc_score(y[val_idx], pv):.6f}', flush=True)
    del Gtr, Gva, Gte, Ytr
    cp.get_default_memory_pool().free_all_blocks()
    return pv, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    folds = fold_ids(y)
    for k in fold_list(a.folds):
        if not have_part(HERE, PAIR, k):
            save_part(HERE, PAIR, k, *run_fold(k, y, folds))
    assemble(HERE, PAIR, folds, y)


if __name__ == '__main__':
    main()
