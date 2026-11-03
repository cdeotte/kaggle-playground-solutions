#!/usr/bin/env python
"""XGBoost on the 99 cuML-built columns of build_cuml_features.py (pair cuml_built_features_xgb).

Per outer fold: early stopping (AUC, patience 300, at most 8000 rounds) on a stratified 10% of the outer-train rows
(train_test_split random_state 99), then a refit on all outer-train rows for int(1.11 x best rounds). One seed (42).
XGBoost: hist on CUDA, depth 6, eta 0.02, min_child_weight 20, subsample 0.7, colsample_bytree 0.7, colsample_bynode 0.6,
lambda 4, alpha 0.02, max_bin 1024, num_parallel_tree 6. Test = mean of the 5 fold models.
Usage: python train_cuml_xgb.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import cache_dir, folds as fold_ids, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'cuml_built_features_xgb'
SEED = 42
XPAR = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=6, eta=0.02,
            min_child_weight=20, subsample=0.7, colsample_bytree=0.7, colsample_bynode=0.6, reg_lambda=4.0, reg_alpha=0.02,
            max_bin=1024, num_parallel_tree=6, seed=SEED)


def run_fold(fold, y, folds):
    t0 = time.time()
    trn_idx, val_idx = np.where(folds != fold)[0], np.where(folds == fold)[0]
    C = cache_dir('cuml_density_models')
    A = pd.read_parquet(C / f'cf_f{fold}_tr.parquet')
    B = pd.read_parquet(C / f'cf_f{fold}_te.parquet')
    X = A.values.astype(np.float32)
    Xtr, ytr = X[trn_idx], y[trn_idx].astype(np.float32)
    i_tr, i_es = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=99, stratify=ytr)
    bst = xgb.train(XPAR, xgb.DMatrix(Xtr[i_tr], label=ytr[i_tr]), num_boost_round=8000,
                    evals=[(xgb.DMatrix(Xtr[i_es], label=ytr[i_es]), 'es')], early_stopping_rounds=300, verbose_eval=False)
    best = bst.best_iteration + 1
    bst2 = xgb.train(XPAR, xgb.DMatrix(Xtr, label=ytr), num_boost_round=int(best * 1.11), verbose_eval=False)
    pv = bst2.predict(xgb.DMatrix(X[val_idx]))
    pt = bst2.predict(xgb.DMatrix(B.values.astype(np.float32)))
    print(f'[{time.time() - t0:.0f}s] {PAIR} fold {fold} VAL_AUC={roc_auc_score(y[val_idx], pv):.6f} '
          f'(es {best}, refit {int(best * 1.11)}, {A.shape[1]} feat)', flush=True)
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
