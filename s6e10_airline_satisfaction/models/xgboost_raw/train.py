#!/usr/bin/env python
"""XGBoost on the 21 raw features (no feature engineering), averaged over 8 seeds.

Input: common.data.raw_arrays (categoricals as integer codes, NaN kept), the same arrays the foundation models read.
The settings come from a grid search on the 5 folds with a fixed number of rounds (no early stopping on the scored fold).
max_bin 4096 is the key setting: Flight Distance has ~3.5k distinct values and acts as a route id, so every distance needs
its own split (+8.9e-4 AUC over the default 256 bins).
Test = mean of the 5 fold models; the pair is the mean of the 8 seeds.
Usage: python train.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, raw_arrays, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PARAMS = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=9, learning_rate=0.026,
              subsample=0.9, colsample_bytree=0.7, min_child_weight=1.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bin=4096)
ROUNDS = 1030
SEEDS = list(range(1, 9))


def main():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    X, Xt, _ = raw_arrays()
    y = target()
    f = folds(y)
    dtest = xgb.DMatrix(Xt, missing=np.nan)
    oof, test = np.zeros(len(y)), np.zeros(len(Xt))
    for seed in SEEDS:
        o = np.zeros(len(y))
        for k in range(5):
            tr, va = f != k, f == k
            dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], missing=np.nan, max_bin=PARAMS['max_bin'])
            bst = xgb.train({**PARAMS, 'seed': seed}, dtr, ROUNDS)
            o[va] = bst.predict(xgb.DMatrix(X[va], missing=np.nan))
            test += bst.predict(dtest) / (5 * len(SEEDS))
        print(f'seed {seed}: OOF AUC {roc_auc_score(y, o):.6f}', flush=True)
        oof += o / len(SEEDS)
    print(f'{len(SEEDS)}-seed OOF AUC {roc_auc_score(y, oof):.6f}')
    save_pair(HERE, 'xgboost_raw', oof, test)


if __name__ == '__main__':
    main()
