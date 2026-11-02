#!/usr/bin/env python
"""Pair gpt2_likelihood_xgb: XGBoost (GPU) per fold on 333 label-free generator features, in this column order:
  e55r12 (23)        order-marginal features of the e55 replica over 12 random orders (score_orders.py, build_order_features.py)
  inv_e55 (4)        inverse query of the e55 replica over 6 random orders (score_inverse.py)
  conditional (306)  build_conditional_features.py
Per outer fold: fit on the outer-train rows with early stopping (100 rounds, at most 3,000) on a stratified 10% split of them,
then predict the held-out fold and the test rows at the best iteration. OOF = sigmoid(margin); test = mean of the 5 fold models'
probabilities.
Usage: python fit_combo_xgb.py"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, save_part  # noqa: E402

NAME = 'gpt2_likelihood_xgb'
FEATURE_FILES = ['e55r12.npy', 'inv_e55.npy', 'conditional_features.npy']


def main():
    y = target()
    f = folds(y)
    n_train = len(y)
    X = np.column_stack([np.load(HERE / 'features' / p) for p in FEATURE_FILES]).astype(np.float64)
    Xtr, Xte = X[:n_train], X[n_train:]
    for k in range(5):
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        fit, es = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=7000 + k)
        params = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=4, eta=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0, seed=42 + k)
        booster = xgb.train(params, xgb.DMatrix(Xtr[fit], label=y[fit]), 3000, evals=[(xgb.DMatrix(Xtr[es], label=y[es]), 'es')],
                            early_stopping_rounds=100, verbose_eval=False)
        it = (0, booster.best_iteration + 1)
        valid = booster.predict(xgb.DMatrix(Xtr[va]), iteration_range=it, output_margin=True).astype(np.float64)
        test = booster.predict(xgb.DMatrix(Xte), iteration_range=it, output_margin=True).astype(np.float64)
        save_part(HERE, NAME, k, expit(valid), expit(test))
        print(f'fold {k}: AUC {roc_auc_score(y[va], valid):.6f}', flush=True)
    assemble(HERE, NAME, f, y)


if __name__ == '__main__':
    main()
