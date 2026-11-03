#!/usr/bin/env python
"""Public notebook "S6E8: XGBoost | SAP" (version 2) run on our 5 folds instead of its 7.
https://www.kaggle.com/code/abdullahsafwan333/s6e8-xgboost-sap

Features (lib/sap_features.py, label-free): the 9 numeric predictors, 8 domain features, ones / tenths / hundredths
digit features and the 3 native categoricals.
Model (the notebook's): xgboost.train, 4,377 rounds (fixed, no early stopping), eta 0.0515, depth 5, subsample 0.778,
colsample_bytree 0.649, colsample_bylevel 0.877, min_child_weight 9.24, lambda 0.129, alpha 0.00924, gamma 0.0705,
max_bin 512, hist on CUDA, native categoricals, seed 42 + fold. The notebook used StratifiedKFold(7, shuffle=True,
random_state=42); our run uses our 5 folds (seeds 42 ... 46). Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import fold_args, run_folds  # noqa: E402
from lib.sap_features import load  # noqa: E402

NAME = 'abdullahsafwan333_xgb'
THREADS = 24
PARAMS = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda', max_depth=5,
              eta=0.0515, subsample=0.778, colsample_bytree=0.649, colsample_bylevel=0.877, min_child_weight=9.24,
              reg_lambda=0.129, reg_alpha=0.00924, gamma=0.0705, max_bin=512, nthread=THREADS)
NUM_ROUNDS = 4377


def main():
    fold_ids = fold_args()
    x, xt, y = load()
    dtest = xgb.DMatrix(xt, enable_categorical=True)

    def fit_fold(fold, tr, va):
        dtrain = xgb.DMatrix(x.iloc[tr], y[tr], enable_categorical=True)
        dvalid = xgb.DMatrix(x.iloc[va], enable_categorical=True)
        booster = xgb.train({**PARAMS, 'seed': 42 + fold}, dtrain, num_boost_round=NUM_ROUNDS)
        return booster.predict(dvalid), booster.predict(dtest)

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
