#!/usr/bin/env python
"""Public notebook "S6E8: LightGBM | SAP" (version 1) run on our 5 folds instead of its 7.
https://www.kaggle.com/code/abdullahsafwan333/s6e8-lightgbm-sap

Features (lib/sap_features.py, label-free): the 9 numeric predictors, 8 domain features, ones / tenths / hundredths
digit features and the 3 native categoricals.
Model (the notebook's): LGBMClassifier, 8,775 rounds (fixed, no early stopping), lr 0.0375, 39 leaves,
min_child_samples 365, colsample_bytree 0.865, subsample 0.976 every round, alpha 0.246, lambda 30.6, min_split_gain
0.34, max_bin 255, GPU, seed 42 + fold. The notebook used StratifiedKFold(7, shuffle=True, random_state=42); our run
uses our 5 folds (seeds 42 ... 46). Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.runner import fold_args, run_folds  # noqa: E402
from lib.sap_features import load  # noqa: E402

NAME = 'abdullahsafwan333_lgbm'
THREADS = 24
PARAMS = dict(objective='binary', metric='auc', device_type='gpu', n_estimators=8775, learning_rate=0.0375,
              num_leaves=39, min_child_samples=365, colsample_bytree=0.865, subsample=0.976, subsample_freq=1,
              reg_alpha=0.246, reg_lambda=30.6, min_split_gain=0.340, max_bin=255, verbose=-1, n_jobs=THREADS)


def main():
    fold_ids = fold_args()
    x, xt, y = load()

    def fit_fold(fold, tr, va):
        model = lgb.LGBMClassifier(**PARAMS, random_state=42 + fold)
        model.fit(x.iloc[tr], y[tr])
        return model.predict_proba(x.iloc[va])[:, 1], model.predict_proba(xt)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
