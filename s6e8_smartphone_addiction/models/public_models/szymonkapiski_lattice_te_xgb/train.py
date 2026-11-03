#!/usr/bin/env python
"""Public notebook "S6E8 full lattice target encoding with XGBoost" (version 1) run on our 5 folds (the notebook's own
split is the same: StratifiedKFold(5, shuffle=True, random_state=42)).
https://www.kaggle.com/code/szymonkapiski/s6e8-full-lattice-target-encoding-with-xgboost

The notebook loads its saved predictions by default (RUN_TRAINING = False); here its training code is run.
Features (lattice.py, the notebook's cells 3-9): the 9 numeric predictors, constrained imputation of the screen-time
block (iterative imputer clipped to the bounds daily >= social + gaming + work_study; imputed values, bounds, bound
widths, missing flags, reconstructed other-screen time and share), category codes, 3 domain relations; and the lattice
keys: every predictor at full resolution, every numeric predictor rounded to 0.1 and floored, and 37 pair cells (all 36
numeric pairs and stress_level x academic_work_impact) at floor and at 0.1 resolution, 104 keys in all. Per fold, every
key gets a smoothed target encoding (smoothing 20) and the cell count:
inner 4-fold out-of-fold (seed 143) for the training rows, the fold's training-row map for validation and test.
Model (the notebook's): XGBClassifier, lr 0.03, depth 8, min_child_weight 10, subsample 0.85, colsample_bytree 0.7,
lambda 2, hist, seed 42, up to 3,000 rounds with early stopping 100 on the validation AUC.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402
from lattice import make_frames, te_block  # noqa: E402

NAME = 'szymonkapiski_lattice_te_xgb'
THREADS = 12
PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=8, min_child_weight=10, subsample=0.85,
              colsample_bytree=0.7, reg_lambda=2.0, eval_metric='auc', early_stopping_rounds=100, tree_method='hist')


def main():
    fold_ids = fold_args()
    train, test = read_train(), read_test()
    y = pd.Series(train['addicted_label'].to_numpy(np.int8))
    x_train, x_test, keys_train, keys_test = make_frames(train, test)
    print(f'{x_train.shape[1]} base features, {keys_train.shape[1]} lattice keys', flush=True)

    def fit_fold(fold, train_index, valid_index):
        encoded_train, encoded_valid, encoded_test = te_block(keys_train.iloc[train_index], y.iloc[train_index],
                                                              keys_train.iloc[valid_index], keys_test)
        a = pd.concat([x_train.iloc[train_index].reset_index(drop=True), encoded_train.reset_index(drop=True)], axis=1)
        b = pd.concat([x_train.iloc[valid_index].reset_index(drop=True), encoded_valid.reset_index(drop=True)], axis=1)
        t = pd.concat([x_test.reset_index(drop=True), encoded_test.reset_index(drop=True)], axis=1)
        model = xgb.XGBClassifier(random_state=42, n_jobs=THREADS, **PARAMS)
        model.fit(a, y.iloc[train_index], eval_set=[(b, y.iloc[valid_index])], verbose=100)
        print(f'fold {fold}: best iteration {model.best_iteration}', flush=True)
        return model.predict_proba(b)[:, 1], model.predict_proba(t)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y.to_numpy())


if __name__ == '__main__':
    main()
