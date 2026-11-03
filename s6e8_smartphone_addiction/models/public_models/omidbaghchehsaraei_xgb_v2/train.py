#!/usr/bin/env python
"""Public notebook "XGBoost v2 for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-v2-for-predicting-smartphone-addiction

features.py (the notebook's cells 1-5, label-free): the 9 numeric predictors, missing count, 22 ratio / intensity /
log features, sin / cos of 6 hour columns and 3 ratios, decile bins (edges from train) of 8 columns with their
sin / cos, and 5 bin-group mean differences / ratios over train + test.
Per fold, the 12 raw predictors and the 8 bins get a smoothed target encoding (smoothing 10) and a frequency encoding
(counts on the fold's training rows). The notebook encoded the training rows with the map fitted on those same rows;
in our run they get 5-fold inner out-of-fold target encodings (lib/ratio_te.nested_encodings; validation and test rows
use the fold's training-row map, as in the notebook). The model sees the numeric columns plus these encodings.
Model (the notebook's): XGBClassifier, lr 0.01, depth 7, colsample_bytree 0.2, max_bin 256, hist, seed 42, up to 100,000
rounds with early stopping 200 on the validation AUC. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import nested_encodings  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'omidbaghchehsaraei_xgb_v2'
THREADS = 12
PARAMS = dict(verbosity=0, random_state=42, n_estimators=100000, learning_rate=0.01, colsample_bytree=0.2, max_depth=7,
              min_child_weight=1, subsample=1, colsample_bynode=1, reg_lambda=1, max_bin=256, num_parallel_tree=1,
              objective='binary:logistic', eval_metric='auc', tree_method='hist', n_jobs=THREADS,
              early_stopping_rounds=200)


def main():
    fold_ids = fold_args()
    import features as nb  # builds the notebook's features on import

    train, test, cat_cols = nb.train_df, nb.test_df, nb.cat_cols
    numeric = train.select_dtypes(include=[np.number]).columns.drop(['id', 'addicted_label'], errors='ignore').tolist()
    y = pd.Series(train['addicted_label'].to_numpy(np.int8))

    def fit_fold(fold, tr_idx, va_idx):
        tr, va, te = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy(), test.copy()
        ytr, yva = y.iloc[tr_idx].reset_index(drop=True), y.iloc[va_idx]
        tr_enc, va_enc, te_enc = nested_encodings(tr, ytr, va, te, cat_cols)
        xtr = pd.concat([tr[numeric].reset_index(drop=True), tr_enc], axis=1)
        xva = pd.concat([va[numeric].reset_index(drop=True), va_enc], axis=1)
        xte = pd.concat([te[numeric].reset_index(drop=True), te_enc], axis=1)
        model = XGBClassifier(**PARAMS)
        model.fit(xtr, ytr, eval_set=[(xtr, ytr), (xva, yva)], verbose=100)
        print(f'fold {fold}: best iteration {model.best_iteration}', flush=True)
        return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y.to_numpy())


if __name__ == '__main__':
    main()
