#!/usr/bin/env python
"""Public notebook "XGBoost for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/xgboost-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories (XGBoost native categoricals), missing_count and
9 screen-time ratio features (NaN kept), and per fold a smoothed target encoding (smoothing 10) and a frequency encoding
of every raw predictor. The notebook computed the encodings once on all training rows before its CV loop; here they are
computed inside each fold (inner 5-fold out-of-fold for the training rows, the fold's training-row map for the
validation and test rows).
Model (the notebook's): XGBClassifier, lr 0.01, depth 7, min_child_weight 10, subsample 0.8, colsample_bytree 0.5,
alpha 0.1, lambda 1, max_bin 2000, hist on CUDA, seed 42, up to 20,000 rounds with early stopping 150 on the validation
AUC. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import ID, TARGET, categorical_frames, nested_encodings, simple_features  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402

NAME = 'omidbaghchehsaraei_xgb'
THREADS = 24
PARAMS = dict(objective='binary:logistic', eval_metric='auc', tree_method='hist', device='cuda',
              enable_categorical=True, learning_rate=0.01, max_depth=7, min_child_weight=10, subsample=0.8,
              colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0, n_estimators=20000, random_state=42, max_bin=2000,
              early_stopping_rounds=150, n_jobs=THREADS)


def main():
    fold_ids = fold_args()
    train, test, initial = categorical_frames()
    engineered, engineered_test = simple_features(train, initial), simple_features(test, initial)
    base_features = [c for c in engineered if c not in [TARGET, ID]]
    y = train[TARGET].to_numpy(np.int8)

    def fit_fold(fold, tr_idx, va_idx):
        tr, va = engineered.iloc[tr_idx].copy(), engineered.iloc[va_idx].copy()
        tr_enc, va_enc, te_enc = nested_encodings(tr, y[tr_idx], va, engineered_test, initial)
        xtr = pd.concat([tr[base_features].reset_index(drop=True), tr_enc], axis=1)
        xva = pd.concat([va[base_features].reset_index(drop=True), va_enc], axis=1)
        xte = pd.concat([engineered_test[base_features].reset_index(drop=True), te_enc], axis=1)
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(xtr, y[tr_idx], eval_set=[(xva, y[va_idx])], verbose=500)
        print(f'fold {fold}: best iteration {model.best_iteration}', flush=True)
        return model.predict_proba(xva)[:, 1], model.predict_proba(xte)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
