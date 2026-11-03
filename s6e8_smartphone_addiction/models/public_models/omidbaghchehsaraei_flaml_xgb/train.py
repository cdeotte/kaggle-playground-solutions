#!/usr/bin/env python
"""Public notebook "FLAML[XGBoost] for Predicting Smartphone Addiction" (version 1) run on our 5 folds.
https://www.kaggle.com/code/omidbaghchehsaraei/flaml-xgboost-for-predicting-smartphone-addiction

Features (lib/ratio_te.py): the 12 raw predictors as string categories, missing_count and 9 screen-time ratio features
(NaN kept), and per fold a smoothed target encoding (smoothing 10) and a frequency encoding of every raw predictor. The
notebook computed the encodings once on all training rows before its CV loop; here they are computed inside each fold
(inner 5-fold out-of-fold for the training rows, the fold's training-row map for the validation and test rows).
Model (the notebook's): FLAML AutoML restricted to xgb_limitdepth, metric roc_auc, 7,000 s budget per fold, the fold's
validation rows as FLAML's validation set, early_stop, no ensemble, seed 42 + fold. The notebook converted each fold's
predictions to ranks (rankdata / n) before averaging; our run keeps the probabilities. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]   (about 2 hours per fold)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from lib.ratio_te import ID, TARGET, categorical_frames, nested_encodings, simple_features  # noqa: E402
from lib.runner import fold_args, run_folds, work_dir, working_directory  # noqa: E402

NAME = 'omidbaghchehsaraei_flaml_xgb'
TIME_BUDGET = 7000


def main():
    fold_ids = fold_args()
    train, test, initial = categorical_frames()
    engineered, engineered_test = simple_features(train, initial), simple_features(test, initial)
    base_features = [column for column in engineered.columns if column not in (TARGET, ID)]
    y = train[TARGET].to_numpy(np.int8)

    def fit_fold(fold, train_index, valid_index):
        train_fold, valid_fold = engineered.iloc[train_index].copy(), engineered.iloc[valid_index].copy()
        train_encoded, valid_encoded, test_encoded = nested_encodings(train_fold, y[train_index], valid_fold,
                                                                      engineered_test, initial)
        x_train = pd.concat([train_fold[base_features].reset_index(drop=True), train_encoded], axis=1)
        x_valid = pd.concat([valid_fold[base_features].reset_index(drop=True), valid_encoded], axis=1)
        x_test = pd.concat([engineered_test[base_features].reset_index(drop=True), test_encoded], axis=1)
        with working_directory(work_dir(NAME)):
            automl = AutoML()
            automl.fit(X_train=x_train, y_train=y[train_index], X_val=x_valid, y_val=y[valid_index],
                       time_budget=TIME_BUDGET, metric='roc_auc', task='classification',
                       estimator_list=['xgb_limitdepth'], ensemble=False, early_stop=True, seed=42 + fold, verbose=0)
            valid_pred = automl.predict_proba(x_valid)[:, 1]
            test_pred = automl.predict_proba(x_test)[:, 1]
        print(f'fold {fold}: best {automl.best_estimator} {automl.best_config}', flush=True)
        return valid_pred, test_pred

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
