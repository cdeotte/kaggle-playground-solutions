#!/usr/bin/env python
"""Public notebook "Ex51 : S6E8 : FLAML : roc_auc" (version 1) run on our 5 folds.
https://www.kaggle.com/code/gauravduttaiiitb/ex51-s6e8-flaml-roc-auc

Features (the notebook's): the 9 numeric predictors and one-hot columns of the 3 categoricals (missing -> 'Unknown'),
memory-downcast (integers / floats downcast, booleans -> int8). No target-dependent features.
Model (the notebook's): FLAML AutoML(task='classification', metric='roc_auc', time_budget=9 hours) with FLAML's default
estimator list and its own internal validation; the selected single estimator predicts (no ensemble). The notebook ran
one search on all training rows; our run runs one search per fold on the fold's training rows only (the validation rows
are not shown to FLAML). In our run FLAML selected CatBoost in 3 folds and XGBoost (limited depth) in 2.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]   (9 hours per fold)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flaml import AutoML

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds, work_dir, working_directory  # noqa: E402

NAME = 'gauravduttaiiitb_flaml'
TARGET, ID = 'addicted_label', 'id'


def notebook_matrix(train, test):
    train = train.drop(columns=[ID]).copy()
    test = test.drop(columns=[ID]).copy()
    for column in test.select_dtypes(include=['object']).columns.tolist():
        train[column] = train[column].fillna('Unknown')
        test[column] = test[column].fillna('Unknown')
        train = pd.get_dummies(train, columns=[column])
        test = pd.get_dummies(test, columns=[column])
    for frame in (train, test):
        for column in frame.columns:
            if frame[column].dtype == 'int64':
                frame[column] = pd.to_numeric(frame[column], downcast='integer')
            elif frame[column].dtype == 'float64':
                frame[column] = pd.to_numeric(frame[column], downcast='float')
            elif frame[column].dtype == 'bool':
                frame[column] = frame[column].astype('int8')
    y = train.pop(TARGET).to_numpy(dtype=np.int8)
    return train, test, y


def main():
    fold_ids = fold_args()
    x, x_test, y = notebook_matrix(read_train(), read_test())

    def fit_fold(fold, train_index, valid_index):
        with working_directory(work_dir(NAME)):
            automl = AutoML()
            automl.fit(x.iloc[train_index], y[train_index], task='classification', metric='roc_auc',
                       time_budget=3600 * 9)
            valid_pred = automl.predict_proba(x.iloc[valid_index])[:, 1]
            test_pred = automl.predict_proba(x_test)[:, 1]
        print(f'fold {fold}: best {automl.best_estimator} {automl.best_config}', flush=True)
        return valid_pred, test_pred

    run_folds(NAME, fit_fold, fold_ids, y=y)


if __name__ == '__main__':
    main()
