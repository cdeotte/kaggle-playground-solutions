#!/usr/bin/env python
"""Public notebook "Smartphone Addiction | XGB" (version 3) run on our 5 folds: its XGB5 model.
https://www.kaggle.com/code/mikhailnaumov/smartphone-addiction-xgb

The notebook loads XGB5's predictions from a dataset; here the declared XGB5 recipe is trained.
Features (preprocessing.py, the notebook's Preprocessing class, refitted on the training rows of every fold): the 12 raw
columns; for every raw column the mean target of that value in the 7,500-row original dataset; every numeric column
also as a string category (_cat) and rounded to 0 and 1 decimals; 9 screen-time ratio features; constrained-imputation
features (bounds of a missing daily total / missing component, slack, case id, number missing); missing count;
normalized frequency of every categorical and _cat column (counted on the fold's training rows); constant columns
dropped.
Per fold, a scikit-learn TargetEncoder (cv=5, smooth 'auto', shuffle, seed 42) of the categorical columns is fitted on
the training rows (cross-fitted for those rows) and applied to validation and test; the _cat columns are then dropped.
The notebook asks for target_type='multiclass' on this binary target; our run uses 'binary' (one column per feature).
Model (the notebook's XGB5): XGBClassifier, lr 0.01, depth 6, min_child_weight 5, subsample 0.894, colsample_bytree
0.184, lambda 4.32, alpha 2.21, max_bin 1024, native categoricals, CUDA, seed 42, up to 50,000 rounds with early
stopping 100 on the validation AUC. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import read_original, read_test, read_train  # noqa: E402
from lib.runner import fold_args, run_folds  # noqa: E402
from preprocessing import Preprocessing  # noqa: E402

NAME = 'mikhailnaumov_xgb'
TARGET, ID = 'addicted_label', 'id'
THREADS = 24
PARAMS = {'n_estimators': 50000, 'learning_rate': 0.01, 'random_state': 42, 'objective': 'binary:logistic',
          'eval_metric': 'auc', 'enable_categorical': True, 'device': 'cuda', 'early_stopping_rounds': 100,
          'max_bin': 1024, 'reg_lambda': 4.315056676630924, 'reg_alpha': 2.214941127024784,
          'colsample_bytree': 0.18360565346067603, 'subsample': 0.8941761954081464, 'max_depth': 6,
          'min_child_weight': 5, 'n_jobs': THREADS}


def main():
    fold_ids = fold_args()
    train = read_train().set_index(ID)
    test = read_test().set_index(ID)
    original = read_original()[train.columns]
    x = train.drop(columns=[TARGET])
    y = pd.Series(train[TARGET].to_numpy(np.int8), index=x.index)

    def fit_fold(fold, train_position, valid_position):
        raw_train = x.iloc[train_position].copy()
        raw_valid = x.iloc[valid_position].copy()
        y_train = y.iloc[train_position].reset_index(drop=True)
        y_valid = y.iloc[valid_position].reset_index(drop=True)

        preprocessor = Preprocessing(TARGET)
        preprocessor.fit(raw_train, y.iloc[train_position], original)
        x_train, categorical, _ = preprocessor.transform(raw_train)
        x_valid, _, _ = preprocessor.transform(raw_valid)
        x_test, _, _ = preprocessor.transform(test)
        x_train.reset_index(drop=True, inplace=True)
        x_valid.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)

        encoder = TargetEncoder(random_state=42, shuffle=True, cv=5, smooth='auto', target_type='binary')
        names = [f'{column}_TE' for column in categorical]
        x_train[names] = encoder.fit_transform(x_train[categorical], y_train).astype('float32')
        x_valid[names] = encoder.transform(x_valid[categorical]).astype('float32')
        x_test[names] = encoder.transform(x_test[categorical]).astype('float32')
        drop = [column for column in categorical if column.endswith('_cat')]
        x_train.drop(columns=drop, inplace=True)
        x_valid.drop(columns=drop, inplace=True)
        x_test.drop(columns=drop, inplace=True)

        model = XGBClassifier(**PARAMS)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        print(f'fold {fold}: best iteration {model.best_iteration}', flush=True)
        return model.predict_proba(x_valid)[:, 1], model.predict_proba(x_test)[:, 1]

    run_folds(NAME, fit_fold, fold_ids, y=y.to_numpy())


if __name__ == '__main__':
    main()
