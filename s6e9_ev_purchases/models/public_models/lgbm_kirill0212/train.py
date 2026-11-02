#!/usr/bin/env python
"""Our run of kirill0212's public notebook "s6e9-lightgbm" (version 2, its active model_type='lgb' path) on our 5 folds.
https://www.kaggle.com/code/kirill0212/s6e9-lightgbm

Label-free features (built once on train + test):
  - the 13 raw columns, the 6 categoricals mapped to the notebook's fixed integers;
  - digit-position features (x.fillna(0) // 10**k) % 10, k = -4..3, of all 13 columns; columns that are constant in train
    or test and exact duplicates (|corr| == 1 on train) are dropped -> 36 columns with the raw ones;
  - frequency of every raw value over train + test (13 columns).
Per fold: two catstat TargetEncoder mean views of the 13 raw columns (as strings), smooth 10 and 'auto', 5-fold cross-fitted
on the training rows (shuffle, seed 42); validation and test rows use the encoder fitted on all training rows -> 75 features.
Model: LGBMClassifier with the notebook's parameters and early stopping (patience 500); predictions at the best iteration.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catstat import TargetEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common.data import TARGET, folds, labels, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'lgbm_kirill0212'
ID = 'id'
NUMS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
        'Charging_Stations_Near_Work', 'Environmental_Concern_Level', 'Gender', 'City_Type', 'Current_Car_Type',
        'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
CATEGORY_MAPS = {
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'City_Type': {'Urban': 0, 'Suburban': 1, 'Rural': 2},
    'Current_Car_Type': {'Sedan': 0, 'SUV': 1, 'Hatchback': 2, 'Truck': 3},
    'Home_Charging_Possible': {'Yes': 0, 'No': 1},
    'Subsidy_Available': {'Yes': 0, 'No': 1},
    'Range_Anxiety_Level': {'Low': 0, 'Medium': 1, 'High': 2},
}
PARAMS = {'random_state': 60, 'feature_pre_filter': False, 'verbose': -1, 'n_estimators': 20_000, 'learning_rate': 0.02,
          'max_depth': 5, 'min_child_samples': 10, 'subsample': 0.812763123433567, 'colsample_bytree': 0.3029300829885024,
          'num_leaves': 247, 'reg_alpha': 0.07094285437903122, 'reg_lambda': 2.033039097703242495, 'max_bin': 1024,
          'n_jobs': 24}


def preprocess(frame):
    frame = frame.copy()
    for column, mapping in CATEGORY_MAPS.items():
        frame[column] = frame[column].map(mapping)
    return frame


def add_digit_features(frame):
    """The 13 columns followed by their digits at the 10^-4 ... 10^3 places (the notebook's FE function)."""
    base = frame[[c for c in frame.columns if c not in (ID, TARGET)]].copy()
    digits = {}
    for column in NUMS:
        for k in range(-4, 4):
            digits[f'{column}_digit{k}'] = (base[column].fillna(0) // (10 ** k) % 10).astype('int8')
    return pd.concat([base, pd.DataFrame(digits, index=base.index)], axis=1)


def build_features(train, test):
    """Label-free features of the (preprocessed) train and test rows, and the raw columns as strings for the encoders."""
    raw_train = train.drop(columns=[ID, TARGET]).copy()
    raw_test = test.drop(columns=[ID]).copy()
    X, X_test = add_digit_features(train), add_digit_features(test)

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    duplicates = [c for c in upper.columns if bool((upper[c] == 1.0).any())]
    constants = {c for c in X.columns if X[c].nunique() == 1} | {c for c in X_test.columns if X_test[c].nunique() == 1}
    drop = sorted(constants | set(duplicates))
    X, X_test = X.drop(columns=drop), X_test.drop(columns=drop)

    raw_train_strings = raw_train.astype('string').fillna('__NA__')
    raw_test_strings = raw_test.astype('string').fillna('__NA__')
    for column in raw_train.columns:
        frequency = pd.concat([raw_train_strings[column], raw_test_strings[column]], ignore_index=True).value_counts(
            normalize=True)
        X[f'{column}_freq'] = raw_train_strings[column].map(frequency).astype('float32')
        X_test[f'{column}_freq'] = raw_test_strings[column].map(frequency).astype('float32')
    assert X.columns.tolist() == X_test.columns.tolist()
    return X.copy(), X_test.copy(), raw_train_strings, raw_test_strings


def run_fold(X, X_test, raw_train, raw_test, y, f, k):
    fit_index, valid_index = np.flatnonzero(f != k), np.flatnonzero(f == k)
    fit_parts = [X.iloc[fit_index].reset_index(drop=True)]
    valid_parts = [X.iloc[valid_index].reset_index(drop=True)]
    test_parts = [X_test.reset_index(drop=True)]
    for smooth in [10, 'auto']:
        encoder = TargetEncoder(cv=5, smooth=smooth, stats=['mean'], shuffle=True, random_state=42, target_type='binary')
        enc_fit = encoder.fit_transform(raw_train.iloc[fit_index].reset_index(drop=True), y[fit_index])
        enc_valid = encoder.transform(raw_train.iloc[valid_index].reset_index(drop=True))
        enc_test = encoder.transform(raw_test.reset_index(drop=True))
        names = [c + f'_seed_42_smooth_{smooth}_inner_n_fold_5' for c in enc_fit.columns]
        for enc, parts in ((enc_fit, fit_parts), (enc_valid, valid_parts), (enc_test, test_parts)):
            enc.columns = names
            parts.append(enc.reset_index(drop=True))
    x_fit, x_valid, x_test = (pd.concat(p, axis=1) for p in (fit_parts, valid_parts, test_parts))

    model = LGBMClassifier(**PARAMS)
    model.fit(x_fit, y[fit_index], eval_set=[(x_valid, y[valid_index])], eval_metric='auc',
              callbacks=[log_evaluation(500), early_stopping(500)])
    pv = model.predict_proba(x_valid, num_iteration=model.best_iteration_)[:, 1]
    pt = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1]
    return pv, pt, model.best_iteration_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    f = folds(y)
    todo = [k for k in fold_list(a.folds) if not have_part(HERE, PAIR, k)]
    if todo:
        X, X_test, raw_train, raw_test = build_features(preprocess(train), preprocess(test))
        print(f'train {X.shape}, test {X_test.shape}', flush=True)
        for k in todo:
            pv, pt, best = run_fold(X, X_test, raw_train, raw_test, y, f, k)
            save_part(HERE, PAIR, k, pv, pt)
            print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}, best iteration {best}', flush=True)
    assemble(HERE, PAIR, f, y)


if __name__ == '__main__':
    main()
