#!/usr/bin/env python
"""Our run of najiama's public notebook "pure-lgbm-model-cv-0-94606-lb-0-94637" (version 1) on our 5 folds.
https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637

Features (built once on train + test; no competition labels):
  - Number_of_Cars_Owned is dropped; digit-position features (x.fillna(0) // 10**k) % 10, k = -4..3, of the 6 other numeric
    columns;
  - for each of the 12 raw columns: the mean target of that value in the original 10,000-row dataset (global mean if unseen);
  - every numeric and digit column also as a string category; frequency of every category over train + test;
  - 4 fixed rules: income == 30,000, income >= 170,537, income in [38,000, 42,000], environmental concern == 1;
  - columns that are constant in train or test and exact duplicates (|corr| == 1 on train) are dropped -> 105 columns,
    37 of them string categories.
Per fold: two scikit-learn TargetEncoders (smooth 'auto' and 10, 5-fold cross-fitted on the training rows, shuffle, seed 42)
replace the 37 string columns; validation and test rows use the encoders fitted on all training rows -> 142 features.
Model: LGBMClassifier with the notebook's parameters and early stopping (patience 700); predictions at the best iteration.
Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4]"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common.data import TARGET, folds, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

PAIR = 'lgbm_najiama'
ID = 'id'
PARAMS = {'n_estimators': 40_000, 'learning_rate': 0.005, 'max_depth': 5, 'num_leaves': 32, 'min_child_samples': 10,
          'subsample': 0.8, 'colsample_bytree': 0.3, 'reg_alpha': 0.071, 'reg_lambda': 2.0, 'max_bin': 1024,
          'random_state': 42, 'feature_pre_filter': False, 'metric': 'auc', 'n_jobs': 24, 'verbose': -1}


def build_features(train, test, original, y):
    """The notebook's feature table of the train and test rows, and the string columns to target-encode."""
    train, test, original = train.copy(), test.copy(), original.copy()
    original[TARGET] = original[TARGET].map({'No': 0, 'Yes': 1})
    train[TARGET] = y
    train['is_train'] = 1
    test['is_train'] = 0
    test[TARGET] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(columns=['Number_of_Cars_Owned'], inplace=True, errors='ignore')

    cat_cols = combined.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = [c for c in combined.columns if c not in cat_cols + [ID, 'is_train', TARGET]]
    digit_cols = []
    for column in list(num_cols):
        for k in range(-4, 4):
            name = f'{column}_digit{k}'
            combined[name] = (combined[column].fillna(0) // (10 ** k) % 10).astype('int8')
            digit_cols.append(name)
    num_cols.extend(digit_cols)

    # target mean of each value in the original dataset
    original_mean = float(original[TARGET].mean())
    for column in cat_cols + num_cols:
        if column in original.columns:
            means = original.groupby(column, observed=False)[TARGET].mean()
            combined[f'{column}_org_mean'] = combined[column].map(means).fillna(original_mean).astype(float)

    # numeric columns as string categories; frequency of every category over train + test
    num_to_cat_cols = []
    for column in num_cols:
        combined[f'{column}_cat'] = combined[column].fillna('NaN').astype(str)
        num_to_cat_cols.append(f'{column}_cat')
    all_cats = cat_cols + num_to_cat_cols
    for column in all_cats:
        frequency = combined[column].value_counts(normalize=True).to_dict()
        combined[f'{column}_fe'] = combined[column].map(frequency).astype(float).fillna(0.0)

    combined['is_30k_spike'] = (combined['Annual_Income_USD'] == 30000.0).astype('int8')
    combined['is_millionaire_cliff'] = (combined['Annual_Income_USD'] >= 170537.0).astype('int8')
    combined['is_dead_zone'] = ((combined['Annual_Income_USD'] >= 38000.0)
                                & (combined['Annual_Income_USD'] <= 42000.0)).astype('int8')
    combined['is_env_hater'] = (combined['Environmental_Concern_Level'] == 1).astype('int8')

    tr = combined.loc[combined['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
    te = combined.loc[combined['is_train'] == 0].drop(columns=['is_train', TARGET]).reset_index(drop=True)
    eval_cols = [c for c in tr.columns if c not in [ID, TARGET] and pd.api.types.is_numeric_dtype(tr[c])]
    corr = tr[eval_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if bool((upper[c] == 1.0).any())]
    drop_const = [c for c in tr.columns if tr[c].nunique() == 1] + [c for c in te.columns if te[c].nunique() == 1]
    dropped = set(drop_corr).union(drop_const)
    dropped.discard(ID)
    dropped.discard(TARGET)
    tr = tr.drop(columns=list(dropped), errors='ignore')
    te = te.drop(columns=list(dropped), errors='ignore')

    features = [c for c in te.columns if c != ID]
    te_cols = [c for c in all_cats if c not in dropped]
    assert len(features) == 105 and len(te_cols) == 37, (len(features), len(te_cols))
    return tr[features].copy(), te[features].copy(), te_cols


def encode_fold(X, X_test, y, fit_index, valid_index, te_cols):
    fit, valid, test = X.iloc[fit_index].copy(), X.iloc[valid_index].copy(), X_test.copy()
    for smooth, suffix in [('auto', 'auto'), (10.0, '10')]:
        encoder = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42, target_type='binary')
        fit_enc = encoder.fit_transform(fit[te_cols], y[fit_index])
        valid_enc = encoder.transform(valid[te_cols])
        test_enc = encoder.transform(test[te_cols])
        for j, column in enumerate(te_cols):
            name = f'{column}_TE_{suffix}'
            fit[name] = fit_enc[:, j].astype('float32')
            valid[name] = valid_enc[:, j].astype('float32')
            test[name] = test_enc[:, j].astype('float32')
    fit, valid, test = (d.drop(columns=te_cols) for d in (fit, valid, test))
    assert fit.columns.tolist() == valid.columns.tolist() == test.columns.tolist() and len(fit.columns) == 142
    return fit, valid, test


def run_fold(X, X_test, te_cols, y, f, k):
    fit_index, valid_index = np.flatnonzero(f != k), np.flatnonzero(f == k)
    x_fit, x_valid, x_test = encode_fold(X, X_test, y, fit_index, valid_index, te_cols)
    model = LGBMClassifier(**PARAMS)
    model.fit(x_fit, y[fit_index], eval_set=[(x_valid, y[valid_index])], eval_metric='auc',
              callbacks=[early_stopping(700, verbose=False), log_evaluation(1000)])
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
        X, X_test, te_cols = build_features(train, test, read_original(), y)
        print(f'train {X.shape}, test {X_test.shape}, {len(te_cols)} target-encoded columns', flush=True)
        for k in todo:
            pv, pt, best = run_fold(X, X_test, te_cols, y, f, k)
            save_part(HERE, PAIR, k, pv, pt)
            print(f'fold {k}: AUC {roc_auc_score(y[f == k], pv):.6f}, best iteration {best}', flush=True)
    assemble(HERE, PAIR, f, y)


if __name__ == '__main__':
    main()
