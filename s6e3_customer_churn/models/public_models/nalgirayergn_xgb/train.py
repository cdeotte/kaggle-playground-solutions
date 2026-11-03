#!/usr/bin/env python
"""Public notebook "XGB - Predict Customer Churn - CV: 0.91819" run on our 5 folds.
https://www.kaggle.com/code/nalgirayergn/xgb-predict-customer-churn-cv-0-91819

Features (the notebook's, label-free, built once on train + test together): the 16 categorical columns as pandas
categories (XGBoost's native categorical support), the 3 numeric columns, and digit-position features of tenure,
MonthlyCharges and TotalCharges: every integer digit position (as many as the largest value has) and, for a column with
a fractional part, its first two decimal digits (13 digit columns).
Model: XGBoost (GPU), depth 3, lr 0.02, subsample 0.8, colsample_bytree 0.8, max_bin 16000, up to 20,000 trees with
early stopping (200 rounds) on the validation fold's AUC, seed 42. Test = mean of the fold models.
Usage: python train.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]
from common.data import TARGET, labels, read_test, read_train  # noqa: E402
from lib.runner import run_folds  # noqa: E402

warnings.simplefilter('ignore')

FOLDER = 'nalgirayergn_xgb'
TEST_FOLDS = 15
SEED = 42
CATS = ['gender', 'Partner', 'Dependents', 'SeniorCitizen', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
DIGIT_BASE_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

XGB_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.02,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_bin': 16000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': SEED,
    'early_stopping_rounds': 200,
    'device': 'cuda',
    'enable_categorical': True,
}


def coerce_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].dtype == 'O' or str(df[c].dtype).startswith('string'):
            s = df[c].astype(str).str.strip()
            s = s.replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
            df[c] = pd.to_numeric(s, errors='coerce')
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def add_digit_features(combined_df, base_cols, max_dec_digits=2):
    """Integer digit positions (from the largest absolute value of train + test) and, if any value has a fractional
    part, the first max_dec_digits decimal digits."""
    digit_cols = []
    for col in base_cols:
        if col not in combined_df.columns:
            continue
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        col_vals = combined_df[col].copy()
        fill_val = 0.0
        if col_vals.notna().any():
            fill_val = float(col_vals.median())
        col_vals = col_vals.fillna(fill_val)

        max_val = float(col_vals.abs().max())
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
        for i in range(max_int_digits):
            new_col = f'{col}_digit_pos_{i + 1}'
            combined_df[new_col] = ((col_vals // (10 ** i)) % 10).astype(int)
            digit_cols.append(new_col)

        if (col_vals % 1 != 0).any():
            for i in range(1, max_dec_digits + 1):
                new_col = f'{col}_digit_dec_{i}'
                combined_df[new_col] = ((col_vals * (10 ** i)).round().astype(int) % 10).astype(int)
                digit_cols.append(new_col)
    return combined_df, digit_cols


def main():
    train, test = read_train(), read_test()
    print('train:', train.shape, 'test:', test.shape, flush=True)
    train[TARGET] = labels(train[TARGET])

    train = coerce_numeric(train, NUMS)
    test = coerce_numeric(test, NUMS)
    for c in NUMS:
        train[c] = train[c].fillna(train[c].median() if train[c].notna().any() else 0.0)
        test[c] = test[c].fillna(test[c].median() if test[c].notna().any() else 0.0)

    combined_df = pd.concat([train, test], axis=0, ignore_index=True)
    for c in CATS:
        combined_df[c] = combined_df[c].astype(str).astype('category')
    combined_df, digit_cols = add_digit_features(combined_df, DIGIT_BASE_COLS, max_dec_digits=2)
    print(f'{len(digit_cols)} digit features: {digit_cols}', flush=True)

    train2 = combined_df.iloc[:len(train)].reset_index(drop=True)
    test2 = combined_df.iloc[len(train):].reset_index(drop=True)
    features = CATS + NUMS + digit_cols
    x, x_test = train2[features], test2[features]
    y = train[TARGET].to_numpy()
    print('n_features:', len(features), flush=True)

    def fit_fold(fold, tr_idx, va_idx, n_folds):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(x.iloc[tr_idx], y[tr_idx], eval_set=[(x.iloc[va_idx], y[va_idx])], verbose=500)
        print(f'fold {fold}: best_iteration={getattr(model, "best_iteration", None)}', flush=True)
        valid_pred = model.predict_proba(x.iloc[va_idx])[:, 1]
        test_pred = model.predict_proba(x_test)[:, 1]
        del model
        gc.collect()
        return valid_pred, test_pred

    run_folds(FOLDER, FOLDER, fit_fold, test_folds=TEST_FOLDS, y=y)


if __name__ == '__main__':
    main()
