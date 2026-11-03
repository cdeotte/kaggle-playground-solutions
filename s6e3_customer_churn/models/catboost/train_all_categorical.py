#!/usr/bin/env python
"""GPU CatBoost with every feature categorical except the 3 numerics, including original-value bins and first decimal
digits (pair catboost_all_categorical_orig_bins_d1).

Preprocessing: id is dropped; the 11 original rows with a blank TotalCharges are dropped; tenure, MonthlyCharges and
TotalCharges are float32; the 16 other columns are label-encoded over train + test + original.
Features (24, 21 of them CatBoost categoricals):
  - tenure, MonthlyCharges, TotalCharges (numeric)
  - the 16 label-encoded columns
  - original-value bins of the 3 numerics: the sorted unique values of the original data split into
    min(100 / 512 / 2048, number of unique values) equal-size chunks for tenure / MonthlyCharges / TotalCharges; a value
    seen in the original data gets its chunk, any other value the chunk whose range contains it, 0 when below the
    first chunk and the last chunk otherwise
  - the first decimal digit of MonthlyCharges and TotalCharges
CatBoost (GPU): depth 6, lr 0.03, l2 8, up to 50,000 iterations with early stopping (200) on the validation fold
(use_best_model=True), seed 42. Test = mean of the fold models.

Usage: python train_all_categorical.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.catboost.lib.cv import run_folds  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'catboost_all_categorical_orig_bins_d1'
SEED = 42
TEST_FOLDS = 15
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
NUMERIC_BIN_CARDINALITY = {'tenure': 100, 'MonthlyCharges': 256 * 2, 'TotalCharges': 1024 * 2}
DECIMAL_DIGIT_COLS = ['MonthlyCharges', 'TotalCharges']

CB_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 50000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 8.0,
    'random_seed': SEED,
    'early_stopping_rounds': 200,
    'use_best_model': True,
    'verbose': 500,
    'task_type': 'GPU',
}


def reduce_mem_float32(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def first_decimal_digit(s: pd.Series) -> pd.Series:
    arr = s.to_numpy(dtype=np.float32, copy=False)
    out = np.full(len(arr), -1, dtype=np.int32)
    for i, v in enumerate(arr):
        if pd.isna(v):
            out[i] = -1
        else:
            out[i] = int(np.floor(np.abs(v) * 10.0 + 1e-6)) % 10
    return pd.Series(out, index=s.index, dtype='int32')


def make_apply_bins_fn(value_to_bin, edges, col_name):
    interval_starts = np.array([a for a, b in edges], dtype=np.float32)
    interval_ends = np.array([b for a, b in edges], dtype=np.float32)

    def apply_bins(s):
        arr = s.to_numpy(dtype=np.float32, copy=False)
        out = np.full(len(arr), -1, dtype=np.int32)
        for i, v in enumerate(arr):
            if pd.isna(v):
                out[i] = -1
            else:
                key = float(v)
                if key in value_to_bin:
                    out[i] = value_to_bin[key]
                else:
                    placed = False
                    for b in range(len(interval_starts)):
                        if interval_starts[b] <= v <= interval_ends[b]:
                            out[i] = b
                            placed = True
                            break
                    if not placed:
                        out[i] = 0 if v < interval_starts[0] else len(interval_starts) - 1
        return pd.Series(out, index=s.index, name=f'{col_name}_bin', dtype='int32')

    return apply_bins


def preprocess():
    train_df, test_df, orig_df = read_train(), read_test(), read_original()
    train_df = train_df.drop(columns=[ID])
    test_df = test_df.drop(columns=[ID])
    for df in (train_df, orig_df):
        df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0}).astype('int8')
    orig_df = orig_df.loc[orig_df['TotalCharges'].notna()].copy()
    orig_df['TotalCharges'] = orig_df['TotalCharges'].astype('float32')
    train_df = reduce_mem_float32(train_df, NUMERIC_COLS)
    test_df = reduce_mem_float32(test_df, NUMERIC_COLS)
    orig_df = reduce_mem_float32(orig_df, NUMERIC_COLS)

    feature_cols = [c for c in train_df.columns if c != TARGET]
    cat_cols = [c for c in feature_cols if c not in NUMERIC_COLS]
    for col in cat_cols:
        combined = pd.concat(
            [train_df[col].astype(str), test_df[col].astype(str), orig_df[col].astype(str)],
            axis=0, ignore_index=True,
        ).fillna('__MISSING__')
        le = LabelEncoder()
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        test_df[col] = le.transform(test_df[col].astype(str).fillna('__MISSING__')).astype('int32')
        orig_df[col] = le.transform(orig_df[col].astype(str).fillna('__MISSING__')).astype('int32')

    # bins of the original data's unique values
    for col in NUMERIC_COLS:
        uniq = np.sort(orig_df[col].dropna().unique())
        chunks = np.array_split(uniq, min(NUMERIC_BIN_CARDINALITY[col], len(uniq)))
        value_to_bin = {}
        edges = []
        for bin_id, arr in enumerate(chunks):
            for v in arr:
                value_to_bin[float(v)] = int(bin_id)
            edges.append((float(arr[0]), float(arr[-1])))
        apply_bins = make_apply_bins_fn(value_to_bin, edges, col)
        train_df[f'{col}_bin'] = apply_bins(train_df[col])
        test_df[f'{col}_bin'] = apply_bins(test_df[col])
        orig_df[f'{col}_bin'] = apply_bins(orig_df[col])

    for col in DECIMAL_DIGIT_COLS:
        train_df[f'{col}_d1'] = first_decimal_digit(train_df[col])
        test_df[f'{col}_d1'] = first_decimal_digit(test_df[col])
        orig_df[f'{col}_d1'] = first_decimal_digit(orig_df[col])

    features = [c for c in train_df.columns if c != TARGET]
    return train_df, test_df, features


def main():
    train_df, test_df, features = preprocess()
    cat_feature_names = [c for c in features if c not in NUMERIC_COLS]
    print(f'features: {len(features)} ({len(cat_feature_names)} categorical)', flush=True)
    X = train_df[features].copy()
    y = train_df[TARGET].astype(np.int8).values
    X_test = test_df[features].copy()
    for c in cat_feature_names:
        X[c] = X[c].astype('int32')
        X_test[c] = X_test[c].astype('int32')
    for c in NUMERIC_COLS:
        X[c] = X[c].astype('float32')
        X_test[c] = X_test[c].astype('float32')

    def fit_fold(fold, tr_idx, va_idx, seed):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(X_tr, y[tr_idx], eval_set=(X_va, y[va_idx]), cat_features=cat_feature_names,
                  verbose=CB_PARAMS['verbose'])
        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_test)[:, 1].astype(np.float32)
        print(f'fold {fold}: best iteration {model.get_best_iteration()}', flush=True)
        del X_tr, X_va, model
        gc.collect()
        return p_va, p_te

    oof, _ = run_folds(fit_fold, y, len(X_test), name=PAIR)
    _, test_pred = run_folds(fit_fold, y, len(X_test), n_folds=TEST_FOLDS, seed=SEED, name=PAIR)
    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
