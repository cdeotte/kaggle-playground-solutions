#!/usr/bin/env python
"""GPU XGBoost with nested target statistics and original-data churn rates (pair xgb_te_stats_original_proba).

Features (missing TotalCharges get the frame's median): the 3 numerics, TotalCharges / (tenure + 1),
tenure x MonthlyCharges, the frequency of each numeric over train + original + test rows, and the original churn rate
of every value of the 16 categoricals and the 3 numerics (unseen: 0.5).
Per fold, for the 16 categoricals and the 3 numerics as strings: std / min / max of the target per value (out-of-fold
over an inner StratifiedKFold(10, shuffle, random_state 42) for the training rows, all training rows for the
validation / test rows; unseen: 0) and sklearn TargetEncoder means (cv 10, shuffle, smooth auto, random_state 42); the
string columns themselves are dropped.
Model: depth 6, lr 0.03, subsample / colsample_bytree 0.8, min_child_weight 5, alpha 0.1, lambda 1, gamma 0.05, up to
50,000 rounds, early stopping 500 on the validation fold, random_state 42.
Usage: python train_te_stats_original_proba.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, labels, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings('ignore')

PAIR = 'xgb_te_stats_original_proba'
N_FOLDS = 5
TEST_FOLDS = 15
INNER_FOLDS = 10
SEED = 42
STATS = ['std', 'min', 'max']

CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'gamma': 0.05,
    'early_stopping_rounds': 500,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda',
    'tree_method': 'hist',
    'verbosity': 0,
    'random_state': SEED,
}


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    orig['TotalCharges'] = orig['TotalCharges'].fillna(orig['TotalCharges'].median())
    return train, test, orig


def build_features(train, test, orig):
    new_nums, num_as_cat = [], []

    for df in (train, test, orig):
        for c in CATS:
            df[c] = df[c].astype(str).fillna('missing').str.strip()
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    for df in (train, test, orig):
        df['AvgChargesPerMonth'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        df['tenure_x_MonthlyCharges'] = (df['tenure'] * df['MonthlyCharges']).astype('float32')
    new_nums.extend(['AvgChargesPerMonth', 'tenure_x_MonthlyCharges'])

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in (train, test, orig):
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        new_nums.append(f'FREQ_{col}')

    for col in NUMS:
        new_col = f'CAT_{col}'
        num_as_cat.append(new_col)
        for df in (train, test, orig):
            df[new_col] = df[col].astype(str)

    for col in CATS + NUMS:
        new_col = f'ORIG_proba_{col}'
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        tmp.name = new_col
        train = train.merge(tmp, on=col, how='left')
        test = test.merge(tmp, on=col, how='left')
        new_nums.append(new_col)
        train[new_col] = train[new_col].fillna(0.5).astype('float32')
        test[new_col] = test[new_col].fillna(0.5).astype('float32')

    features = list(dict.fromkeys(NUMS + CATS + new_nums + num_as_cat))
    te_columns = num_as_cat + CATS
    to_remove = num_as_cat + CATS
    print(f'{len(features)} features before encoding, {len(te_columns)} encoded columns', flush=True)
    return train, test, features, te_columns, to_remove, num_as_cat


def add_nested_te(X_tr, y_tr, X_val, X_te, features, te_columns, seed, num_as_cat, to_remove):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for in_tr, in_va in skf_inner.split(X_tr, y_tr):
        X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
        X_va2 = X_tr.loc[in_va, features].copy()
        for col in te_columns:
            tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
            X_va2 = X_va2.merge(tmp, on=col, how='left')
            for c in tmp.columns:
                X_tr.loc[in_va, c] = X_va2[c].values.astype('float32')

    for col in te_columns:
        tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
        tmp.columns = [f'TE1_{col}_{s}' for s in STATS]
        tmp = tmp.astype('float32')
        X_val = X_val.merge(tmp, on=col, how='left')
        X_te = X_te.merge(tmp, on=col, how='left')
        for c in tmp.columns:
            for df_ in (X_tr, X_val, X_te):
                df_[c] = df_[c].fillna(0).astype('float32')

    te_mean_cols = [f'TE_{col}' for col in te_columns]
    te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=seed)
    X_tr[te_mean_cols] = te.fit_transform(X_tr[te_columns], y_tr).astype('float32')
    X_val[te_mean_cols] = te.transform(X_val[te_columns]).astype('float32')
    X_te[te_mean_cols] = te.transform(X_te[te_columns]).astype('float32')

    for df_ in (X_tr, X_val, X_te):
        for c in [c for c in (CATS + num_as_cat) if c in df_.columns]:
            df_[c] = df_[c].astype(str).astype('category')
        drop_cols = [c for c in to_remove if c in df_.columns]
        if drop_cols:
            df_.drop(columns=drop_cols, inplace=True)

    X_tr.drop(columns=[TARGET], inplace=True)
    cols = list(X_tr.columns)
    return X_tr, X_val[cols].copy(), X_te[cols].copy(), cols


def run_cv(train, test, features, te_columns, to_remove, num_as_cat, n_folds):
    oof = np.zeros(len(train), dtype=np.float64)
    pred = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET]), 1):
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, TARGET].values.copy()
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, TARGET].values.copy()
        X_te = test[features].reset_index(drop=True).copy()

        X_tr, X_val, X_te, cols = add_nested_te(X_tr, y_tr, X_val, X_te, features=features, te_columns=te_columns,
                                                seed=SEED, num_as_cat=num_as_cat, to_remove=to_remove)
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=500)
        oof_p = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = oof_p
        pred += model.predict_proba(X_te[cols])[:, 1]
        print(f'{n_folds} folds, fold {fold}: AUC {roc_auc_score(y_val, oof_p):.6f}', flush=True)

        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    pred /= n_folds
    print(f'{n_folds} folds: OOF AUC {roc_auc_score(train[TARGET].values, oof):.6f}', flush=True)
    return oof.astype(np.float32), pred.astype(np.float32)


def main():
    train, test, orig = load_data()
    train, test, features, te_columns, to_remove, num_as_cat = build_features(train, test, orig)
    oof, _ = run_cv(train, test, features, te_columns, to_remove, num_as_cat, N_FOLDS)
    _, test_pred = run_cv(train, test, features, te_columns, to_remove, num_as_cat, TEST_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
