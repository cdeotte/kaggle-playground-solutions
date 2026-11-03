#!/usr/bin/env python
"""GPU XGBoost with native categoricals plus a TabPFN prediction learned from the original data
(pair xgb_native_cat_tabpfn_original).

Needs cache/tabpfn_original_pred.npy from build_tabpfn_original.py (run it first): the churn probability of every
train / test row from TabPFN models fitted on the original rows only.
Features (20): the 19 raw columns (id dropped; the 15 string columns as pandas categoricals with sorted categories
shared by train and test; missing TotalCharges -> 0) and the TabPFN prediction.
Model: native categoricals, depth 3, lr 0.1, min_child_weight 5, subsample / colsample_bytree 0.85, up to 100,000
rounds, early stopping 200 on the validation fold, random_state 42.
Usage: python train_tabpfn_original_feature.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, cache_dir, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'xgb_native_cat_tabpfn_original'
TABPFN_FILE = 'tabpfn_original_pred.npy'
N_SPLITS = 5
TEST_FOLDS = 15
RANDOM_STATE = 42
DROP_COLS = ['customerID', ID]

XGB_PARAMS = dict(
    n_estimators=100_000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=200,
    device='cuda',
)


def preprocess_together(X_tr, X_te):
    """Train + test together, so every string column gets the same categories in both."""
    X_tr = X_tr.drop(columns=[c for c in DROP_COLS if c in X_tr.columns])
    X_te = X_te.drop(columns=[c for c in DROP_COLS if c in X_te.columns])
    n_tr = len(X_tr)
    X_all = pd.concat([X_tr, X_te], axis=0, ignore_index=True)

    X_all['TotalCharges'] = pd.to_numeric(X_all['TotalCharges'], errors='coerce').fillna(0.0).astype('float32')
    for c in ['tenure', 'MonthlyCharges']:
        X_all[c] = pd.to_numeric(X_all[c], errors='coerce')

    cat_cols = set(X_all.select_dtypes(include=['object']).columns.tolist())
    cat_cols |= set(X_all.select_dtypes(include=['category']).columns.tolist())
    for c in sorted(cat_cols):
        s = X_all[c]
        if str(s.dtype) == 'category':
            s = s.astype('object')
        non_null = s[~pd.isna(s)]
        cats = pd.Index(non_null.map(lambda v: str(v)).unique(), dtype='object').sort_values()
        X_all[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(pd.CategoricalDtype(categories=cats,
                                                                                                 ordered=False))
    return X_all.iloc[:n_tr].reset_index(drop=True), X_all.iloc[n_tr:].reset_index(drop=True)


def run_cv(X_train, y, X_test, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train), dtype=np.float32)
    test_pred_sum = np.zeros(len(X_test), dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=500)
        oof[val_idx] = model.predict_proba(X_val)[:, 1].astype(np.float32)
        test_pred_sum += model.predict_proba(X_test)[:, 1].astype(np.float32)
        print(f'{n_splits} folds, fold {fold}: AUC {roc_auc_score(y_val, oof[val_idx]):.6f}', flush=True)
    print(f'{n_splits} folds: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    return oof, (test_pred_sum / n_splits).astype(np.float32)


def main():
    train_df = read_train()
    test_df = read_test()
    tabpfn_pred = np.load(cache_dir('xgboost') / TABPFN_FILE)
    assert len(tabpfn_pred) == len(train_df) + len(test_df), 'run build_tabpfn_original.py first'
    train_df['tabpfn_original'] = tabpfn_pred[:len(train_df)]
    test_df['tabpfn_original'] = tabpfn_pred[len(train_df):]

    y = labels(train_df[TARGET]).astype(np.int32)
    X_train, X_test = preprocess_together(train_df.drop(columns=[TARGET]), test_df)
    print(f'{X_train.shape[1]} features', flush=True)

    oof, _ = run_cv(X_train, y, X_test, N_SPLITS)
    _, test_pred = run_cv(X_train, y, X_test, TEST_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
