#!/usr/bin/env python
"""GPU XGBoost stumps on snap, decimal-string and radix features (pair xgb_snap_radix_original_priors).

Reads the snap-feature table of the train + test rows written by ../snap_features (combo_b.csv); run that folder
first. Features (81):
- the 16 categoricals as native XGBoost categories, tenure, MonthlyCharges, TotalCharges;
- the decimal strings of the two charges (tc, tc0, tc1, mc, mc0, mc1 as categories; tc_len, mc_len);
- the snap values of MonthlyCharges / TotalCharges (nearest value present in the original data), snap - value, and
  the original churn rate of each snapped value (orig_MonthlyCharges, orig_TotalCharges); missing charges / snaps get
  the train median;
- 48 radix keys: round(100 x snap) + 100,000 x (category code) for both snaps and tenure + 100,000 x (category code),
  each crossed with the 16 categoricals (codes over train + test).
Model: depth 1, lr 0.1, max_bin 16,000, min_child_weight 5, subsample / colsample_bytree 0.85, up to 1,000,000 rounds,
early stopping 300 on the validation fold, random_state 42.
Usage: python train_snap_radix.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import cache_dir, target  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'xgb_snap_radix_original_priors'
RANDOM_STATE = 42
N_FOLDS = 5
N_THREADS = 32
RADIX = 100_000

LOW_CARD_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod',
]

SNAP_DTYPES = {
    'TotalCharges': 'string',
    'MonthlyCharges': 'string',
    'tc': 'object',
    'tc0': 'object',
    'tc1': 'object',
    'mc': 'object',
    'mc0': 'object',
    'mc1': 'object',
    'MonthlyCharges_snap': 'string',
    'TotalCharges_snap': 'string',
}

XGB_PARAMS = dict(
    n_estimators=1_000_000,
    learning_rate=0.1,
    max_depth=1,
    min_child_weight=5,
    max_bin=16_000,
    subsample=0.85,
    colsample_bytree=0.85,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=N_THREADS,
    early_stopping_rounds=300,
    device='cuda',
)


def build_features(n_train):
    combo = pd.read_csv(cache_dir('snap_features') / 'combo_b.csv', dtype=SNAP_DTYPES)

    for c in ['mc', 'mc0', 'mc1', 'tc', 'tc0', 'tc1']:
        combo[c] = combo[c].astype('string').fillna('NA')

    charge_cols = ['MonthlyCharges', 'TotalCharges', 'MonthlyCharges_snap', 'TotalCharges_snap']
    for col in charge_cols:
        combo[col] = pd.to_numeric(combo[col], errors='coerce').astype('float32')
    for col in charge_cols:
        med = float(np.nanmedian(combo.loc[:n_train - 1, col].to_numpy()))
        combo[col] = combo[col].fillna(med).astype('float32')
    combo['MonthlyCharges_snap_diff'] = combo['MonthlyCharges_snap_diff'].astype('float32')
    combo['TotalCharges_snap_diff'] = combo['TotalCharges_snap_diff'].astype('float32')

    # radix keys
    combo['MonthlyCharges_snap_i'] = np.round(combo['MonthlyCharges_snap'] * 100.0).astype('int32')
    combo['TotalCharges_snap_i'] = np.round(combo['TotalCharges_snap'] * 100.0).astype('int32')
    combo['tenure_i'] = pd.to_numeric(combo['tenure'], errors='coerce').fillna(0).astype('int32')
    combo['tenure_i'] = combo['tenure_i'].clip(lower=0, upper=RADIX - 1).astype('int32')

    lowcard_codes = {}
    for c in LOW_CARD_COLS:
        cat = pd.Categorical(combo[c].astype('string').fillna('NA'))
        lowcard_codes[c] = cat.codes.astype('int32')
        combo[c] = cat

    radix_cols = []
    for snap_i, prefix in [('MonthlyCharges_snap_i', 'mc'), ('TotalCharges_snap_i', 'tc')]:
        for c in LOW_CARD_COLS:
            newc = f'{prefix}_radix_{c}'
            combo[newc] = (combo[snap_i] + lowcard_codes[c] * RADIX).astype('int32')
            radix_cols.append(newc)
    tenure_radix_cols = []
    for c in LOW_CARD_COLS:
        newc = f'tenure_radix_{c}'
        combo[newc] = (combo['tenure_i'] + lowcard_codes[c] * RADIX).astype('int32')
        tenure_radix_cols.append(newc)

    base_feats = LOW_CARD_COLS + ['tenure', 'MonthlyCharges', 'TotalCharges']
    engineered = [
        'tc', 'tc_len', 'tc0', 'tc1',
        'mc', 'mc0', 'mc1', 'mc_len',
        'MonthlyCharges_snap', 'MonthlyCharges_snap_diff',
        'TotalCharges_snap', 'TotalCharges_snap_diff',
        'orig_MonthlyCharges', 'orig_TotalCharges',
    ]
    all_feats = base_feats + engineered + radix_cols + tenure_radix_cols

    for c in ['tc', 'tc0', 'tc1', 'mc', 'mc0', 'mc1']:
        combo[c] = combo[c].astype('category')

    X_all = combo.iloc[:n_train][all_feats].copy()
    T_all = combo.iloc[n_train:][all_feats].copy()
    # XGBoost needs object-backed (not pandas string) categories
    for c in X_all.select_dtypes(include=['category']).columns.tolist():
        X_all[c] = X_all[c].astype('object').astype('category')
        T_all[c] = T_all[c].astype('object').astype('category')
    print(f'{len(all_feats)} features: X {X_all.shape}, test {T_all.shape}', flush=True)
    return X_all, T_all


def main():
    y = target().astype(np.int32)
    X_all, T_all = build_features(len(y))

    oof = np.zeros(len(y), dtype=np.float64)
    pred_test = np.zeros(len(T_all), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), start=1):
        X_tr, y_tr = X_all.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_all.iloc[va_idx], y[va_idx]
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        pred_test += model.predict_proba(T_all)[:, 1] / N_FOLDS
        print(f'fold {fold}: AUC {roc_auc_score(y_va, oof[va_idx]):.6f}, best_iteration '
              f'{getattr(model, "best_iteration", None)}', flush=True)
        del model
        gc.collect()

    print(f'{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}', flush=True)
    save_pair(HERE, PAIR, oof, pred_test)


if __name__ == '__main__':
    main()
