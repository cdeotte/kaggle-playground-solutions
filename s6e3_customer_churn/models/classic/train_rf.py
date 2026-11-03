#!/usr/bin/env python
"""cuML random forest on snap / decimal-digit features with per-fold target encodings (pair rf_snap_digits_te).

Features (train and test rows; the original rows only provide the snap values, after dropping its 11 rows with a blank
TotalCharges): all raw columns including `id`; MonthlyCharges and TotalCharges snapped to the nearest original value
(MC_snap, TC_snap) and value - snap; for MonthlyCharges and TotalCharges the first and second decimal digits and a
round-value flag (fractional part < 0.005); TotalCharges - tenure x MonthlyCharges; MonthlyCharges / TotalCharges;
the number of "Yes" among 7 service columns; the bigram Contract x InternetService.
Per fold, every string column (15 categoricals + the bigram) is replaced by its cuML TargetEncoder value (5 interleaved
inner folds, smoothing 20; fitted on the fold's training rows).
Model: cuML RandomForestClassifier, 1,500 trees, max_depth 16, max_features 0.3, min_samples_leaf 15,
random_state 42, n_streams 1. Test = mean of the 5 fold models.
Usage: python train_rf.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, folds, labels, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'rf_snap_digits_te'
FOLDER = 'classic'
SEED = 42
N_FOLDS = 5


def engineer_features(df, orig_ibm_df):
    df = df.copy()

    # snap to the nearest original value
    orig_mc = np.sort(orig_ibm_df['MonthlyCharges'].dropna().unique())
    orig_tc = np.sort(pd.to_numeric(orig_ibm_df['TotalCharges'], errors='coerce').dropna().unique())
    df['MC_snap'] = df['MonthlyCharges'].apply(lambda x: orig_mc[np.abs(orig_mc - x).argmin()])
    df['TC_snap'] = df['TotalCharges'].apply(lambda x: orig_tc[np.abs(orig_tc - x).argmin()])
    df['MC_snap_diff'] = df['MonthlyCharges'] - df['MC_snap']
    df['TC_snap_diff'] = df['TotalCharges'] - df['TC_snap']

    # decimal digits
    for col in ['MonthlyCharges', 'TotalCharges']:
        frac = df[col] - np.floor(df[col])
        df[f'{col}_d1'] = np.floor(frac * 10).astype(int)
        df[f'{col}_d2'] = (np.floor(frac * 100) % 10).astype(int)
        df[f'{col}_is_round'] = (frac < 0.005).astype(int)

    # arithmetic interactions
    df['TC_deviation'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
    df['MC_to_TC_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-9)

    # service count
    svc_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    df['service_yes_count'] = sum((df[c] == 'Yes').astype(int) for c in svc_cols)

    # bigram
    df['bi_Contract_Internet'] = df['Contract'] + '__' + df['InternetService']
    return df


def main():
    train, test = read_train(), read_test()
    orig = read_original()
    orig = orig.dropna(subset=['TotalCharges']).reset_index(drop=True)
    train[TARGET] = labels(train[TARGET]).astype(int)

    train_df = engineer_features(train, orig)
    test_df = engineer_features(test, orig)
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object' and c != TARGET]

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros((len(test_df), N_FOLDS))
    fold_id = folds(train_df[TARGET].values)
    for fold in range(N_FOLDS):
        train_idx, val_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]

        X_tr = train_df.iloc[train_idx].drop(columns=[TARGET]).copy()
        y_tr = train_df[TARGET].iloc[train_idx].copy()
        X_va = train_df.iloc[val_idx].drop(columns=[TARGET]).copy()
        y_va = train_df[TARGET].iloc[val_idx].copy()
        X_te = test_df.drop(columns=[TARGET], errors='ignore').copy()

        for col in cat_cols:
            encoder = TargetEncoder(n_folds=5, smooth=20, split_method='interleaved')
            X_tr[f'{col}_TE'] = encoder.fit_transform(X_tr[col], y_tr).to_numpy().ravel()
            X_va[f'{col}_TE'] = encoder.transform(X_va[col]).to_numpy().ravel()
            X_te[f'{col}_TE'] = encoder.transform(X_te[col]).to_numpy().ravel()
            X_tr.drop(columns=[col], inplace=True)
            X_va.drop(columns=[col], inplace=True)
            X_te.drop(columns=[col], inplace=True)

        rf_model = RandomForestClassifier(
            n_estimators=1500,
            max_depth=16,
            max_features=0.3,
            min_samples_leaf=15,
            random_state=SEED,
            n_streams=1,
        )
        rf_model.fit(X_tr.astype('float32').to_numpy(), y_tr.astype('int32').to_numpy())

        oof_preds[val_idx] = rf_model.predict_proba(X_va.astype('float32').to_numpy())[:, 1]
        test_preds[:, fold] = rf_model.predict_proba(X_te.astype('float32').to_numpy())[:, 1]
        print(f'fold {fold}: AUC {roc_auc_score(y_va, oof_preds[val_idx]):.6f}', flush=True)

    print(f'{PAIR}: OOF AUC {roc_auc_score(train_df[TARGET], oof_preds):.6f}', flush=True)
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds.mean(axis=1))


if __name__ == '__main__':
    main()
