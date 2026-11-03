#!/usr/bin/env python
"""TabPFN fitted on the original rows, predicting every train and test row (input of train_tabpfn_original_feature.py).

The 7,032 original rows that have a TotalCharges value (19 feature columns; the string columns become pandas
categories over original + train + test rows) are split by StratifiedKFold(5, shuffle=True, random_state=42). Per fold
a TabPFNClassifier (32 estimators, random_state 42 + fold number) is fitted on 4 folds of the original rows, scored on
the 5th, and predicts all train + test rows. Writes cache/tabpfn_original_pred.npy: the mean of the 5 fold predictions
for the 594,194 train rows followed by the 254,655 test rows. Only original labels are used.
Usage: python build_tabpfn_original.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import TARGET, cache_dir, labels, read_original, read_test, read_train  # noqa: E402

OUT_FILE = 'tabpfn_original_pred.npy'
N_SPLITS = 5
RANDOM_STATE = 42
TABPFN_N_ESTIMATORS = 32
TABPFN_DEVICE = 'auto'
TABPFN_INFERENCE_PRECISION = 'auto'


def object_to_category_global(df):
    """Object columns -> pandas categories (string values, missing -> '__MISSING__') on the stacked frame."""
    out = df.copy()
    for c in out.select_dtypes(include=['object']).columns.tolist():
        s = out[c].astype('string')
        if s.isna().any():
            s = s.fillna('__MISSING__')
        out[c] = s.astype('category')
    return out


def main():
    from tabpfn import TabPFNClassifier

    orig = read_original()
    orig = orig.loc[orig['TotalCharges'].notna()].copy()
    orig['TotalCharges'] = orig['TotalCharges'].astype('float32')
    orig_X_raw = orig.drop(columns=[TARGET])
    orig_y = labels(orig[TARGET]).astype(np.int32)

    ext_tr = read_train()
    ext_te = read_test()
    for df in (ext_tr, ext_te):
        df['TotalCharges'] = df['TotalCharges'].astype('float32')
    ext_all_raw = pd.concat([ext_tr, ext_te], axis=0, ignore_index=True)[orig_X_raw.columns].copy()

    n_orig = len(orig_X_raw)
    all_X = object_to_category_global(pd.concat([orig_X_raw, ext_all_raw], axis=0, ignore_index=True))
    X = all_X.iloc[:n_orig].reset_index(drop=True)
    ext_all = all_X.iloc[n_orig:].reset_index(drop=True)
    print(f'original rows {X.shape}, positives {int(orig_y.sum())}; train + test rows {ext_all.shape}', flush=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype=np.float32)
    ext_pred_by_fold = np.zeros((len(ext_all), N_SPLITS), dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, orig_y), 1):
        clf = TabPFNClassifier(
            n_estimators=TABPFN_N_ESTIMATORS,
            device=TABPFN_DEVICE,
            random_state=RANDOM_STATE + fold,
            inference_precision=TABPFN_INFERENCE_PRECISION,
        )
        clf.fit(X.iloc[trn_idx], orig_y[trn_idx])
        oof[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1].astype(np.float32)
        print(f'fold {fold}: original AUC {roc_auc_score(orig_y[val_idx], oof[val_idx]):.6f}', flush=True)
        ext_pred_by_fold[:, fold - 1] = clf.predict_proba(ext_all)[:, 1].astype(np.float32)

    print(f'original OOF AUC {roc_auc_score(orig_y, oof):.6f}', flush=True)
    ext_pred_mean = ext_pred_by_fold.mean(axis=1).astype(np.float32)
    np.save(cache_dir('xgboost') / OUT_FILE, ext_pred_mean)
    print(f'wrote {OUT_FILE} {ext_pred_mean.shape}', flush=True)


if __name__ == '__main__':
    main()
