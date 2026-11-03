#!/usr/bin/env python
"""Compact TabM-mini on the raw columns plus fold-safe snapped numeric proxies and rare flags
(pair tabm_mini_snapped_proxies_rare_flags).

Features (25 columns, competition rows only; numeric NaN -> train median): the 16 categoricals as strings; for tenure,
MonthlyCharges and TotalCharges a snapper fitted on the fold's training rows maps every value seen fewer than 25 times
to the nearest frequent value: the snapped value (`__cat`) and the rare flag (`__is_rare`) as categories; the 3 numerics
standardized on the fold's training rows.
Model: pytabkit TabM_D_Classifier, tabm-mini, k 16, 2 blocks x 256, piecewise-linear numeric embeddings (d 8),
dropout 0.2, lr 3e-4, batch 512, 5 epochs (best epoch on the validation fold's AUC), random_state 100.
Folds: the 5 folds of common/data.py. Test = mean of the 5 fold models.
Usage: python train_mini_snapped_proxies_rare_flags.py"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import TabM_D_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import N_FOLDS, TARGET, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "tabm"
PAIR = "tabm_mini_snapped_proxies_rare_flags"
SEED = 42

BASE_FEATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_BASE = [c for c in BASE_FEATS if c not in NUMS]

RARE_MIN_COUNT = 25

TABM_PARAMS = {
    "device": "cuda",
    "val_metric_name": "1-auc_ovr",
    "random_state": 100,
    "verbosity": 2,
    "arch_type": "tabm-mini",
    "tabm_k": 16,
    "num_emb_type": "pwl",
    "d_embedding": 8,
    "batch_size": 512,
    "lr": 3e-4,
    "n_epochs": 5,
    "dropout": 0.20,
    "d_block": 256,
    "n_blocks": 2,
}


def clean_total_charges(series):
    """Strip spaces, blank -> NaN, float32."""
    s = series.astype(str).str.strip()
    s = s.replace("", np.nan)
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def preprocess_base(df, is_train):
    df = df.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype(np.float32)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype(np.float32)
    df["TotalCharges"] = clean_total_charges(df["TotalCharges"])
    for c in CAT_BASE:
        df[c] = df[c].astype(str).fillna("nan")
    if is_train:
        y = df[TARGET]
        if y.dtype == object:
            y = y.map({"Yes": 1, "No": 0})
        df[TARGET] = y.astype(np.int8)
    return df


def build_numeric_snapper(train_series, rare_min_count):
    """From the training fold only: values seen >= rare_min_count times are frequent; every other value is snapped to
    the nearest frequent value (ties -> the larger one) and flagged rare."""
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)

    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)

    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any):
        x = pd.to_numeric(series_any, errors="coerce").astype(np.float32).values
        is_nan = np.isnan(x)

        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if float(v) in frequent_set else 1

        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.searchsorted(frequent_vals, v)
                pos = np.clip(pos, 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                right = pos
                left_vals = frequent_vals[left]
                right_vals = frequent_vals[right]
                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                nearest = np.where(choose_right, right_vals, left_vals)
                x_snapped[idx_snap] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def add_fold_safe_proxy_features(tr_df, va_df, te_df):
    """{num}__cat (snapped value) and {num}__is_rare, both as strings, from snappers fitted on the training fold."""
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    te_df = te_df.copy()
    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr_df[col])
        va_snap, va_israre = snapper(va_df[col])
        te_snap, te_israre = snapper(te_df[col])

        tr_df[f"{col}__cat"] = pd.Series(tr_snap).astype(str).values
        va_df[f"{col}__cat"] = pd.Series(va_snap).astype(str).values
        te_df[f"{col}__cat"] = pd.Series(te_snap).astype(str).values

        tr_df[f"{col}__is_rare"] = pd.Series(tr_israre).astype(str).values
        va_df[f"{col}__is_rare"] = pd.Series(va_israre).astype(str).values
        te_df[f"{col}__is_rare"] = pd.Series(te_israre).astype(str).values
    return tr_df, va_df, te_df


def build_tabm_matrices(tr_df, va_df, te_df):
    """Numerics standardized on the training fold; categorical columns cast to category."""
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    te_df = te_df.copy()

    CAT_ALL = CAT_BASE + [f"{c}__cat" for c in NUMS] + [f"{c}__is_rare" for c in NUMS]
    NUM_ALL = NUMS
    FEATURES = CAT_ALL + NUM_ALL

    scaler = StandardScaler()
    tr_df[NUM_ALL] = scaler.fit_transform(tr_df[NUM_ALL].astype(np.float32)).astype(np.float32)
    va_df[NUM_ALL] = scaler.transform(va_df[NUM_ALL].astype(np.float32)).astype(np.float32)
    te_df[NUM_ALL] = scaler.transform(te_df[NUM_ALL].astype(np.float32)).astype(np.float32)

    for c in CAT_ALL:
        tr_df[c] = tr_df[c].astype("category")
        va_df[c] = va_df[c].astype("category")
        te_df[c] = te_df[c].astype("category")

    return tr_df[FEATURES].copy(), va_df[FEATURES].copy(), te_df[FEATURES].copy(), CAT_ALL, NUM_ALL


def main():
    np.random.seed(SEED)

    train = preprocess_base(read_train(), is_train=True)
    test = preprocess_base(read_test(), is_train=False)
    print(f"train {train.shape}  test {test.shape}")

    for c in NUMS:
        med = train[c].median()
        train[c] = train[c].fillna(med).astype(np.float32)
        test[c] = test[c].fillna(med).astype(np.float32)

    y = train[TARGET].values.astype(np.int8)
    fold_id = folds(y)
    oof = np.zeros(len(train), dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)

    for fold in range(N_FOLDS):
        tr_idx, va_idx = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        te_df = test.copy()

        tr_df, va_df, te_df = add_fold_safe_proxy_features(tr_df, va_df, te_df)
        X_tr, X_va, X_te, CAT_ALL, NUM_ALL = build_tabm_matrices(tr_df, va_df, te_df)
        y_tr = tr_df[TARGET].values.astype(np.int8)
        y_va = va_df[TARGET].values.astype(np.int8)
        print(f"fold {fold + 1}/{N_FOLDS}: X_tr {X_tr.shape}, n_cat {len(CAT_ALL)}, n_num {len(NUM_ALL)}", flush=True)

        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_tr, y_tr, X_va, y_va, cat_col_names=CAT_ALL)

        p_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
        p_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
        oof[va_idx] = p_va
        pred_test += p_te / N_FOLDS
        print(f"fold {fold + 1} AUC = {roc_auc_score(y_va, p_va):.6f}", flush=True)

        del model, tr_df, va_df, te_df, X_tr, X_va, X_te, y_tr, y_va
        gc.collect()

    print(f"OOF AUC: {roc_auc_score(y, oof):.6f}")
    save_pair(model_dir(FOLDER), PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
