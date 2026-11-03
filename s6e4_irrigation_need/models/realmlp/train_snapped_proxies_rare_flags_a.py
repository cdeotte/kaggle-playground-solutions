#!/usr/bin/env python
"""Compact RealMLP on the raw columns plus fold-safe snapped numeric proxies and rare flags
(pair realmlp_snapped_proxies_rare_flags_a).

Features (25 columns, competition rows only): the 8 categoricals and 11 numerics (NaN -> the fold's training median);
for Rainfall_mm, Previous_Irrigation_mm and Soil_Moisture, a snapper fitted on the fold's training rows maps every
value seen fewer than 25 times to the nearest frequent value: the snapped value as a category (`__cat`) and the rare
flag as a category (`__is_rare`).
Model: pytabkit RealMLP_TD_Classifier, 3 x 384 mish, n_ens 8, lr 0.03, wd 0.02, p_drop 0.08, batch 1024, up to
80 epochs with early stopping on the validation fold (classification error, additive patience 20), random_state 42.
Test = mean of the 5 fold models.
Usage: python train_snapped_proxies_rare_flags_a.py"""
import gc
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_snapped_proxies_rare_flags_a"
SEED = 42
RARE_MIN_COUNT = 25

BASE_CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
BASE_NUMS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
SNAP_NUMS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"]

CAT_PROXY = [f"{c}__cat" for c in SNAP_NUMS]
CAT_RARE = [f"{c}__is_rare" for c in SNAP_NUMS]
CAT_COLS = BASE_CATS + CAT_PROXY + CAT_RARE
NUM_COLS = BASE_NUMS[:]
FEATURES = CAT_COLS + NUM_COLS

REALMLP_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": SEED,
    "verbosity": 2,
    "val_metric_name": "class_error",
    "n_epochs": 80,
    "batch_size": 1024,
    "n_ens": 8,
    "use_early_stopping": True,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 1,
    "act": "mish",
    "hidden_sizes": "rectangular",
    "hidden_width": 384,
    "n_hidden_layers": 3,
    "embedding_size": 8,
    "p_drop": 0.08,
    "p_drop_sched": "flat_cos",
    "lr": 0.03,
    "wd": 0.02,
    "use_ls": False,
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """Values seen >= rare_min_count times in train_series are frequent; others snap to the nearest frequent value
    (ties -> the larger one)."""
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
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
                choose_right = np.abs(v - right_vals) <= np.abs(v - left_vals)
                nearest = np.where(choose_right, right_vals, left_vals)
                x_snapped[idx_snap] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


def preprocess_base(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    if ID in train.columns:
        train = train.drop(columns=[ID])
    if ID in test.columns:
        test = test.drop(columns=[ID])

    keep_train = [c for c in BASE_CATS + BASE_NUMS + [TARGET] if c in train.columns]
    keep_test = [c for c in BASE_CATS + BASE_NUMS if c in test.columns]
    train = train[keep_train].copy()
    test = test[keep_test].copy()

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int8)

    for c in BASE_NUMS:
        train[c] = pd.to_numeric(train[c], errors="coerce").astype(np.float32)
        test[c] = pd.to_numeric(test[c], errors="coerce").astype(np.float32)

    for c in BASE_CATS:
        train[c] = train[c].astype("object").fillna("__MISSING__").astype(str)
        test[c] = test[c].astype("object").fillna("__MISSING__").astype(str)

    return train, test


def build_fold_features(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    te_df = te_df.copy()

    for c in BASE_NUMS:
        med = pd.to_numeric(tr_df[c], errors="coerce").median()
        med = 0.0 if pd.isna(med) else float(med)
        tr_df[c] = pd.to_numeric(tr_df[c], errors="coerce").fillna(med).astype(np.float32)
        va_df[c] = pd.to_numeric(va_df[c], errors="coerce").fillna(med).astype(np.float32)
        te_df[c] = pd.to_numeric(te_df[c], errors="coerce").fillna(med).astype(np.float32)

    for col in SNAP_NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr_df[col])
        va_snap, va_israre = snapper(va_df[col])
        te_snap, te_israre = snapper(te_df[col])

        tr_df[f"{col}__cat"] = pd.Series(tr_snap, index=tr_df.index).astype(str).values
        va_df[f"{col}__cat"] = pd.Series(va_snap, index=va_df.index).astype(str).values
        te_df[f"{col}__cat"] = pd.Series(te_snap, index=te_df.index).astype(str).values

        tr_df[f"{col}__is_rare"] = pd.Series(tr_israre, index=tr_df.index).astype(str).values
        va_df[f"{col}__is_rare"] = pd.Series(va_israre, index=va_df.index).astype(str).values
        te_df[f"{col}__is_rare"] = pd.Series(te_israre, index=te_df.index).astype(str).values

    for c in CAT_COLS:
        tr_df[c] = tr_df[c].astype("category")
        va_df[c] = va_df[c].astype("category")
        te_df[c] = te_df[c].astype("category")

    x_tr = tr_df[FEATURES].copy()
    x_va = va_df[FEATURES].copy()
    x_te = te_df[FEATURES].copy()
    y_tr = tr_df[TARGET].astype(np.int8).values
    y_va = va_df[TARGET].astype(np.int8).values
    return x_tr, y_tr, x_va, y_va, x_te


def main():
    seed_everything(SEED)

    train = read_train()
    test = read_test()
    log(f"train shape = {train.shape}")
    log(f"test  shape = {test.shape}")

    train, test = preprocess_base(train, test)
    y = train[TARGET].values.astype(np.int8)
    log(f"Target distribution = {np.bincount(y, minlength=3).tolist()}")

    oof = np.zeros((len(train), 3), dtype=np.float32)
    pred_test = np.zeros((len(test), 3), dtype=np.float32)

    fold_id = folds(y)
    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        fold_t0 = time.time()
        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        te_df = test.copy()

        x_tr, y_tr, x_va, y_va, x_te = build_fold_features(tr_df, va_df, te_df)
        log(f"================ Fold {fold}/{N_FOLDS} ================")
        log(f"  -> fitting RealMLP: n_tr={len(x_tr):,} n_va={len(x_va):,}")

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(x_tr, y_tr, X_val=x_va, y_val=y_va)

        va_pred = np.asarray(model.predict_proba(x_va), dtype=np.float32)
        te_pred = np.asarray(model.predict_proba(x_te), dtype=np.float32)

        oof[va_idx] = va_pred
        pred_test += te_pred / N_FOLDS

        fold_bac = balanced_accuracy_score(y_va, np.argmax(va_pred, axis=1))
        log(f"[Fold {fold}] val BAC: {fold_bac:.6f} | fold time {time.time() - fold_t0:.1f}s")

        del tr_df, va_df, te_df, x_tr, y_tr, x_va, y_va, x_te, model, va_pred, te_pred
        gc.collect()

    cv = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    log("=" * 70)
    log(f"OOF CV BAC: {cv:.6f}")
    save_pair(model_dir(FOLDER), PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
