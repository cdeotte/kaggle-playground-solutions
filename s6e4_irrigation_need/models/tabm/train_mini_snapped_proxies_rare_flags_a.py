#!/usr/bin/env python
"""Compact TabM-mini on the raw columns plus fold-safe snapped numeric proxies and rare flags
(pair tabm_mini_snapped_proxies_rare_flags_a).

Features (41 columns, competition rows only; numeric NaN -> train median): the 8 categoricals; for each of the 11
numerics a snapper fitted on the fold's training rows maps every value seen fewer than 25 times to the nearest frequent
value: the snapped value (`__cat`) and the rare flag (`__is_rare`) as categories; the 11 numerics standardized on the
fold's training rows.
Model: pytabkit TabM_D_Classifier, tabm-mini, k 16, 2 blocks x 256, piecewise-linear numeric embeddings (d 8),
dropout 0.2, lr 3e-4, batch 512, 5 epochs (best epoch on the validation fold's classification error),
random_state 100. Test = mean of the 5 fold models.
Usage: python train_mini_snapped_proxies_rare_flags_a.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "tabm"
PAIR = "tabm_mini_snapped_proxies_rare_flags_a"
SEED = 42
LABELS = list(CLASSES)

BASE_FEATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
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

NUMS = [
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
CAT_BASE = [c for c in BASE_FEATS if c not in NUMS]

RARE_MIN_COUNT = 25

TABM_PARAMS = {
    "device": "cuda",
    "val_metric_name": "class_error",
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preprocess_base(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    for c in NUMS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
    for c in CAT_BASE:
        df[c] = df[c].astype(str).fillna("nan")
    if is_train:
        df[TARGET] = df[TARGET].map(TARGET_MAP).astype(np.int8)
    return df


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


def add_fold_safe_proxy_features(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
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


def build_tabm_matrices(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
    tr_df = tr_df.copy()
    va_df = va_df.copy()
    te_df = te_df.copy()

    cat_all = CAT_BASE + [f"{c}__cat" for c in NUMS] + [f"{c}__is_rare" for c in NUMS]
    num_all = NUMS
    features = cat_all + num_all

    scaler = StandardScaler()
    tr_df[num_all] = scaler.fit_transform(tr_df[num_all].astype(np.float32)).astype(np.float32)
    va_df[num_all] = scaler.transform(va_df[num_all].astype(np.float32)).astype(np.float32)
    te_df[num_all] = scaler.transform(te_df[num_all].astype(np.float32)).astype(np.float32)

    for c in cat_all:
        tr_df[c] = tr_df[c].astype("category")
        va_df[c] = va_df[c].astype("category")
        te_df[c] = te_df[c].astype("category")

    x_tr = tr_df[features].copy()
    x_va = va_df[features].copy()
    x_te = te_df[features].copy()
    return x_tr, x_va, x_te, cat_all, num_all


def main():
    np.random.seed(SEED)
    log("Loading train/test...")
    train = read_train()
    test = read_test()

    log(f"train shape = {train.shape}")
    log(f"test  shape = {test.shape}")

    train = preprocess_base(train, is_train=True)
    test = preprocess_base(test, is_train=False)

    for c in NUMS:
        med = train[c].median()
        train[c] = train[c].fillna(med).astype(np.float32)
        test[c] = test[c].fillna(med).astype(np.float32)

    y = train[TARGET].values.astype(np.int8)
    log(f"Class order for saved probabilities: {LABELS}")

    fold_id = folds(y)
    oof = np.zeros((len(train), len(LABELS)), dtype=np.float32)
    pred_test = np.zeros((len(test), len(LABELS)), dtype=np.float32)
    fold_bacs = []

    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        fold_t0 = time.time()
        log("=" * 80)
        log(f"Fold {fold}/{N_FOLDS}")

        tr_df = train.iloc[tr_idx].reset_index(drop=True).copy()
        va_df = train.iloc[va_idx].reset_index(drop=True).copy()
        te_df = test.copy()

        tr_df, va_df, te_df = add_fold_safe_proxy_features(tr_df, va_df, te_df)
        x_tr, x_va, x_te, cat_all, num_all = build_tabm_matrices(tr_df, va_df, te_df)

        y_tr = tr_df[TARGET].values.astype(np.int8)
        y_va = va_df[TARGET].values.astype(np.int8)

        log(f"X_tr shape = {x_tr.shape}")
        log(f"n_cat = {len(cat_all)}, n_num = {len(num_all)}")

        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(
            x_tr,
            y_tr,
            x_va,
            y_va,
            cat_col_names=cat_all,
        )

        p_va = model.predict_proba(x_va).astype(np.float32)
        p_te = model.predict_proba(x_te).astype(np.float32)

        oof[va_idx] = p_va
        pred_test += p_te / N_FOLDS

        fold_bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
        fold_bacs.append(fold_bac)
        log(f"Fold {fold} BAC = {fold_bac:.6f} | time = {time.time() - fold_t0:.1f}s")

        del model, tr_df, va_df, te_df, x_tr, x_va, x_te, y_tr, y_va
        gc.collect()

    cv_bac = balanced_accuracy_score(y, oof.argmax(axis=1))
    log("=" * 80)
    log(f"Fold BACs: {[round(float(x), 6) for x in fold_bacs]}")
    log(f"OOF CV BAC: {cv_bac:.6f}")
    save_pair(model_dir(FOLDER), PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
