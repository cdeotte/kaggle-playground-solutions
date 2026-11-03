#!/usr/bin/env python
"""Depth-2 GPU XGBoost on the raw columns plus 7 water-balance interactions (pair xgb_minimal_fe_depth2_a).

Features (built on train + test): the 11 numerics (float32) and 7 interactions: soil moisture - (rain + previous
irrigation), its absolute value, (rain + previous irrigation) / (field area + 1), moisture / water and water / moisture
ratios, previous irrigation x rain, temperature x humidity; the 8 categoricals as native categoricals with the
train + test category sets. No original rows, priors, counts, bins or pairs.
Model: XGBoost hist on GPU, depth 2, lr 0.1, max_bin 16,000, up to 10,000 rounds, early stopping 300 (mlogloss) on the
validation fold, random_state 42. Test = mean of the 5 fold models.
About 6 minutes per fold on one GPU.
Usage: python train_minimal_fe_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "xgb_minimal_fe_depth2_a"
ID_COL = ID
CLASS_TO_INT = TARGET_MAP

RANDOM_STATE = 42
N_THREADS = 32
EVAL_LOG_EVERY = 500

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

xgb_params = dict(
    n_estimators=10000,
    learning_rate=0.1,
    max_depth=2,
    max_bin=16000,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=N_THREADS,
    early_stopping_rounds=300,
    verbosity=0,
    device="cuda",
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def preprocess_minimal_fe_together(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    t0 = time.time()
    log("Preprocess: start (concat train+test)...")

    X_tr = X_tr.copy()
    X_te = X_te.copy()

    X_tr = X_tr.drop(columns=[c for c in [ID_COL] if c in X_tr.columns], errors="ignore")
    X_te = X_te.drop(columns=[c for c in [ID_COL] if c in X_te.columns], errors="ignore")

    n_tr = len(X_tr)
    X_all = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    log(f"Preprocess: concatenated -> X_all shape={X_all.shape}")

    for c in BASE_NUMS:
        if c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce").astype("float32")
            med = float(np.nanmedian(X_all[c].values))
            if not np.isfinite(med):
                med = 0.0
            X_all[c] = X_all[c].fillna(med).astype("float32")

    log("Preprocess: adding 7 numeric interaction features...")
    rain = X_all["Rainfall_mm"].astype("float32")
    prev = X_all["Previous_Irrigation_mm"].astype("float32")
    moist = X_all["Soil_Moisture"].astype("float32")
    hum = X_all["Humidity"].astype("float32")
    temp = X_all["Temperature_C"].astype("float32")
    area = X_all["Field_Area_hectare"].astype("float32")

    X_all["F_water_deviation"] = (moist - (rain + prev)).astype("float32")
    X_all["F_abs_water_dev"] = np.abs(X_all["F_water_deviation"]).astype("float32")
    X_all["F_avg_water_input"] = ((rain + prev) / (area + 1.0)).astype("float32")
    X_all["F_moisture_to_water_ratio"] = (moist / (rain + prev + 1.0)).astype("float32")
    X_all["F_water_to_moisture_ratio"] = ((rain + prev) / (moist + 1.0)).astype("float32")
    X_all["F_prev_x_rain"] = (prev * rain).astype("float32")
    X_all["F_temp_x_humidity"] = (temp * hum).astype("float32")

    log("Preprocess: aligning categorical dtypes across train/test...")
    cat_cols = set(X_all.select_dtypes(include=["object"]).columns.tolist())
    cat_cols |= set(X_all.select_dtypes(include=["category"]).columns.tolist())
    cat_cols = sorted(cat_cols)

    for i, c in enumerate(cat_cols, 1):
        s = X_all[c]
        if str(s.dtype) == "category":
            s = s.astype("object")
        non_null = s[~pd.isna(s)]
        cats = pd.Index(non_null.map(lambda v: str(v)).unique(), dtype="object").sort_values()
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)
        X_all[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(dtype)
        if i % 10 == 0 or i == len(cat_cols):
            log(f"  ...dtype aligned {i}/{len(cat_cols)}")

    X_tr2 = X_all.iloc[:n_tr].reset_index(drop=True)
    X_te2 = X_all.iloc[n_tr:].reset_index(drop=True)

    log(f"Preprocess: done in {time.time()-t0:.1f}s | X_train={X_tr2.shape} X_test={X_te2.shape}")
    return X_tr2, X_te2


def build_features():
    log("Loading train/test...")
    train_df = read_train()
    test_df = read_test()
    log(f"Loaded train: {train_df.shape} | test: {test_df.shape}")

    assert TARGET in train_df.columns, f"Missing target column '{TARGET}' in train.csv"
    assert TARGET not in test_df.columns, "test.csv should not contain the target column"

    y = train_df[TARGET].astype(str).str.strip().map(CLASS_TO_INT).astype(np.int32).values

    X_train_raw = train_df.drop(columns=[TARGET]).copy()
    X_test_raw = test_df.copy()
    X_train, X_test = preprocess_minimal_fe_together(X_train_raw, X_test_raw)

    if X_train.isna().any().any() or X_test.isna().any().any():
        n_nan_tr = int(X_train.isna().sum().sum())
        n_nan_te = int(X_test.isna().sum().sum())
        log(f"NaNs -> train: {n_nan_tr}, test: {n_nan_te}. XGBoost will handle NaNs natively.")
    return X_train, X_test, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_train, X_test, y = build_features()
        log(f"Ready to train | X_train={X_train.shape} X_test={X_test.shape}")
        fold_id = folds(y)
        for k in todo:
            t0 = time.time()
            trn_idx, val_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
            y_trn, y_val = y[trn_idx], y[val_idx]

            model = XGBClassifier(**xgb_params)
            log(f"Fold {k}: fitting XGB...")
            model.fit(
                X_trn,
                y_trn,
                eval_set=[(X_val, y_val)],
                verbose=EVAL_LOG_EVERY,
            )

            p_val = model.predict_proba(X_val).astype(np.float32)
            p_test = model.predict_proba(X_test).astype(np.float32)
            save_part(HERE, PAIR, k, p_val, p_test)
            bac = balanced_accuracy_score(y_val, p_val.argmax(axis=1))
            log(f"{PAIR} fold {k}: BAC={bac:.6f} | best_iteration={getattr(model, 'best_iteration', None)} | "
                f"elapsed={time.time() - t0:.1f}s")
            del model, X_trn, X_val, y_trn, y_val, p_val, p_test
            gc.collect()
    y = read_train()[TARGET].astype(str).str.strip().map(CLASS_TO_INT).astype(np.int32).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
