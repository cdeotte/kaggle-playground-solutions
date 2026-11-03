#!/usr/bin/env python
"""GPU XGBoost on fold-safe class target encodings of numeric x categorical keys, digit features and the rule score
of the original data (pairs xgb_numeric_pair_te_a, xgb_numeric_pair_te_maxbin16000_a).

Features:
- the 8 categoricals (native categoricals) and the 11 numerics;
- digit features of the 11 numerics (built on train + test): each integer digit position and the first 2 decimal
  digits (44 int16 columns);
- the rule score 2 [Soil_Moisture < 25] + 2 [Rainfall_mm < 300] + [Temperature_C > 30] + [Wind_Speed_kmh > 10]
  - 2 [stage Harvest] - 2 [stage Sowing] - [mulch Yes], its low (<= 0) / high (> 3) flags, the 4 threshold flags,
  Soil_Moisture / (Rainfall_mm + 1) and Temperature_C x Wind_Speed_kmh;
- per fold, 99 keys: every numeric as an exact-value key (11) and every numeric x categorical pair (88). Each key gets
  3 class-rate target encodings (cuML TargetEncoder, 5 random inner folds, smooth 0, seed 42 + fold + class): out-of-
  fold on the training rows (fit_transform), refitted on all training rows for the validation / test rows; plus the
  count of every numeric value on the training rows (11 columns).
Model: XGBoost hist on GPU with native categoricals, depth 4, lr 0.03, up to 12,000 rounds, subsample 0.8,
colsample_bytree 0.8, min_child_weight 5, alpha 0.05, lambda 1, early stopping 200 (mlogloss) on the validation fold,
balanced class weights, random_state 42; the maxbin16000 preset uses max_bin 16,000 (default 256). Test = mean of the
5 fold models. Competition rows only.
About 4 minutes per fold on one GPU (20 minutes with max_bin 16,000).
Usage: python train_numeric_pair_te_a.py <pair> [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, TARGET, TARGET_MAP, folds, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

LABELS = list(CLASSES)
N_CLASSES = len(LABELS)

N_FOLDS = 5
INNER_TE_SPLITS = 5
SEED = 42

CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
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
DIGIT_BASE_COLS = NUMS

XGB_PARAMS = {
    "n_estimators": 12000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "num_class": N_CLASSES,
    "n_jobs": -1,
    "random_state": SEED,
    "early_stopping_rounds": 200,
    "device": "cuda",
    "tree_method": "hist",
    "enable_categorical": True,
    "verbosity": 0,
}

# pair -> overrides of XGB_PARAMS
PRESETS = {
    "xgb_numeric_pair_te_a": {},
    "xgb_numeric_pair_te_maxbin16000_a": {"max_bin": 16000},
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def add_formula_features(df):
    sm = df["Soil_Moisture"]
    rm = df["Rainfall_mm"]
    tc = df["Temperature_C"]
    ws = df["Wind_Speed_kmh"]
    cgs = df["Crop_Growth_Stage"].astype(str)
    mu = df["Mulching_Used"].astype(str)
    score = (
        (sm < 25).astype(int) * 2
        + (rm < 300).astype(int) * 2
        + (tc > 30).astype(int)
        + (ws > 10).astype(int)
        - (cgs == "Harvest").astype(int) * 2
        - (cgs == "Sowing").astype(int) * 2
        - (mu == "Yes").astype(int)
    )
    out = pd.DataFrame(index=df.index)
    out["formula_score"] = score.astype("float32")
    out["is_low"] = (score <= 0).astype("float32")
    out["is_high"] = (score > 3).astype("float32")
    out["SM_thr"] = (sm < 25).astype("float32")
    out["RM_thr"] = (rm < 300).astype("float32")
    out["TC_thr"] = (tc > 30).astype("float32")
    out["WS_thr"] = (ws > 10).astype("float32")
    out["SM_div_RM"] = (sm / (rm + 1.0)).astype("float32")
    out["TC_x_WS"] = (tc * ws).astype("float32")
    return out


def add_digit_features(combined_df: pd.DataFrame, base_cols, max_dec_digits=2):
    digit_cols = []
    out = combined_df.copy()
    for col in base_cols:
        if col not in out.columns:
            continue
        col_vals = pd.to_numeric(out[col], errors="coerce").fillna(out[col].median()).astype(np.float64)

        max_val = float(col_vals.abs().max())
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1

        new_cols = {}
        for i in range(max_int_digits):
            name = f"{col}_digit_pos_{i + 1}"
            new_cols[name] = ((np.floor(col_vals.abs()) // (10 ** i)) % 10).astype("int16")
            digit_cols.append(name)

        frac_exists = (np.abs(col_vals) % 1 != 0).any()
        if frac_exists:
            for i in range(1, max_dec_digits + 1):
                name = f"{col}_digit_dec_{i}"
                new_cols[name] = ((np.abs(col_vals) * (10 ** i)).round().astype(int) % 10).astype("int16")
                digit_cols.append(name)

        if new_cols:
            out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out, digit_cols


def numeric_to_key(s: pd.Series, decimals=6) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(np.float64)
    x = x.fillna(-999999.123456)
    out = x.map(lambda v: f"{v:.{decimals}f}")
    out = out.str.rstrip("0").str.rstrip(".")
    out = out.replace("", "0")
    return out.astype(str)


def make_count_encoding_features(train_df, valid_df, test_df, nums):
    tr_out, va_out, te_out = {}, {}, {}
    for n in nums:
        col = f"CNTNUM__{n}"
        tr_key = numeric_to_key(train_df[n])
        va_key = numeric_to_key(valid_df[n])
        te_key = numeric_to_key(test_df[n])
        cnt = tr_key.value_counts(dropna=False)
        tr_out[col] = tr_key.map(cnt).fillna(0).astype(np.float32)
        va_out[col] = va_key.map(cnt).fillna(0).astype(np.float32)
        te_out[col] = te_key.map(cnt).fillna(0).astype(np.float32)
    return (
        pd.DataFrame(tr_out, index=train_df.index),
        pd.DataFrame(va_out, index=valid_df.index),
        pd.DataFrame(te_out, index=test_df.index),
    )


def make_pair_feature_frames(train_df, valid_df, test_df, cats, nums):
    train_blocks, valid_blocks, test_blocks = {}, {}, {}
    for n in nums:
        nk_tr = numeric_to_key(train_df[n])
        nk_va = numeric_to_key(valid_df[n])
        nk_te = numeric_to_key(test_df[n])
        train_blocks[f"TENUM__{n}"] = nk_tr
        valid_blocks[f"TENUM__{n}"] = nk_va
        test_blocks[f"TENUM__{n}"] = nk_te
        for c in cats:
            tr_cat = train_df[c].astype(str).fillna("__MISSING__")
            va_cat = valid_df[c].astype(str).fillna("__MISSING__")
            te_cat = test_df[c].astype(str).fillna("__MISSING__")
            col = f"TEPAIR__{n}__X__{c}"
            train_blocks[col] = nk_tr + "__" + tr_cat
            valid_blocks[col] = nk_va + "__" + va_cat
            test_blocks[col] = nk_te + "__" + te_cat
    return (
        pd.DataFrame(train_blocks, index=train_df.index),
        pd.DataFrame(valid_blocks, index=valid_df.index),
        pd.DataFrame(test_blocks, index=test_df.index),
    )


def make_cuml_te_features(train_raw, valid_raw, test_raw, y_tr, seed):
    cols = list(train_raw.columns)
    tr_enc = np.zeros((len(train_raw), len(cols) * N_CLASSES), dtype=np.float32)
    va_enc = np.zeros((len(valid_raw), len(cols) * N_CLASSES), dtype=np.float32)
    te_enc = np.zeros((len(test_raw), len(cols) * N_CLASSES), dtype=np.float32)
    out_cols = []

    train_g = cudf.DataFrame({c: cudf.Series(train_raw[c].astype(str).values) for c in cols})
    valid_g = cudf.DataFrame({c: cudf.Series(valid_raw[c].astype(str).values) for c in cols})
    test_g = cudf.DataFrame({c: cudf.Series(test_raw[c].astype(str).values) for c in cols})

    for j, col in enumerate(cols):
        if (j == 0) or ((j + 1) % 20 == 0) or (j + 1 == len(cols)):
            log(f"    TE col {j + 1}/{len(cols)}: {col}")
        X_tr_col = train_g[[col]]
        X_va_col = valid_g[[col]]
        X_te_col = test_g[[col]]

        for k in range(N_CLASSES):
            y_k_g = cudf.Series((y_tr == k).astype(np.int32))
            te_enc_model = TargetEncoder(
                n_folds=INNER_TE_SPLITS,
                smooth=0,
                seed=seed + k,
                split_method="random",
                stat="mean",
                output_type="cupy",
            )
            idx = j * N_CLASSES + k
            tr_enc[:, idx] = cp.asnumpy(te_enc_model.fit_transform(X_tr_col, y_k_g)).ravel()
            te_enc_model.fit(X_tr_col, y_k_g)
            va_enc[:, idx] = cp.asnumpy(te_enc_model.transform(X_va_col)).ravel()
            te_enc[:, idx] = cp.asnumpy(te_enc_model.transform(X_te_col)).ravel()
            out_cols.append(f"{col}__p{k}")

    return (
        pd.DataFrame(tr_enc, columns=out_cols, index=train_raw.index),
        pd.DataFrame(va_enc, columns=out_cols, index=valid_raw.index),
        pd.DataFrame(te_enc, columns=out_cols, index=test_raw.index),
    )


def make_sample_weights(y: np.ndarray):
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1.0))
    return weights[y].astype(np.float32)


def build_features():
    train = read_train()
    test = read_test()
    log(f"train: {train.shape} test: {test.shape}")
    log(f"XGBoost version: {xgb.__version__}")

    y = train[TARGET].map(TARGET_MAP).astype(np.int8).values

    for df in (train, test):
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            df[c] = df[c].fillna(df[c].median())
        for c in CATS:
            df[c] = df[c].astype(str).fillna("__MISSING__").astype("category")

    formula_train = add_formula_features(train)
    formula_test = add_formula_features(test)

    combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    combined, digit_cols = add_digit_features(combined, DIGIT_BASE_COLS, max_dec_digits=2)
    train_base = combined.iloc[: len(train)].reset_index(drop=True).copy()
    test_base = combined.iloc[len(train):].reset_index(drop=True).copy()
    return train_base, test_base, formula_train, formula_test, digit_cols, y


def run_fold(fold, tr_idx, va_idx, train_base, test_base, formula_train, formula_test, digit_cols, y, xgb_params):
    """fold: 1..5 (seeds the inner target encoders)."""
    raw_feature_cols = CATS + NUMS + digit_cols
    log(f"### OUTER Fold {fold}/{N_FOLDS} ###")
    t0 = time.time()
    X_tr_base = train_base.iloc[tr_idx].reset_index(drop=True).copy()
    X_va_base = train_base.iloc[va_idx].reset_index(drop=True).copy()
    X_te_base = test_base.reset_index(drop=True).copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    tr_raw, va_raw, te_raw = make_pair_feature_frames(X_tr_base, X_va_base, X_te_base, CATS, NUMS)
    X_tr_te, X_va_te, X_te_te = make_cuml_te_features(tr_raw, va_raw, te_raw, y_tr, seed=SEED + fold)
    X_tr_cnt, X_va_cnt, X_te_cnt = make_count_encoding_features(X_tr_base, X_va_base, X_te_base, NUMS)

    X_tr = pd.concat([X_tr_base[raw_feature_cols].reset_index(drop=True), X_tr_te, X_tr_cnt, formula_train.iloc[tr_idx].reset_index(drop=True)], axis=1)
    X_va = pd.concat([X_va_base[raw_feature_cols].reset_index(drop=True), X_va_te, X_va_cnt, formula_train.iloc[va_idx].reset_index(drop=True)], axis=1)
    X_te = pd.concat([X_te_base[raw_feature_cols].reset_index(drop=True), X_te_te, X_te_cnt, formula_test.reset_index(drop=True)], axis=1)

    for df in (X_tr, X_va, X_te):
        for c in CATS:
            df[c] = df[c].astype(str).astype("category")
        for c in df.columns:
            if c not in CATS:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.float32)

    sw = make_sample_weights(y_tr)
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        verbose=500,
    )

    va_pred = model.predict_proba(X_va)
    te_pred = model.predict_proba(X_te)

    bac = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
    best_iter = getattr(model, "best_iteration", None)
    log(f"Fold {fold} BAC: {bac:.8f} | best_iteration={best_iter} | time={time.time() - t0:.1f}s")

    del X_tr_base, X_va_base, X_te_base, tr_raw, va_raw, te_raw
    del X_tr_te, X_va_te, X_te_te, X_tr_cnt, X_va_cnt, X_te_cnt
    del X_tr, X_va, X_te, model, sw
    gc.collect()
    return va_pred.astype(np.float32), te_pred.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(PRESETS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    pair = args.pair
    xgb_params = {**XGB_PARAMS, **PRESETS[pair]}
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        train_base, test_base, formula_train, formula_test, digit_cols, y = build_features()
        log(f"Raw cats: {len(CATS)} | raw nums: {len(NUMS)} | digit cols: {len(digit_cols)}")
        fold_id = folds(y)
        for k in todo:
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            va_pred, te_pred = run_fold(k + 1, tr_idx, va_idx, train_base, test_base, formula_train, formula_test,
                                        digit_cols, y, xgb_params)
            save_part(HERE, pair, k, va_pred, te_pred)
    y = read_train()[TARGET].map(TARGET_MAP).astype(np.int8).values
    assemble(HERE, pair, folds(y), y)


if __name__ == "__main__":
    main()
