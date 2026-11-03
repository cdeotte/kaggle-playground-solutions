#!/usr/bin/env python
"""Depth-1 GPU XGBoost on nearest-original "snap" values, decimal digits and radix interactions
(pairs xgb_snap_radix_depth1_a, xgb_snap_radix_field_area_depth1_a, xgb_snap_radix_sunlight_depth1_a).

Features (built once on train + test; the original rows only provide the snap targets):
- the 8 categoricals as native categoricals and the 11 numerics as float32;
- snap: every train / test value of Rainfall_mm and Previous_Irrigation_mm that does not occur in the original data is
  replaced by the nearest original value (<col>_snap), plus the difference snap - value (<col>_snap_diff);
- the decimal part of Rainfall_mm / Previous_Irrigation_mm written with 2 decimals: the 2-digit string and each digit
  (native categoricals) and its length;
- radix interactions: round(100 x snapped value) + 100,000 x (code of a categorical), for both snapped columns and each
  of the 8 categoricals, i.e. the snapped value within each category (16 integer columns; 47 columns in total);
- the field-area / sunlight presets add one more block: round(100 x Field_Area_hectare) or round(100 x Sunlight_Hours),
  clipped to [0, 99,999], combined with each categorical the same way (8 more columns).
Model: XGBoost hist on GPU with native categoricals, depth 1 (stumps), lr 0.1, up to 1,000,000 rounds,
min_child_weight 5, max_bin 16,000, subsample 0.85, colsample_bytree 0.85, early stopping 300 (mlogloss) on the
validation fold, random_state 42. Test = mean of the 5 fold models. The original rows are never training rows.
The feature build takes about a minute; each fold 4-10 minutes on one GPU (9,000-19,000 rounds).
Usage: python train_snap_radix_a.py <pair> [--folds 0,1,2,3,4] [--force]"""
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
from common.data import CATEGORICALS, NUMERICS, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_THREADS = 32

CATEGORICAL_COLS = list(CATEGORICALS)
SNAP_COLS = ["Rainfall_mm", "Previous_Irrigation_mm"]
RADIX = 100_000

xgb_params = dict(
    n_estimators=1_000_000,
    learning_rate=0.1,
    max_depth=1,
    min_child_weight=5,
    max_bin=16000,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=N_THREADS,
    early_stopping_rounds=300,
    device="cuda",
)

# pair -> extra raw numeric radix block (column, prefix of the new columns) or None
PRESETS = {
    "xgb_snap_radix_depth1_a": None,
    "xgb_snap_radix_field_area_depth1_a": ("Field_Area_hectare", "fa"),
    "xgb_snap_radix_sunlight_depth1_a": ("Sunlight_Hours", "sh"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_snap_map(combo_s: pd.Series, orig_s: pd.Series, colname: str) -> dict:
    combo_vals = pd.to_numeric(combo_s, errors="coerce")
    orig_vals = pd.to_numeric(orig_s, errors="coerce")

    orig_unique = np.sort(orig_vals.dropna().unique())
    combo_unique = np.sort(combo_vals.dropna().unique())
    missing = combo_unique[~np.isin(combo_unique, orig_unique)]

    snap_map = {}
    if len(orig_unique) == 0 or len(missing) == 0:
        return snap_map

    for x in missing:
        j = int(np.argmin(np.abs(orig_unique - x)))
        snap_map[float(x)] = float(orig_unique[j])

    log(f"{colname}: will snap {len(snap_map):,} values not in orig -> closest orig value")
    return snap_map


def apply_snap(combo_df: pd.DataFrame, orig_df: pd.DataFrame, col: str) -> None:
    snap_col = f"{col}_snap"
    diff_col = f"{col}_snap_diff"

    orig_num = pd.to_numeric(combo_df[col], errors="coerce").astype("float32")
    combo_df[col] = orig_num
    combo_df[snap_col] = orig_num.copy()

    snap_map = _build_snap_map(combo_df[col], orig_df[col], col)
    if snap_map:
        combo_df[snap_col] = combo_df[snap_col].replace(snap_map)

    combo_df[diff_col] = (combo_df[snap_col] - combo_df[col]).astype("float32")


def add_decimal_parts(df: pd.DataFrame, col: str, prefix: str) -> list[str]:
    s = df[col].map(lambda x: format(float(x), ".2f") if pd.notna(x) else "NA")
    frac = s.str.split(".").str[1].fillna("")
    df[prefix] = frac.astype("string")
    df[f"{prefix}0"] = frac.str[0].fillna("-1").astype("string")
    df[f"{prefix}1"] = frac.str[1].fillna("-1").astype("string")
    df[f"{prefix}_len"] = frac.str.len().fillna(0).astype("int32")
    return [prefix, f"{prefix}0", f"{prefix}1", f"{prefix}_len"]


def build_features(extra):
    log("Loading train/test/orig...")
    train = read_train()
    test = read_test()
    orig = read_original()

    n_train = len(train)
    y = train[TARGET].map(TARGET_MAP).astype("int32").to_numpy()

    test[TARGET] = "Low"
    combo = pd.concat([train, test], ignore_index=True)

    for col in SNAP_COLS:
        apply_snap(combo, orig, col)

    engineered = []
    engineered += add_decimal_parts(combo, "Rainfall_mm", "rf")
    engineered += add_decimal_parts(combo, "Previous_Irrigation_mm", "pi")

    for c in ["rf", "rf0", "rf1", "pi", "pi0", "pi1"]:
        combo[c] = combo[c].astype("category")

    num_cols = list(NUMERICS)
    base_feats = CATEGORICAL_COLS + num_cols

    for col in num_cols + [f"{c}_snap" for c in SNAP_COLS] + [f"{c}_snap_diff" for c in SNAP_COLS]:
        combo[col] = pd.to_numeric(combo[col], errors="coerce").astype("float32")
        med = float(np.nanmedian(combo.loc[: n_train - 1, col].to_numpy()))
        combo[col] = combo[col].fillna(med).astype("float32")

    combo["Rainfall_mm_snap_i"] = np.round(combo["Rainfall_mm_snap"] * 100.0).astype("int32")
    combo["Previous_Irrigation_mm_snap_i"] = np.round(combo["Previous_Irrigation_mm_snap"] * 100.0).astype("int32")
    radix_sources = [("Rainfall_mm_snap_i", "rf"), ("Previous_Irrigation_mm_snap_i", "pi")]
    if extra is not None:
        extra_col, extra_prefix = extra
        combo[f"{extra_col}_i"] = np.round(combo[extra_col] * 100.0).astype("int32")
        combo[f"{extra_col}_i"] = np.clip(combo[f"{extra_col}_i"], 0, RADIX - 1).astype("int32")
        radix_sources.append((f"{extra_col}_i", extra_prefix))

    lowcard_codes = {}
    for c in CATEGORICAL_COLS:
        s = combo[c].astype("string").fillna("NA")
        cat = pd.Categorical(s)
        lowcard_codes[c] = cat.codes.astype("int32")
        combo[c] = cat

    radix_cols = []
    for snap_i, prefix in radix_sources:
        for c in CATEGORICAL_COLS:
            newc = f"{prefix}_radix_{c}"
            combo[newc] = (combo[snap_i].astype("int32") + lowcard_codes[c] * RADIX).astype("int32")
            radix_cols.append(newc)

    engineered += [
        "Rainfall_mm_snap",
        "Rainfall_mm_snap_diff",
        "Previous_Irrigation_mm_snap",
        "Previous_Irrigation_mm_snap_diff",
    ]

    all_feats = base_feats + engineered + radix_cols
    all_feats = list(dict.fromkeys(all_feats))
    all_feats = [c for c in all_feats if c not in {"id", TARGET}]

    x_all = combo.iloc[:n_train][all_feats].copy()
    t_all = combo.iloc[n_train:][all_feats].copy()

    cat_cols = x_all.select_dtypes(include=["category"]).columns.tolist()
    for c in cat_cols:
        x_all[c] = x_all[c].astype("object").astype("category")
        t_all[c] = t_all[c].astype("object").astype("category")

    log(f"Final features: {len(all_feats)} | X={x_all.shape} | T={t_all.shape}")
    return x_all, t_all, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(PRESETS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    pair = args.pair
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        x_all, t_all, y = build_features(PRESETS[pair])
        fold_id = folds(y)
        for k in todo:
            tf = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            x_tr, y_tr = x_all.iloc[tr_idx], y[tr_idx]
            x_va, y_va = x_all.iloc[va_idx], y[va_idx]

            model = XGBClassifier(**xgb_params)
            model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=200)
            va_pred = model.predict_proba(x_va)
            te_pred = model.predict_proba(t_all)
            save_part(HERE, pair, k, va_pred, te_pred)

            bac = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
            log(f"{pair} fold {k}: BAC={bac:.6f} | best_iter={getattr(model, 'best_iteration', None)} | "
                f"dt={time.time() - tf:.1f}s")
            del model
            gc.collect()
    y = read_train()[TARGET].map(TARGET_MAP).astype("int32").to_numpy()
    assemble(HERE, pair, folds(y), y)


if __name__ == "__main__":
    main()
