#!/usr/bin/env python
"""GPU XGBoost on snapped values, quantile / round / digit keys, original-data priors, drift ratios and nested 3-class
target encodings (pair xgb_snap_priors_drift_te_a).

Label-free features (train / test / original rows):
- snap: every train / test value of Rainfall_mm and Previous_Irrigation_mm (as a string) that does not occur in the
  original data is replaced by the nearest original value (Rainfall_snap, Previous_Irrigation_snap), plus the difference;
- the 3-decimal fractional part of rain / previous irrigation / soil moisture and its first two digits; management
  counts (mulching, drip / sprinkler, water source);
- 16- and 32-quantile bins of rain / previous irrigation (edges from the train rows); rounding to 0 / 1 decimals of rain /
  previous irrigation / soil moisture; digits at positions 10^-2 .. 10^2 of the same 3 columns;
- counts (train + test + original) and rare flags (<= 50) of the categoricals, quantile bins, snapped values and first
  decimal digits; for the categoricals, quantile bins and snapped values: seen-in-original flags, original class rates
  and counts; original class rates, counts and seen flags of 4 category pairs; log drift ratio (train frequency /
  original frequency).
Per fold: 25 keys (quantile bins, digits, snapped values, the 4 category pairs) get nested 3-class target encodings
(sklearn TargetEncoder, cv 2, smooth 'auto', one binary encoder per class; out-of-fold over an inner StratifiedKFold(5,
shuffle, 42) of the training rows, fitted on all training rows for the validation / test rows), the keys are dropped,
every string column is integer-coded over the training + validation + test frames and the columns are sorted by name.
Balanced class weights.
Model: XGBoost hist on GPU, lr 0.02, depth 6, subsample 0.85, colsample_bytree 0.55, min_child_weight 5, alpha 0.25,
lambda 3, max_bin 4096, up to 50,000 rounds, early stopping 300 (mlogloss) on the validation fold, random_state 42.
Test = mean of the 5 fold models. Competition rows only (the original rows only feed the statistics).
The feature build takes about 45 minutes on CPU (mostly the snap step); each fold about 4.5 minutes on one GPU.
Usage: python train_snap_priors_drift_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "xgb_snap_priors_drift_te_a"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
N_THREADS = 32

LABELS = list(CLASSES)
MISSING = "<NA>"

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
PAIR_PRIOR_COLS = [
    ("Crop_Type", "Season"),
    ("Irrigation_Type", "Water_Source"),
    ("Crop_Type", "Irrigation_Type"),
    ("Soil_Type", "Crop_Type"),
]

ROUND_SPECS = {
    "Rainfall_mm": [0, 1],
    "Previous_Irrigation_mm": [0, 1],
    "Soil_Moisture": [0, 1],
}
QCOLS = ["Rainfall_mm", "Previous_Irrigation_mm"]
DIGIT_COLS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"]
RARE_MIN_COUNT = 50

XGB_PARAMS = dict(
    n_estimators=50_000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.55,
    min_child_weight=5,
    gamma=0.0,
    reg_alpha=0.25,
    reg_lambda=3.0,
    max_bin=4096,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=False,
    device="cuda",
    random_state=SEED,
    n_jobs=N_THREADS,
    early_stopping_rounds=300,
    verbosity=0,
)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_target_codes(s: pd.Series) -> np.ndarray:
    ser = pd.Series(s)
    if pd.api.types.is_numeric_dtype(ser):
        return pd.to_numeric(ser, errors="coerce").fillna(0).astype("int8").values
    return ser.map(TARGET_MAP).astype("int8").values


def safe_str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna(MISSING).str.strip()


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype("float32")
            med = float(np.nanmedian(s.values)) if np.isfinite(np.nanmedian(s.values)) else 0.0
            df[c] = s.fillna(med).astype("float32")


def add_decimal_parts(df: pd.DataFrame):
    mapping = [
        ("Rainfall_mm", "rain"),
        ("Previous_Irrigation_mm", "prev"),
        ("Soil_Moisture", "moist"),
    ]
    for src, prefix in mapping:
        if src not in df.columns:
            continue
        ss = df[src].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "nan")
        frac = ss.str.split(".").str[1].fillna("")
        df[prefix] = frac
        df[f"{prefix}0"] = frac.str[0].fillna("0")
        df[f"{prefix}1"] = frac.str[1].fillna("0")


def build_snap_map(combo_s: pd.Series, orig_s: pd.Series):
    combo_unique = pd.Series(safe_str(combo_s).dropna().unique(), dtype="string")
    orig_unique = pd.Series(safe_str(orig_s).dropna().unique(), dtype="string")
    rare = combo_unique[~combo_unique.isin(orig_unique)]
    if len(rare) == 0:
        return {}
    rare_num = pd.to_numeric(rare, errors="coerce")
    keep = rare_num.notna()
    rare = rare[keep].reset_index(drop=True)
    rare_num = rare_num[keep].to_numpy()
    orig_num = pd.to_numeric(orig_unique, errors="coerce")
    keep = orig_num.notna()
    orig_unique = orig_unique[keep].reset_index(drop=True)
    orig_num = orig_num[keep].to_numpy()
    if len(orig_num) == 0 or len(rare_num) == 0:
        return {}
    order = np.argsort(orig_num)
    orig_num = orig_num[order]
    orig_unique = orig_unique.iloc[order].values
    mp = {}
    pos = np.searchsorted(orig_num, rare_num)
    for s_val, val, p in zip(rare.values, rare_num, pos):
        if p <= 0:
            snap = orig_unique[0]
        elif p >= len(orig_num):
            snap = orig_unique[-1]
        else:
            left, right = orig_num[p - 1], orig_num[p]
            snap = orig_unique[p - 1] if abs(val - left) <= abs(val - right) else orig_unique[p]
        mp[str(s_val)] = str(snap)
    return mp


def add_snap_features(train_df, test_df, orig_df):
    combo = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for col, snap_name in [("Rainfall_mm", "Rainfall_snap"), ("Previous_Irrigation_mm", "Previous_Irrigation_snap")]:
        combo[snap_name] = safe_str(combo[col])
        mp = build_snap_map(combo[col], orig_df[col])
        if mp:
            combo[snap_name] = combo[snap_name].replace(mp)
        orig_num = pd.to_numeric(combo[col], errors="coerce")
        snap_num = pd.to_numeric(combo[snap_name], errors="coerce")
        combo[f"{snap_name}_diff"] = (snap_num - orig_num).astype("float32")
    n_train = len(train_df)
    train_df = combo.iloc[:n_train].reset_index(drop=True)
    test_df = combo.iloc[n_train:].reset_index(drop=True)
    orig_df["Rainfall_snap"] = safe_str(orig_df["Rainfall_mm"])
    orig_df["Previous_Irrigation_snap"] = safe_str(orig_df["Previous_Irrigation_mm"])
    orig_df["Rainfall_snap_diff"] = np.zeros(len(orig_df), dtype=np.float32)
    orig_df["Previous_Irrigation_snap_diff"] = np.zeros(len(orig_df), dtype=np.float32)
    ensure_numeric(train_df, ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])
    ensure_numeric(test_df, ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])
    ensure_numeric(orig_df, ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])
    return train_df, test_df, orig_df


def add_management_counts(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    ir = safe_str(df["Irrigation_Type"])
    ws = safe_str(df["Water_Source"])
    mu = safe_str(df["Mulching_Used"])
    out["service_yes_count"] = (
        (mu == "Yes").astype("float32")
        + ir.isin(["Drip", "Sprinkler"]).astype("float32")
        + (~ws.isin(["Rainwater"])).astype("float32")
    )
    out["service_no_count"] = (
        (mu == "No").astype("float32")
        + (ir == "Rainfed").astype("float32")
        + (ws == "Rainwater").astype("float32")
    )
    out["service_other_count"] = (
        (ir == "Canal").astype("float32")
        + ws.isin(["Reservoir", "River"]).astype("float32")
    )
    out["service_count"] = out["service_yes_count"].astype("float32")
    return out


def add_qcut_bins(train_df, test_df, orig_df):
    made = []
    for c in QCOLS:
        for q in [16, 32]:
            name = f"{c}_qcut_{q}"
            try:
                train_s = pd.to_numeric(train_df[c], errors="coerce")
                _, bins = pd.qcut(train_s, q=q, labels=False, retbins=True, duplicates="drop")
                train_df[name] = pd.cut(train_s, bins=bins, labels=False, include_lowest=True)
                test_df[name] = pd.cut(pd.to_numeric(test_df[c], errors="coerce"), bins=bins, labels=False, include_lowest=True)
                orig_df[name] = pd.cut(pd.to_numeric(orig_df[c], errors="coerce"), bins=bins, labels=False, include_lowest=True)
            except Exception:
                train_df[name] = 0
                test_df[name] = 0
                orig_df[name] = 0
            for df in (train_df, test_df, orig_df):
                df[name] = safe_str(df[name])
            made.append(name)
    return train_df, test_df, orig_df, made


def add_round_features(train_df, test_df, orig_df):
    made = []
    for c, rr in ROUND_SPECS.items():
        for r in rr:
            name = f"{c}_round_{r}"
            for df in (train_df, test_df, orig_df):
                df[name] = pd.to_numeric(df[c], errors="coerce").round(r).astype("float32")
            made.append(name)
    return train_df, test_df, orig_df, made


def add_digit_features(train_df, test_df, orig_df):
    made = []
    combo = pd.concat([train_df[DIGIT_COLS], test_df[DIGIT_COLS], orig_df[DIGIT_COLS]], axis=0, ignore_index=True)
    for c in DIGIT_COLS:
        vals = pd.to_numeric(combo[c], errors="coerce").fillna(0)
        for k in range(-2, 3):
            name = f"{c}_digit_{k}"
            x = (((vals * (10 ** k)) % 10).round()).astype("int16")
            train_df[name] = x.iloc[: len(train_df)].values
            test_df[name] = x.iloc[len(train_df): len(train_df) + len(test_df)].values
            orig_df[name] = x.iloc[len(train_df) + len(test_df):].values
            made.append(name)
    return train_df, test_df, orig_df, made


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        vc = safe_str(all_cat[c]).value_counts(dropna=False)
        tr_cnt = pd.to_numeric(safe_str(train_df[c]).map(vc), errors="coerce").fillna(0).astype("float32")
        te_cnt = pd.to_numeric(safe_str(test_df[c]).map(vc), errors="coerce").fillna(0).astype("float32")
        or_cnt = pd.to_numeric(safe_str(orig_df[c]).map(vc), errors="coerce").fillna(0).astype("float32")
        add_tr[f"CAT_CNT_{c}"] = tr_cnt.values
        add_te[f"CAT_CNT_{c}"] = te_cnt.values
        add_or[f"CAT_CNT_{c}"] = or_cnt.values
        add_tr[f"CAT_RARE_{c}"] = (tr_cnt <= RARE_MIN_COUNT).astype("float32").values
        add_te[f"CAT_RARE_{c}"] = (te_cnt <= RARE_MIN_COUNT).astype("float32").values
        add_or[f"CAT_RARE_{c}"] = (or_cnt <= RARE_MIN_COUNT).astype("float32").values
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def make_pair_key(df, c1, c2):
    return safe_str(df[c1]) + "|" + safe_str(df[c2])


def add_orig_membership(df, orig, cols):
    out = {}
    for c in cols:
        if c in df.columns and c in orig.columns:
            orig_vals = set(safe_str(orig[c]).unique())
            out[f"{c}_in_orig"] = safe_str(df[c]).isin(orig_vals).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_orig_priors(df, orig, cols):
    out = {}
    y_orig = to_target_codes(orig[TARGET])
    priors = np.bincount(y_orig, minlength=3).astype(np.float32) / len(y_orig)
    for c in cols:
        if c not in df.columns or c not in orig.columns:
            continue
        grp = pd.DataFrame({"k": safe_str(orig[c]), "y": y_orig}).groupby("k")["y"]
        cnt = grp.size()
        for cls in range(3):
            mean_map = grp.apply(lambda s, kk=cls: (s == kk).mean())
            out[f"orig_mean_{c}_c{cls}"] = safe_str(df[c]).map(mean_map).fillna(priors[cls]).astype("float32")
        out[f"orig_count_{c}"] = safe_str(df[c]).map(cnt).fillna(0).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_orig_pair_priors(df, orig, pairs):
    out = {}
    y_orig = to_target_codes(orig[TARGET])
    priors = np.bincount(y_orig, minlength=3).astype(np.float32) / len(y_orig)
    for c1, c2 in pairs:
        if not all(c in df.columns for c in (c1, c2)) or not all(c in orig.columns for c in (c1, c2)):
            continue
        name = f"{c1}__{c2}"
        orig_key = make_pair_key(orig, c1, c2)
        tmp = pd.DataFrame({"k": orig_key, "y": y_orig})
        grp = tmp.groupby("k")["y"]
        cnt = grp.size()
        df_key = make_pair_key(df, c1, c2)
        for cls in range(3):
            mean_map = grp.apply(lambda s, kk=cls: (s == kk).mean())
            out[f"orig_mean_{name}_c{cls}"] = df_key.map(mean_map).fillna(priors[cls]).astype("float32")
        out[f"orig_count_{name}"] = df_key.map(cnt).fillna(0).astype("float32")
        out[f"orig_seen_{name}"] = df_key.isin(set(cnt.index)).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    add_tr, add_te = {}, {}
    for c in cols:
        if c not in train_df.columns or c not in orig_df.columns:
            continue
        p_tr = safe_str(train_df[c]).value_counts(normalize=True, dropna=False)
        p_or = safe_str(orig_df[c]).value_counts(normalize=True, dropna=False)
        name = f"drift_ratio_{c}"
        tr_num = safe_str(train_df[c]).map(p_tr).fillna(0)
        te_num = safe_str(test_df[c]).map(p_tr).fillna(0)
        tr_den = safe_str(train_df[c]).map(p_or).fillna(eps) + eps
        te_den = safe_str(test_df[c]).map(p_or).fillna(eps) + eps
        add_tr[name] = np.log1p(tr_num / tr_den).astype("float32")
        add_te[name] = np.log1p(te_num / te_den).astype("float32")
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df


def _te_to_float_array(x, col_name):
    if isinstance(x, pd.DataFrame):
        return x[col_name].to_numpy(dtype=np.float32, copy=False)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float32, copy=False)
    arr = np.asarray(x, dtype=np.float32)
    return arr[:, 0] if arr.ndim == 2 else arr.reshape(-1)


def target_encode_nested(X_tr, y_tr, X_va, X_te, te_cols):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()
    made_cols = []
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for c in te_cols:
        oof = np.zeros((len(X_tr), 3), dtype=np.float32)
        tr_col = safe_str(X_tr[c])
        va_col = safe_str(X_va[c])
        te_col = safe_str(X_te[c])
        for itr, iva in inner.split(tr_col, y_tr):
            X_itr = tr_col.iloc[itr].to_frame(c)
            X_iva = tr_col.iloc[iva].to_frame(c)
            for cls in range(3):
                enc = TargetEncoder(cv=2, smooth="auto", target_type="binary")
                enc.fit(X_itr, (y_tr[itr] == cls).astype(np.int8))
                oof[iva, cls] = _te_to_float_array(enc.transform(X_iva), c)
        for cls in range(3):
            enc = TargetEncoder(cv=2, smooth="auto", target_type="binary")
            enc.fit(tr_col.to_frame(c), (y_tr == cls).astype(np.int8))
            va_enc = _te_to_float_array(enc.transform(va_col.to_frame(c)), c)
            te_enc = _te_to_float_array(enc.transform(te_col.to_frame(c)), c)
            newc = f"{c}_TE_c{cls}"
            X_tr[newc] = oof[:, cls]
            X_va[newc] = va_enc
            X_te[newc] = te_enc
            made_cols.append(newc)
    return X_tr, X_va, X_te, made_cols


def factorize_train_valid_test(tr, va, te):
    tr = tr.copy()
    va = va.copy()
    te = te.copy()
    all_cols = sorted(set(tr.columns) | set(va.columns) | set(te.columns))
    tr = tr.reindex(columns=all_cols)
    va = va.reindex(columns=all_cols)
    te = te.reindex(columns=all_cols)
    for c in all_cols:
        comb = pd.concat([tr[c], va[c], te[c]], axis=0, ignore_index=True)
        if str(comb.dtype).startswith(("object", "string", "category")):
            codes, _ = pd.factorize(safe_str(comb), sort=False)
            codes = codes.astype("int32") + 1
            tr[c] = codes[: len(tr)]
            va[c] = codes[len(tr): len(tr) + len(va)]
            te[c] = codes[len(tr) + len(va):]
        else:
            tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0).astype("float32")
            va[c] = pd.to_numeric(va[c], errors="coerce").fillna(0).astype("float32")
            te[c] = pd.to_numeric(te[c], errors="coerce").fillna(0).astype("float32")
    return tr, va, te


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()

    train = train.drop(columns=[ID], errors="ignore")
    test = test.drop(columns=[ID], errors="ignore")
    add_decimal_parts(train)
    add_decimal_parts(test)
    add_decimal_parts(orig)
    ensure_numeric(train, BASE_NUMS)
    ensure_numeric(test, BASE_NUMS)
    ensure_numeric(orig, BASE_NUMS)

    train, test, orig = add_snap_features(train, test, orig)

    y = to_target_codes(train[TARGET])
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype("int8")

    for df in (train, test, orig):
        for c in BASE_CATS + ["rain", "rain0", "rain1", "prev", "prev0", "prev1", "moist", "moist0", "moist1", "Rainfall_snap", "Previous_Irrigation_snap"]:
            if c in df.columns:
                df[c] = safe_str(df[c])
        ensure_numeric(df, BASE_NUMS + ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])

    log("Adding management count features...")
    train = pd.concat([train, add_management_counts(train)], axis=1)
    test = pd.concat([test, add_management_counts(test)], axis=1)
    orig = pd.concat([orig, add_management_counts(orig)], axis=1)

    log("Adding qcut / round / digit features...")
    train, test, orig, qcut_cols = add_qcut_bins(train, test, orig)
    train, test, orig, round_cols = add_round_features(train, test, orig)
    train, test, orig, digit_cols = add_digit_features(train, test, orig)

    log("Adding categorical count / rare features...")
    cat_cols = BASE_CATS + qcut_cols + ["Rainfall_snap", "Previous_Irrigation_snap", "rain0", "prev0", "moist0"]
    cat_cols = [c for c in cat_cols if c in train.columns]
    train, test, orig = add_cat_counts_and_rare(train, test, orig, cat_cols)

    log("Adding orig membership / priors...")
    orig_prior_cols = BASE_CATS + qcut_cols + ["Rainfall_snap", "Previous_Irrigation_snap"]
    orig_prior_cols = [c for c in orig_prior_cols if c in train.columns and c in orig.columns]
    train = pd.concat([train, add_orig_membership(train, orig, orig_prior_cols)], axis=1)
    test = pd.concat([test, add_orig_membership(test, orig, orig_prior_cols)], axis=1)
    train = pd.concat([train, add_orig_priors(train, orig, orig_prior_cols)], axis=1)
    test = pd.concat([test, add_orig_priors(test, orig, orig_prior_cols)], axis=1)

    log("Adding orig pair priors...")
    train = pd.concat([train, add_orig_pair_priors(train, orig, PAIR_PRIOR_COLS)], axis=1)
    test = pd.concat([test, add_orig_pair_priors(test, orig, PAIR_PRIOR_COLS)], axis=1)

    log("Adding drift ratio features...")
    drift_cols = [c for c in BASE_CATS + qcut_cols + ["Rainfall_snap", "Previous_Irrigation_snap"] if c in train.columns]
    train, test = add_drift_ratio(train, test, orig, drift_cols)

    te_cols = []
    for c in qcut_cols + digit_cols + ["Rainfall_snap", "Previous_Irrigation_snap"]:
        if c in train.columns:
            te_cols.append(c)
    for c1, c2 in PAIR_PRIOR_COLS:
        name = f"{c1}__{c2}_te"
        train[name] = make_pair_key(train, c1, c2)
        test[name] = make_pair_key(test, c1, c2)
        te_cols.append(name)
    te_cols = sorted(set(te_cols))
    log(f"n_te_candidates= {len(te_cols)}")

    X_train = train.drop(columns=[TARGET]).copy()
    X_test = test.copy()
    return X_train, X_test, y, te_cols


def run_fold(fold, tr_idx, va_idx, X_train, X_test, y, te_cols):
    t0 = time.perf_counter()
    class_counts = np.bincount(y, minlength=3).astype(np.float32)
    class_weights = class_counts.sum() / (len(LABELS) * class_counts)

    X_tr = X_train.iloc[tr_idx].reset_index(drop=True).copy()
    X_va = X_train.iloc[va_idx].reset_index(drop=True).copy()
    X_te = X_test.reset_index(drop=True).copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    X_tr, X_va, X_te, te_made = target_encode_nested(X_tr, y_tr, X_va, X_te, te_cols)
    base_drop = [c for c in te_cols if c in X_tr.columns]
    X_tr = X_tr.drop(columns=base_drop, errors="ignore")
    X_va = X_va.drop(columns=base_drop, errors="ignore")
    X_te = X_te.drop(columns=base_drop, errors="ignore")

    X_tr, X_va, X_te = factorize_train_valid_test(X_tr, X_va, X_te)
    sw = np.array([class_weights[yy] for yy in y_tr], dtype=np.float32)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    p_va = model.predict_proba(X_va).astype(np.float32)
    p_te = model.predict_proba(X_te).astype(np.float32)
    bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
    log(f"{PAIR} fold {fold}: BAC {bac:.6f} | best iteration {getattr(model, 'best_iteration', None)} | "
        f"time={time.perf_counter() - t0:.1f}s")

    del X_tr, X_va, X_te, y_tr, y_va, model
    gc.collect()
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X_train, X_test, y, te_cols = build_features()
        fold_id = folds(y)
        for k in todo:
            p_va, p_te = run_fold(k, np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k), X_train, X_test, y, te_cols)
            save_part(HERE, PAIR, k, p_va, p_te)
    y = to_target_codes(read_train()[TARGET])
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
