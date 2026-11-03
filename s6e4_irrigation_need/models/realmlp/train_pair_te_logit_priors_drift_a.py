#!/usr/bin/env python
"""RealMLP on pairwise multiclass target encodings (logit, logit^2, logit^3), original priors, drift, digit / bin /
rare features, original rows appended (pair realmlp_pair_te_logit_priors_drift_a).

Label-free features, built once on the train + test + original rows:
- per numeric (11): frequency of the value rounded to 3 decimals, percentile rank, log1p, sqrt, 1 / (1 + x), NaN flag;
- a weighted water score, rain + previous irrigation - moisture (and its absolute value), rain / previous irrigation and
  the inverse ratio, water per field area, field area x rain, field area x previous irrigation;
- counts and indicator flags of practice values (mulching, drip / sprinkler, water sources, Kharif / Rabi, ...);
- Field_Area_hectare quartile bins and Rainfall_mm / Previous_Irrigation_mm 16-quantile bins (edges from the original
  rows), 6 categorical pairs and Crop_Type x Season x Irrigation_Type; their counts and rare flags (count <= 50);
- per-class original target rates of the 19 raw columns (numerics rounded to 3 decimals);
- drift: log1p(frequency in all rows / frequency in the original rows) of the categoricals, bins and crosses;
- for Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Humidity, Temperature_C: digits of floor(10 |x|) (up to 4),
  quantile (4, 5, 10, 20) / equal-width (5, 10, 20) / rounded bins, the value rounded to 0.5 / 0.1 as a category with a
  rare flag (< 50 rows);
- percentile ranks of Previous_Irrigation_mm against the original Low / Medium / High / all values and a z-score gap;
- per-class original target rates of the 19 raw columns again (second block, 57 columns).
Columns ending in _bin or _cat or containing "__" are categorical (strings), all others numeric (NaN / inf -> 0).
Per fold: smoothed (smoothing 20) multiclass target encodings of the 28 pairs of the 8 categoricals - out-of-fold over 5
inner folds (seed 42 + fold) for the training rows, fitted on all training rows for the validation and test rows -
mapped to logit, logit^2, logit^3 and standardized (252 columns). All 10,000 original rows are appended to every
fold's training rows (their pair-encoding columns are 0).
Model: pytabkit RealMLP_TD_Classifier, 2 x 512 mish, PBLD numeric embeddings (64 / 16, sigma 14.7), p_drop 0.42,
lr 0.068 with coslog4 schedule, n_ens 8, 30 epochs, no early stopping, random_state 42; the validation fold is passed
to fit as validation set. Test = mean of the 5 fold models.
Usage: python train_pair_te_logit_priors_drift_a.py"""
import gc
import os
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CLASSES, ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original,  # noqa: E402
                         read_test, read_train)
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_pair_te_logit_priors_drift_a"
N_CLASSES = len(CLASSES)

SEED = 42
TE_INNER_SPLITS = 5
TE_SMOOTH = 20.0
USE_LOGIT3 = True

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
ARTIFACT_NUMS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]

REALMLP_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_epochs": 30,
    "random_state": 42,
    "val_metric_name": "class_error",
    "verbosity": 2,
    "hidden_sizes": "rectangular",
    "max_one_hot_cat_size": 9,
    "embedding_size": 8,
    "weight_param": "ntk",
    "weight_init_mode": "std",
    "bias_init_mode": "he+5",
    "bias_lr_factor": 0.1,
    "act": "mish",
    "use_parametric_act": True,
    "act_lr_factor": 0.1,
    "wd": 0.013148218987344902,
    "wd_sched": "flat_cos",
    "bias_wd_factor": 0.0,
    "block_str": "w-b-a-d",
    "p_drop": 0.4201348420594983,
    "p_drop_sched": "flat_cos",
    "add_front_scale": True,
    "scale_lr_factor": 3.7236960970475845,
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
    "num_emb_type": "pbld",
    "plr_sigma": 14.70686604689513,
    "plr_hidden_1": 64,
    "plr_hidden_2": 16,
    "plr_lr_factor": 0.05152926209149046,
    "lr": 0.0682536435787459,
    "lr_sched": "coslog4",
    "opt": "adam",
    "sq_mom": 0.9940457705991664,
    "n_hidden_layers": 2,
    "hidden_width": 512,
    "first_layer_lr_factor": 0.5086979438288887,
    "ls_eps_sched": "coslog4",
    "ls_eps": 0.08315114923465,
    "use_ls": False,
    "use_early_stopping": False,
    "n_ens": 8,
}


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pctrank_against(values, reference):
    ref_sorted = np.sort(np.asarray(reference, dtype=np.float32))
    if len(ref_sorted) == 0:
        return np.zeros(len(values), dtype=np.float32)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype(np.float32)


def zscore_against(values, reference):
    ref = np.asarray(reference, dtype=np.float32)
    if len(ref) == 0:
        return np.zeros(len(values), dtype=np.float32)
    mu = np.mean(ref)
    sigma = np.std(ref)
    if sigma == 0:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - mu) / sigma).astype(np.float32)


def make_pair_key(df, f1, f2):
    return df[f1].astype(str) + "___" + df[f2].astype(str)


def smoothed_multiclass_encoding(train_keys, train_y, apply_keys, smooth=20.0):
    train_y = np.asarray(train_y, dtype=np.int64)
    prior = np.bincount(train_y, minlength=N_CLASSES).astype(np.float32)
    prior = prior / prior.sum()

    stat_df = pd.DataFrame({"key": train_keys.astype(str), "y": train_y})
    counts = pd.crosstab(stat_df["key"], stat_df["y"]).reindex(columns=range(N_CLASSES), fill_value=0)
    total = counts.sum(axis=1).astype(np.float32).values[:, None]
    enc = (counts.values.astype(np.float32) + smooth * prior[None, :]) / (total + smooth)

    enc_df = pd.DataFrame(enc, index=counts.index, columns=[f"c{i}" for i in range(N_CLASSES)])
    mapped = enc_df.reindex(apply_keys.astype(str)).fillna(pd.Series(prior, index=enc_df.columns)).to_numpy(dtype=np.float32)
    return mapped


def build_pair_te_features(Xtr, ytr, Xva, Xte, pair_cols, inner_splits=5, seed=0, smooth=20.0, use_logit3=True):
    n_tr, n_va, n_te = len(Xtr), len(Xva), len(Xte)
    n_pair = len(pair_cols)
    tr_pair = np.zeros((n_tr, n_pair * N_CLASSES), dtype=np.float32)
    va_pair = np.zeros((n_va, n_pair * N_CLASSES), dtype=np.float32)
    te_pair = np.zeros((n_te, n_pair * N_CLASSES), dtype=np.float32)
    inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    ytr_np = np.asarray(ytr, dtype=np.int64)

    for j, (f1, f2) in enumerate(pair_cols):
        tr_keys_full = make_pair_key(Xtr, f1, f2)
        va_keys = make_pair_key(Xva, f1, f2)
        te_keys = make_pair_key(Xte, f1, f2)
        tr_oof = np.zeros((n_tr, N_CLASSES), dtype=np.float32)
        for it, iv in inner_kf.split(Xtr, ytr_np):
            fit_keys = tr_keys_full.iloc[it]
            fit_y = ytr_np[it]
            val_keys = tr_keys_full.iloc[iv]
            tr_oof[iv] = smoothed_multiclass_encoding(fit_keys, fit_y, val_keys, smooth=smooth)
        start = j * N_CLASSES
        stop = start + N_CLASSES
        tr_pair[:, start:stop] = tr_oof
        va_pair[:, start:stop] = smoothed_multiclass_encoding(tr_keys_full, ytr_np, va_keys, smooth=smooth)
        te_pair[:, start:stop] = smoothed_multiclass_encoding(tr_keys_full, ytr_np, te_keys, smooth=smooth)

    if use_logit3:
        eps = 1e-5

        def logit3(mat):
            mat = np.clip(mat, eps, 1.0 - eps)
            z = np.log(mat / (1.0 - mat)).astype(np.float32)
            return np.hstack([z, z ** 2, z ** 3]).astype(np.float32)

        tr_pair = logit3(tr_pair)
        va_pair = logit3(va_pair)
        te_pair = logit3(te_pair)

    return tr_pair, va_pair, te_pair


def make_te_names(pair_cols, use_logit3=True):
    base = []
    for f1, f2 in pair_cols:
        for cls_idx, cls_name in enumerate(["low", "med", "high"]):
            base.append(f"te__{f1}__{f2}__{cls_name}")
    if not use_logit3:
        return base
    return [f"{c}__z1" for c in base] + [f"{c}__z2" for c in base] + [f"{c}__z3" for c in base]


def add_orig_prob_features(df, orig):
    priors = orig[TARGET].value_counts(normalize=True).reindex(range(N_CLASSES), fill_value=0).astype(np.float32)
    for col in BASE_CATS + BASE_NUMS:
        if col in BASE_NUMS:
            key_train = df[col].round(3)
            key_orig = orig[col].round(3)
        else:
            key_train = df[col].astype(str)
            key_orig = orig[col].astype(str)
        stats = pd.DataFrame({"key": key_orig, TARGET: orig[TARGET]})
        probs = pd.crosstab(stats["key"], stats[TARGET]).reindex(columns=range(N_CLASSES), fill_value=0).astype(np.float32)
        probs = probs.div(probs.sum(axis=1), axis=0)
        probs = probs.reindex(key_train, fill_value=np.nan)
        for cls in range(N_CLASSES):
            df[f"ORIG_proba_{col}_c{cls}"] = probs[cls].fillna(priors.loc[cls]).to_numpy(dtype=np.float32)
    return df


def add_orig_mean_by_value_features(train_df, test_df, orig_df, cols):
    priors = orig_df[TARGET].value_counts(normalize=True).reindex(range(N_CLASSES), fill_value=0).astype(np.float32)
    added = []
    for c in cols:
        if c not in train_df.columns or c not in orig_df.columns:
            continue
        key_orig = orig_df[c]
        if c in BASE_NUMS:
            key_orig = pd.to_numeric(key_orig, errors="coerce").round(3)
        else:
            key_orig = key_orig.astype(str)
        tmp = pd.DataFrame({"k": key_orig, "y": orig_df[TARGET].values})
        grp = pd.crosstab(tmp["k"], tmp["y"]).reindex(columns=range(N_CLASSES), fill_value=0).astype(np.float32)
        grp = grp.div(grp.sum(axis=1), axis=0)
        for cls in range(N_CLASSES):
            nm = f"orig_{c}_c{cls}"
            added.append(nm)
            if c in BASE_NUMS:
                ktr = pd.to_numeric(train_df[c], errors="coerce").round(3)
                kte = pd.to_numeric(test_df[c], errors="coerce").round(3)
            else:
                ktr = train_df[c].astype(str)
                kte = test_df[c].astype(str)
            train_df[nm] = ktr.map(grp[cls]).fillna(priors.loc[cls]).astype(np.float32).values
            test_df[nm] = kte.map(grp[cls]).fillna(priors.loc[cls]).astype(np.float32).values
    return train_df, test_df, added


def feature_engineering(df_all, orig):
    df = df_all.copy()
    orig = orig.copy()

    for frame in (df, orig):
        for c in BASE_NUMS:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(np.float32)
        for c in BASE_CATS:
            frame[c] = frame[c].astype(str).fillna("missing")

    for col in BASE_NUMS:
        freq = df[col].round(3).value_counts(normalize=True, dropna=False)
        df[f"FREQ_{col}"] = df[col].round(3).map(freq).fillna(0).astype(np.float32)
        df[f"RANK_{col}"] = df[col].rank(method="average", pct=True).astype(np.float32)
        v = df[col].astype(np.float32)
        df[f"LOG1P_{col}"] = np.log1p(v.clip(lower=0)).astype(np.float32)
        df[f"SQRT_{col}"] = np.sqrt(v.clip(lower=0)).astype(np.float32)
        df[f"INV1P_{col}"] = (1.0 / (1.0 + v.clip(lower=0))).astype(np.float32)
        df[f"MISS_{col}"] = v.isna().astype(np.float32)

    df["Meta"] = (
        df["Rainfall_mm"] * 0.25
        + df["Previous_Irrigation_mm"] * 0.25
        + df["Soil_Moisture"] * 0.25
        + df["Humidity"] * 0.15
        - df["Temperature_C"] * 0.10
    ).astype(np.float32)
    df["water_deviation"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"]).astype(np.float32)
    df["abs_water_deviation"] = np.abs(df["water_deviation"]).astype(np.float32)
    df["rain_to_prev_ratio"] = (df["Rainfall_mm"] / (df["Previous_Irrigation_mm"] + 1.0)).astype(np.float32)
    df["prev_to_rain_ratio"] = (df["Previous_Irrigation_mm"] / (df["Rainfall_mm"] + 1.0)).astype(np.float32)
    df["water_per_area"] = ((df["Rainfall_mm"] + df["Previous_Irrigation_mm"]) / (df["Field_Area_hectare"] + 1.0)).astype(np.float32)
    df["area_x_rain"] = (df["Field_Area_hectare"] * df["Rainfall_mm"]).astype(np.float32)
    df["area_x_prev"] = (df["Field_Area_hectare"] * df["Previous_Irrigation_mm"]).astype(np.float32)

    df["practice_yes_count"] = (
        (df["Mulching_Used"] == "Yes").astype(np.float32)
        + df["Irrigation_Type"].isin(["Drip", "Sprinkler"]).astype(np.float32)
        + df["Water_Source"].isin(["Canal", "River", "Reservoir", "Well"]).astype(np.float32)
    )
    df["practice_no_count"] = (
        (df["Mulching_Used"] == "No").astype(np.float32)
        + (df["Irrigation_Type"] == "Rainfed").astype(np.float32)
        + (df["Water_Source"] == "Rainwater").astype(np.float32)
    )
    df["practice_other_count"] = (
        (df["Irrigation_Type"] == "Canal").astype(np.float32)
        + df["Water_Source"].isin(["Reservoir", "River"]).astype(np.float32)
    )
    df["practice_count"] = df["practice_yes_count"].astype(np.float32)
    df["has_supply_water"] = df["Water_Source"].isin(["Canal", "River", "Reservoir", "Well"]).astype(np.float32)
    df["has_irrigation_system"] = df["Irrigation_Type"].isin(["Drip", "Sprinkler", "Canal"]).astype(np.float32)

    ynoflags = {
        "Mulching_Used": ["Yes", "No"],
        "Season": ["Kharif", "Rabi"],
        "Irrigation_Type": ["Drip", "Sprinkler", "Rainfed"],
        "Water_Source": ["Rainwater", "Groundwater"],
    }
    for c, vals in ynoflags.items():
        for v in vals:
            df[f"ISYES_{c}_{v}"] = (df[c] == v).astype(np.float32)
        df[f"ISOTHER_{c}"] = (~df[c].isin(vals)).astype(np.float32)

    area_bins = np.quantile(orig["Field_Area_hectare"].dropna(), [0, 0.25, 0.5, 0.75, 1.0])
    df["Field_Area_bin"] = pd.cut(df["Field_Area_hectare"], bins=np.unique(area_bins), include_lowest=True).astype(str)
    for col, name in [("Rainfall_mm", "Rainfall_q16_bin"), ("Previous_Irrigation_mm", "Previous_Irrigation_q16_bin")]:
        try:
            bins = pd.qcut(orig[col], q=16, retbins=True, duplicates="drop")[1]
            df[name] = pd.cut(df[col], bins=bins, include_lowest=True).astype(str)
        except Exception:
            df[name] = "NA"

    cross_pairs = [
        ("Crop_Type", "Season"),
        ("Irrigation_Type", "Water_Source"),
        ("Crop_Type", "Region"),
        ("Season", "Crop_Growth_Stage"),
        ("Water_Source", "Region"),
        ("Crop_Type", "Irrigation_Type"),
    ]
    for a, b in cross_pairs:
        df[f"{a}__{b}"] = df[a].astype(str) + "|" + df[b].astype(str)
    df["Crop_Type__Season__Irrigation_Type"] = (
        df["Crop_Type"].astype(str) + "|" + df["Season"].astype(str) + "|" + df["Irrigation_Type"].astype(str)
    )

    bin_keys = ["Field_Area_bin", "Rainfall_q16_bin", "Previous_Irrigation_q16_bin"]
    all_for_counts = BASE_CATS + bin_keys + [f"{a}__{b}" for a, b in cross_pairs] + ["Crop_Type__Season__Irrigation_Type"]
    for c in all_for_counts:
        vc = df[c].astype(str).value_counts(dropna=False)
        df[f"CAT_CNT_{c}"] = df[c].astype(str).map(vc).fillna(0).astype(np.float32)
        df[f"CAT_RARE_{c}"] = (df[f"CAT_CNT_{c}"] <= 50).astype(np.float32)

    df = add_orig_prob_features(df, orig)

    drift_base = BASE_CATS + bin_keys + [f"{a}__{b}" for a, b in cross_pairs] + ["Crop_Type__Season__Irrigation_Type"]
    for c in drift_base:
        p_df = df[c].astype(str).value_counts(normalize=True)
        p_or = orig[c].astype(str).value_counts(normalize=True) if c in orig.columns else pd.Series(dtype=float)
        df[f"DRIFT_{c}"] = np.log1p(df[c].astype(str).map(p_df).fillna(0) / (df[c].astype(str).map(p_or).fillna(1e-6) + 1e-6)).astype(np.float32)

    for col in ARTIFACT_NUMS:
        v = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[f"{col}__is_nan"] = v.isna().astype(np.int8)
        scaled = np.floor(np.abs(v) * 10).astype(np.int64)
        max_val = int(scaled.max()) if len(scaled) else 0
        max_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
        for i in range(min(max_digits, 4)):
            df[f"{col}_digit_pos_{i+1}"] = ((scaled // (10 ** i)) % 10).astype(np.int8)
        df[f"{col}__neg"] = (scaled < 0).astype(np.int8)

    bin_specs = {
        "Field_Area_hectare": (1.0, 20),
        "Rainfall_mm": (0.5, 20),
        "Previous_Irrigation_mm": (0.5, 20),
    }
    for col in ARTIFACT_NUMS:
        x = pd.to_numeric(df[col], errors="coerce")
        for q in [4, 5, 10, 20]:
            try:
                df[f"{col}_bin_q{q}"] = pd.qcut(x, q=q, labels=False, duplicates="drop").astype(str)
            except Exception:
                df[f"{col}_bin_q{q}"] = "NA"
        for b in [5, 10, 20]:
            df[f"{col}_bin_cut{b}"] = pd.cut(x, bins=b, labels=False).astype(str)
        step = bin_specs.get(col, (0.1, 0))[0]
        df[f"{col}_bin_round_{str(step).replace('.', 'p')}"] = ((x / step).round() * step).fillna(-9999).astype(str)

    for col, step in {
        "Rainfall_mm": 0.5,
        "Previous_Irrigation_mm": 0.5,
        "Soil_Moisture": 0.1,
        "Humidity": 0.1,
        "Temperature_C": 0.1,
    }.items():
        rounded = ((df[col] / step).round() * step).astype(np.float32)
        df[f"{col}_cat"] = rounded.fillna(-9999).astype(str)
        vc = df[f"{col}_cat"].value_counts(dropna=False)
        df[f"{col}_cat__is_rare"] = (df[f"{col}_cat"].map(vc).fillna(0) < 50).astype(np.int8)

    orig_low = orig.loc[orig[TARGET] == 0, "Previous_Irrigation_mm"].to_numpy(dtype=np.float32)
    orig_med = orig.loc[orig[TARGET] == 1, "Previous_Irrigation_mm"].to_numpy(dtype=np.float32)
    orig_high = orig.loc[orig[TARGET] == 2, "Previous_Irrigation_mm"].to_numpy(dtype=np.float32)
    orig_all = orig["Previous_Irrigation_mm"].to_numpy(dtype=np.float32)
    vals = df["Previous_Irrigation_mm"].to_numpy(dtype=np.float32)
    df["_pctrank_low_prev"] = pctrank_against(vals, orig_low)
    df["_pctrank_med_prev"] = pctrank_against(vals, orig_med)
    df["_pctrank_high_prev"] = pctrank_against(vals, orig_high)
    df["_pctrank_orig_prev"] = pctrank_against(vals, orig_all)
    df["_zscore_high_gap_prev"] = (np.abs(zscore_against(vals, orig_high)) - np.abs(zscore_against(vals, orig_low))).astype(np.float32)

    new_cat_cols = [c for c in df.columns if c.endswith("_bin") or "__" in c or c.endswith("_cat")]
    new_num_cols = [c for c in df.columns if c not in BASE_CATS and c not in new_cat_cols]
    return df, new_cat_cols, new_num_cols


def main():
    print("Class order:", list(CLASSES))

    train = read_train()
    test = read_test()
    orig = read_original()

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(np.int64)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(np.int64)

    X = train.drop(columns=[ID, TARGET]).copy()
    y = train[TARGET].copy()
    X_test = test.drop(columns=[ID]).copy()
    X_orig = orig.drop(columns=[TARGET]).copy()
    y_orig = orig[TARGET].copy()

    df_all = pd.concat([X, X_test, X_orig], axis=0, ignore_index=True)
    print("combined rows:", len(df_all))

    df_fe, new_cat_cols, new_num_cols = feature_engineering(df_all, orig)
    n_train = len(X)
    n_test = len(X_test)
    X = df_fe.iloc[:n_train].copy().reset_index(drop=True)
    X_test = df_fe.iloc[n_train:n_train + n_test].copy().reset_index(drop=True)
    X_orig = df_fe.iloc[n_train + n_test:].copy().reset_index(drop=True)

    base_cat_set = set(BASE_CATS)
    cat_cols = list(BASE_CATS) + [c for c in new_cat_cols if c not in base_cat_set]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__")
        X_orig[c] = X_orig[c].astype("string").fillna("__MISSING__")
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.float32)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0).astype(np.float32)
        X_orig[c] = pd.to_numeric(X_orig[c], errors="coerce").fillna(0).astype(np.float32)

    orig_with_y = pd.concat([X_orig, y_orig], axis=1)
    X, X_test, orig_mean_cols = add_orig_mean_by_value_features(X, X_test, orig_with_y, cols=BASE_CATS + BASE_NUMS)
    X_orig, _, _ = add_orig_mean_by_value_features(X_orig, X_test.iloc[:0].copy(), orig_with_y, cols=BASE_CATS + BASE_NUMS)
    print(f"Added orig-mean features: {len(orig_mean_cols)}")

    # RealMLP rejects NaN / inf in continuous features: sanitize all non-categorical columns again
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        X_orig[c] = pd.to_numeric(X_orig[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    pair_cols = list(combinations(BASE_CATS, 2))
    te_col_names = make_te_names(pair_cols, use_logit3=USE_LOGIT3)

    print("num cat cols:", len(cat_cols))
    print("num num cols:", len(num_cols))
    print("num TE pairs:", len(pair_cols))

    fold_id = folds(y.values)
    oof_preds = np.zeros((len(X), N_CLASSES), dtype=np.float32)
    test_preds = np.zeros((len(X_test), N_CLASSES), dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        t_idx, v_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print("\n" + "=" * 80)
        print(f"fold {fold}")
        print("=" * 80)

        Xt = X.iloc[t_idx].copy().reset_index(drop=True)
        Xv = X.iloc[v_idx].copy().reset_index(drop=True)
        yt = y.iloc[t_idx].copy().reset_index(drop=True)
        yv = y.iloc[v_idx].copy().reset_index(drop=True)

        tr_te, va_te, te_te = build_pair_te_features(
            Xt,
            yt.values,
            Xv,
            X_test,
            pair_cols=pair_cols,
            inner_splits=TE_INNER_SPLITS,
            seed=SEED + fold,
            smooth=TE_SMOOTH,
            use_logit3=USE_LOGIT3,
        )

        scaler_te = StandardScaler()
        tr_te = scaler_te.fit_transform(tr_te).astype(np.float32)
        va_te = scaler_te.transform(va_te).astype(np.float32)
        te_te = scaler_te.transform(te_te).astype(np.float32)

        tr_te_df = pd.DataFrame(tr_te, columns=te_col_names)
        va_te_df = pd.DataFrame(va_te, columns=te_col_names)
        te_te_df = pd.DataFrame(te_te, columns=te_col_names)

        Xt_model = pd.concat([Xt.reset_index(drop=True), tr_te_df], axis=1)
        Xv_model = pd.concat([Xv.reset_index(drop=True), va_te_df], axis=1)
        Xte_model = pd.concat([X_test.reset_index(drop=True), te_te_df], axis=1)

        # the original rows (pair-encoding columns 0) are appended to the fold's training rows
        Xorig_model = X_orig.copy().reset_index(drop=True)
        orig_zero_te = pd.DataFrame(np.zeros((len(Xorig_model), len(te_col_names)), dtype=np.float32), columns=te_col_names)
        Xorig_model = pd.concat([Xorig_model, orig_zero_te], axis=1)
        yorig_model = y_orig.reset_index(drop=True)
        Xt_model = pd.concat([Xt_model, Xorig_model], axis=0, ignore_index=True)
        yt_model = pd.concat([yt, yorig_model], axis=0, ignore_index=True)

        # align the validation / test tables to the training table's columns
        for c in Xt_model.columns:
            if c not in Xv_model.columns:
                Xv_model[c] = 0 if c not in cat_cols else "__MISSING__"
            if c not in Xte_model.columns:
                Xte_model[c] = 0 if c not in cat_cols else "__MISSING__"
        Xv_model = Xv_model[Xt_model.columns]
        Xte_model = Xte_model[Xt_model.columns]

        for c in cat_cols:
            Xt_model[c] = Xt_model[c].astype("string")
            Xv_model[c] = Xv_model[c].astype("string")
            Xte_model[c] = Xte_model[c].astype("string")
        for c in te_col_names:
            Xt_model[c] = pd.to_numeric(Xt_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            Xv_model[c] = pd.to_numeric(Xv_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            Xte_model[c] = pd.to_numeric(Xte_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        non_cat_cols = [c for c in Xt_model.columns if c not in cat_cols]
        for c in non_cat_cols:
            Xt_model[c] = pd.to_numeric(Xt_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            Xv_model[c] = pd.to_numeric(Xv_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            Xte_model[c] = pd.to_numeric(Xte_model[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(Xt_model, yt_model, X_val=Xv_model, y_val=yv, cat_col_names=cat_cols)

        val_preds = model.predict_proba(Xv_model).astype(np.float32)
        fold_test_preds = model.predict_proba(Xte_model).astype(np.float32)

        oof_preds[v_idx] = val_preds
        test_preds += fold_test_preds / N_FOLDS

        fold_score = balanced_accuracy_score(yv, val_preds.argmax(axis=1))
        print(f"fold {fold} balanced accuracy: {fold_score:.6f}")

        del model, Xt_model, Xv_model, Xte_model, Xorig_model
        del tr_te, va_te, te_te, tr_te_df, va_te_df, te_te_df, orig_zero_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_score = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
    print("\nOverall OOF balanced accuracy: %.6f" % oof_score)
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
