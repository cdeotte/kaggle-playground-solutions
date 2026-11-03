#!/usr/bin/env python
"""RealMLP on digit categories, rounded values, bins, categorical crosses, global rank / frequency and original
priors, original rows appended (pair realmlp_digit_cats_crosses_rank_priors_a).

Features (built on the train, test and original rows; numeric NaNs -> the frame's median):
- the 19 raw columns; 9 row features (water total / gap / per area, moisture - humidity, sunlight x temperature,
  conductivity / pH, organic carbon x moisture, wind / sunlight, field area squared);
- for Soil_Moisture, Humidity, Rainfall_mm, Previous_Irrigation_mm, Temperature_C, Field_Area_hectare: last digit,
  integer part mod 100, the two decimals and the value rounded to 1 decimal, as categories;
- every numeric rounded to its step (ROUND_MAP) as a category;
- quantile bins (edges over train + test + original) of Rainfall_mm (80, 150), Previous_Irrigation_mm (60, 120),
  Soil_Moisture (60), Humidity (60), as categories;
- 5 categorical pairs and Crop_Type x Season x Irrigation_Type;
- frequency and percentile rank of the 11 numerics over train + test + original;
- per-class original target rate of the 8 categoricals (24 columns).
Training rows of fold k: the competition rows of the 4 other folds followed by all 10,000 original rows.
Model: pytabkit RealMLP_TD_Classifier, silu [512, 256, 128], numeric embeddings (plr_hidden 16 / 8, gelu), n_ens 8,
lr 0.075 with flat_anneal schedule, batch 256, 3 epochs, early stopping on the validation fold, random_state 42.
Test = mean of the 5 fold models.
Usage: python train_digit_cats_crosses_rank_priors_a.py"""
import gc
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CLASSES, ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original,  # noqa: E402
                         read_test, read_train)
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_digit_cats_crosses_rank_priors_a"
SEED = 42

BASE_CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
BASE_NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
NUM_DIGIT_COLS = ["Soil_Moisture", "Humidity", "Rainfall_mm", "Previous_Irrigation_mm", "Temperature_C", "Field_Area_hectare"]
ROUND_MAP = {
    "Soil_pH": 0.1,
    "Soil_Moisture": 0.1,
    "Organic_Carbon": 0.1,
    "Electrical_Conductivity": 0.1,
    "Temperature_C": 0.1,
    "Humidity": 0.1,
    "Rainfall_mm": 0.5,
    "Sunlight_Hours": 0.1,
    "Wind_Speed_kmh": 0.1,
    "Field_Area_hectare": 0.1,
    "Previous_Irrigation_mm": 0.5,
}
CROSS_PAIRS = [
    ("Crop_Type", "Season"),
    ("Crop_Type", "Irrigation_Type"),
    ("Irrigation_Type", "Water_Source"),
    ("Soil_Type", "Region"),
    ("Crop_Growth_Stage", "Season"),
]
TRIPLE = [("Crop_Type", "Season", "Irrigation_Type")]
BIN_CONFIG = {
    "Rainfall_mm": [80, 150],
    "Previous_Irrigation_mm": [60, 120],
    "Soil_Moisture": [60],
    "Humidity": [60],
}

params = {
    "random_state": 42,
    "verbosity": 2,
    "n_ens": 8,
    "n_epochs": 3,
    "batch_size": 256,
    "use_early_stopping": True,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "lr": 0.075,
    "wd": 0.0236,
    "sq_mom": 0.988,
    "lr_sched": "flat_anneal",
    "first_layer_lr_factor": 0.25,
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "expm4t",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "tfms": [
        "one_hot",
        "median_center",
        "robust_scale",
        "smooth_clip",
        "embedding",
        "l2_normalize",
    ],
}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def safe_str_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("missing").str.strip()


def round_to_step(x, step):
    x = pd.to_numeric(x, errors="coerce")
    return (x / step).round() * step


def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    add_tr, add_te, add_or = {}, {}, {}
    for c in num_cols:
        vals = pd.to_numeric(all_num[c], errors="coerce").fillna(0)
        freq = vals.value_counts(normalize=True, dropna=False)
        tr_vals = pd.to_numeric(train_df[c], errors="coerce").fillna(0)
        te_vals = pd.to_numeric(test_df[c], errors="coerce").fillna(0)
        or_vals = pd.to_numeric(orig_df[c], errors="coerce").fillna(0)
        add_tr[f"FREQ_{c}"] = tr_vals.map(freq).fillna(0).astype("float32")
        add_te[f"FREQ_{c}"] = te_vals.map(freq).fillna(0).astype("float32")
        add_or[f"FREQ_{c}"] = or_vals.map(freq).fillna(0).astype("float32")
        sorted_vals = np.sort(vals.to_numpy())
        add_tr[f"RANK_{c}"] = (np.searchsorted(sorted_vals, tr_vals.to_numpy()) / len(sorted_vals)).astype("float32")
        add_te[f"RANK_{c}"] = (np.searchsorted(sorted_vals, te_vals.to_numpy()) / len(sorted_vals)).astype("float32")
        add_or[f"RANK_{c}"] = (np.searchsorted(sorted_vals, or_vals.to_numpy()) / len(sorted_vals)).astype("float32")
    return pd.DataFrame(add_tr), pd.DataFrame(add_te), pd.DataFrame(add_or)


def add_orig_priors(train_df, test_df, orig_df):
    add_tr, add_te, add_or = {}, {}, {}
    for col in BASE_CATS:
        for cls in range(3):
            gp = orig_df.groupby(col, observed=False)[TARGET].apply(lambda s, cls=cls: (s == cls).mean())
            add_tr[f"ORIGP_{col}_c{cls}"] = train_df[col].map(gp).fillna(1 / 3).astype("float32")
            add_te[f"ORIGP_{col}_c{cls}"] = test_df[col].map(gp).fillna(1 / 3).astype("float32")
            add_or[f"ORIGP_{col}_c{cls}"] = orig_df[col].map(gp).fillna(1 / 3).astype("float32")
    return pd.DataFrame(add_tr), pd.DataFrame(add_te), pd.DataFrame(add_or)


def build_features(train_df, test_df, orig_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    orig_df = orig_df.copy()

    for df in [train_df, test_df, orig_df]:
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            df[c] = df[c].fillna(df[c].median())
        safe_str_inplace(df, BASE_CATS)

    def add_row_features(df):
        eps = 1e-6
        out = df.copy()
        out["water_total"] = (out["Rainfall_mm"] + out["Previous_Irrigation_mm"]).astype("float32")
        out["water_gap"] = (out["Rainfall_mm"] - out["Previous_Irrigation_mm"]).astype("float32")
        out["water_per_area"] = ((out["Rainfall_mm"] + out["Previous_Irrigation_mm"]) / (out["Field_Area_hectare"] + eps)).astype("float32")
        out["moisture_humidity_gap"] = (out["Soil_Moisture"] - out["Humidity"]).astype("float32")
        out["sun_temp_x"] = (out["Sunlight_Hours"] * out["Temperature_C"]).astype("float32")
        out["conductivity_ph_ratio"] = (out["Electrical_Conductivity"] / (out["Soil_pH"] + eps)).astype("float32")
        out["organic_moisture_x"] = (out["Organic_Carbon"] * out["Soil_Moisture"]).astype("float32")
        out["wind_sun_ratio"] = (out["Wind_Speed_kmh"] / (out["Sunlight_Hours"] + eps)).astype("float32")
        out["field_sq"] = (out["Field_Area_hectare"] ** 2).astype("float32")
        return out

    train_df = add_row_features(train_df)
    test_df = add_row_features(test_df)
    orig_df = add_row_features(orig_df)

    for df in [train_df, test_df, orig_df]:
        for c in NUM_DIGIT_COLS:
            vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
            int_part = np.floor(np.abs(vals)).astype(np.int64)
            frac = np.abs(vals) - np.floor(np.abs(vals))
            df[f"{c}_last_digit_cat"] = pd.Categorical((int_part % 10).astype(str))
            df[f"{c}_mod100_cat"] = pd.Categorical((int_part % 100).astype(str))
            df[f"{c}_frac2_cat"] = pd.Categorical((frac * 100).round().astype(int).astype(str))
            df[f"{c}_round1_cat"] = pd.Categorical(vals.round(1).astype(str))

    for c, step in ROUND_MAP.items():
        train_df[f"ROUND_{c}"] = pd.Categorical(round_to_step(train_df[c], step).astype(str))
        test_df[f"ROUND_{c}"] = pd.Categorical(round_to_step(test_df[c], step).astype(str))
        orig_df[f"ROUND_{c}"] = pd.Categorical(round_to_step(orig_df[c], step).astype(str))

    for col, bins_list in BIN_CONFIG.items():
        all_vals = pd.concat([train_df[col], test_df[col], orig_df[col]], axis=0).reset_index(drop=True)
        for n_bins in bins_list:
            try:
                edges = pd.qcut(all_vals, q=n_bins, duplicates="drop", retbins=True)[1]
            except Exception:
                edges = np.unique(np.linspace(all_vals.min(), all_vals.max(), n_bins + 1))
            train_df[f"{col}_{n_bins}_bin_cat"] = pd.Categorical(pd.cut(train_df[col], bins=edges, include_lowest=True, duplicates="drop").astype(str))
            test_df[f"{col}_{n_bins}_bin_cat"] = pd.Categorical(pd.cut(test_df[col], bins=edges, include_lowest=True, duplicates="drop").astype(str))
            orig_df[f"{col}_{n_bins}_bin_cat"] = pd.Categorical(pd.cut(orig_df[col], bins=edges, include_lowest=True, duplicates="drop").astype(str))

    for c1, c2 in CROSS_PAIRS:
        name = f"{c1}_{c2}_pair"
        train_df[name] = pd.Categorical(train_df[c1].astype(str) + "_" + train_df[c2].astype(str))
        test_df[name] = pd.Categorical(test_df[c1].astype(str) + "_" + test_df[c2].astype(str))
        orig_df[name] = pd.Categorical(orig_df[c1].astype(str) + "_" + orig_df[c2].astype(str))

    for cols in TRIPLE:
        name = "_".join(cols) + "_triple"
        train_df[name] = pd.Categorical(train_df[list(cols)].astype(str).agg("_".join, axis=1))
        test_df[name] = pd.Categorical(test_df[list(cols)].astype(str).agg("_".join, axis=1))
        orig_df[name] = pd.Categorical(orig_df[list(cols)].astype(str).agg("_".join, axis=1))

    tr_rank, te_rank, or_rank = add_global_rank_and_freq(train_df, test_df, orig_df, BASE_NUMS)
    tr_prior, te_prior, or_prior = add_orig_priors(train_df, test_df, orig_df)

    train_df = pd.concat([train_df, tr_rank, tr_prior], axis=1)
    test_df = pd.concat([test_df, te_rank, te_prior], axis=1)
    orig_df = pd.concat([orig_df, or_rank, or_prior], axis=1)

    use_cols = [c for c in train_df.columns if c not in [TARGET, ID]]
    return train_df[use_cols], test_df[use_cols], orig_df[use_cols]


def main():
    seed_everything(SEED)

    train = read_train()
    test = read_test()
    orig = read_original()

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)

    y = train[TARGET].copy()
    y_orig = orig[TARGET].copy()
    X, X_tst, X_orig = build_features(train, test, orig)

    print(f"Train features: {X.shape}")
    print(f"Test features : {X_tst.shape}")
    print(f"Orig features : {X_orig.shape}")

    fold_id = folds(y.values)
    oof_preds = np.zeros((len(X), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_tst), 3), dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        tr_idx, val_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print(f"\n### Fold {fold}/{N_FOLDS} ...")

        X_tr = pd.concat([X.iloc[tr_idx].copy(), X_orig.copy()], axis=0, ignore_index=True)
        y_tr = pd.concat([y.iloc[tr_idx].copy(), y_orig.copy()], axis=0, ignore_index=True)
        X_val = X.iloc[val_idx].copy()

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y.iloc[val_idx])

        val_preds = model.predict_proba(X_val).astype("float32")
        fold_test_preds = model.predict_proba(X_tst).astype("float32")

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / N_FOLDS

        fold_score = balanced_accuracy_score(y.iloc[val_idx], val_preds.argmax(axis=1))
        print(f"Fold {fold} | Balanced Accuracy: {fold_score:.5f}")

        del model, X_tr, y_tr, X_val, val_preds, fold_test_preds
        gc.collect()

    oof_score = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
    print(f"\nOverall OOF Balanced Accuracy: {oof_score:.5f}")
    print(f"Class order: {list(CLASSES)}")
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
