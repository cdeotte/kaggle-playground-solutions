#!/usr/bin/env python
"""RealMLP on ratios, digit flags, KBins bins, numeric value codes and a Crop_Type x Season x Irrigation_Type target
encoding (pair realmlp_kbins_value_codes_combo_te_a).

Features (competition rows only; bins and codes fitted on all train rows and applied to the test rows):
- the 19 raw columns; 7 ratios / products (moisture / humidity, rain / sunlight, previous irrigation / field area,
  temperature x humidity, conductivity x pH, rain + previous irrigation - moisture, field area squared);
- categories: large field (>= 8 ha), second decimal of Rainfall_mm and Previous_Irrigation_mm, previous irrigation is a
  multiple of 10; numerics: the two decimals of Soil_Moisture and Humidity, Rainfall_mm and Previous_Irrigation_mm
  integer part mod 100;
- KBinsDiscretizer quantile and k-means bins of Rainfall_mm (5000, 400) and Previous_Irrigation_mm (300);
- every numeric (rounded to 2 decimals for Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Humidity, to 0 for the
  others) factorized as a category (unseen test values -> -1); the Crop_Type x Season x Irrigation_Type code;
- per fold: scikit-learn TargetEncoder (multiclass, smooth auto, 5-fold cross fitting, seed 42) of that code (3 columns).
Model: pytabkit RealMLP_TD_Classifier, silu [512, 256, 128], numeric embeddings (plr_hidden 16 / 8, gelu), n_ens 8,
lr 0.075 with flat_anneal schedule, batch 256, 3 epochs, early stopping on the validation fold, random_state 42.
Test = mean of the 5 fold models.
Usage: python train_kbins_value_codes_combo_te_a.py"""
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
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_kbins_value_codes_combo_te_a"
SEED = 42

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

category_map = {}
important_combos = [
    ("Crop_Type", "Season", "Irrigation_Type"),
]


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def feature_engineering(df, num_cols, fit=False):
    """New categorical columns end with '_', new numeric columns start with '_'. fit=True fits the bins / codes."""
    df = df.copy()
    eps = 1e-6

    df["_SoilMoisture_/_Humidity"] = (df["Soil_Moisture"] / (df["Humidity"] + eps)).astype("float32")
    df["_Rainfall_/_Sunlight"] = (df["Rainfall_mm"] / (df["Sunlight_Hours"] + eps)).astype("float32")
    df["_PrevIrrigation_/_FieldArea"] = (df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + eps)).astype("float32")
    df["_Temperature_x_Humidity"] = (df["Temperature_C"] * df["Humidity"]).astype("float32")
    df["_EC_x_pH"] = (df["Electrical_Conductivity"] * df["Soil_pH"]).astype("float32")
    df["_WaterBalance"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"]).astype("float32")
    df["_FieldArea_sq"] = (df["Field_Area_hectare"] ** 2).astype("float32")
    df["is_large_field_"] = (df["Field_Area_hectare"] >= 8).astype("category")

    for col in ["Rainfall_mm", "Previous_Irrigation_mm"]:
        digit_name = f"{col}_d2_"
        df[digit_name] = (np.floor((df[col] % 1) * 100).astype("int16") % 10).astype("int8")
        df[digit_name] = df[digit_name].astype("category")

    for col in ["Soil_Moisture", "Humidity"]:
        decimal_name = f"_{col}_decimal"
        df[decimal_name] = (df[col] % 1).round(2).astype("float32")

    df["_Rainfall_mod100"] = (np.floor(df["Rainfall_mm"]) % 100).astype("float32")
    df["_PrevIrrigation_mod100"] = (np.floor(df["Previous_Irrigation_mm"]) % 100).astype("float32")
    df["Previous_Irrigation_is_multiple_10_"] = (np.floor(df["Previous_Irrigation_mm"]) % 10 == 0).astype("category")

    bin_config = {
        "Rainfall_mm": [5000, 400],
        "Previous_Irrigation_mm": [300],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ["quantile", "kmeans"]:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="ordinal",
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype("int32")
                df[bin_name] = pd.Categorical(binned)

    for col in num_cols:
        cat_name = f"{col}_cat_"
        round_level = 2 if col in {"Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity"} else 0
        if fit:
            series = df[col].round(round_level)
            codes, uniques = series.factorize()
            category_map[cat_name] = {"uniques": uniques, "round_level": round_level}
        else:
            meta = category_map[cat_name]
            uniques = meta["uniques"]
            series = df[col].round(meta["round_level"])
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = series.map(code_map).fillna(-1).astype("int32")
        df[cat_name] = pd.Categorical(codes)

    combo_names = []
    for col1, col2, col3 in important_combos:
        combo_name = f"{col1}_{col2}_{col3}_"
        combo_names.append(combo_name)
        combo_series = df[col1].astype(str) + "_" + df[col2].astype(str) + "_" + df[col3].astype(str)
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = pd.Categorical(codes)

    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


def main():
    seed_everything(SEED)

    print("Loading data...")
    train = read_train()
    test = read_test()
    train[TARGET] = train[TARGET].map(TARGET_MAP)

    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET].astype("int64")
    X_test = test.drop([ID], axis=1)

    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print("Applying feature engineering...")
    X, _, _, combo_names = feature_engineering(X, num_cols, fit=True)
    X_test, _, _, _ = feature_engineering(X_test, num_cols, fit=False)

    print("Starting K-Fold Training...")
    fold_id = folds(y.values)
    oof_preds = np.zeros((len(X), 3), dtype=np.float32)
    test_preds = np.zeros((len(X_test), 3), dtype=np.float32)

    for fold in range(1, N_FOLDS + 1):
        tr_idx, val_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print(f"### Fold {fold}/{N_FOLDS} ...")

        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_tst = X_test.copy()

        te_cols = combo_names
        te = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
        tr_enc = te.fit_transform(X_tr[te_cols], y.iloc[tr_idx])
        val_enc = te.transform(X_val[te_cols])
        tst_enc = te.transform(X_tst[te_cols])

        n_classes = tr_enc.shape[1] // len(te_cols)
        te_names = [f"_{col}TE_class{class_idx}" for col in te_cols for class_idx in range(n_classes)]
        X_tr[te_names] = tr_enc.astype("float32")
        X_val[te_names] = val_enc.astype("float32")
        X_tst[te_names] = tst_enc.astype("float32")

        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y.iloc[tr_idx], X_val, y.iloc[val_idx])

        val_preds = model.predict_proba(X_val).astype("float32")
        fold_test_preds = model.predict_proba(X_tst).astype("float32")

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / N_FOLDS

        fold_score = balanced_accuracy_score(y.iloc[val_idx], val_preds.argmax(axis=1))
        print(f"Fold {fold} | Balanced Accuracy: {fold_score:.5f}\n")
        torch.cuda.empty_cache()

    oof_score = balanced_accuracy_score(y, oof_preds.argmax(axis=1))
    print(f"Overall OOF Balanced Accuracy: {oof_score:.5f}")
    print("Class order:", list(CLASSES))
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
