#!/usr/bin/env python
"""RealMLP on digit / rounding features, categorical pairs, formula score, per-class original priors, frequencies,
counts and drift (pair realmlp_digits_pairs_priors_drift_b).

Features (competition rows only in training; the original rows feed the priors and the drift):
- the 19 raw columns;
- for each of the 11 numerics: the 3 lowest integer digits and the first decimal of |x|;
- Soil_Moisture, Rainfall_mm, Temperature_C, Wind_Speed_kmh rounded to 1 and to 5;
- 6 categorical pairs (Crop_Growth_Stage x Mulching_Used / Season, Crop_Type x Soil_Type / Season / Crop_Growth_Stage,
  Region x Season), as categoricals;
- formula score (Soil_Moisture < 25, Rainfall_mm < 300, Temperature_C > 30, Wind_Speed_kmh > 10, growth stage
  Harvest / Sowing, mulching) and the flags score <= 0 and score > 3;
- per-class original target rate of the 8 categoricals and the 6 pairs (42 columns);
- value frequency of the 11 numerics over train + test; count and rare flag (< 50) of the 8 categoricals;
- drift log1p(train frequency / original frequency) of the 8 categoricals.
Model: pytabkit RealMLP_TD_Classifier, 2 x 512 mish, PBLD numeric embeddings (64 / 16, sigma 14.7), p_drop 0.42,
lr 0.068 with coslog4 schedule, n_ens 8, 30 epochs, no early stopping, random_state 42; the validation fold is passed
to fit as validation set. Test = mean of the 5 fold models.
Usage: python train_digits_pairs_priors_drift_b.py"""
import gc
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
from common.data import ID, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_digits_pairs_priors_drift_b"
SEED = 42
LABEL = "target"  # integer label column (0 / 1 / 2)

CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
NUMS = ["Soil_pH", "Organic_Carbon", "Electrical_Conductivity", "Sunlight_Hours",
        "Wind_Speed_kmh", "Field_Area_hectare", "Temperature_C", "Soil_Moisture",
        "Humidity", "Previous_Irrigation_mm", "Rainfall_mm"]
FORMULA_NUMS = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"]

REALMLP_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_epochs": 30,
    "random_state": SEED,
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


def build_features(train, test, orig, y_orig):
    """Adds every feature to train / test (and the digit / pair / formula ones to orig) in place."""
    print("Building features...")

    # 1. digit features
    for c in NUMS:
        for df in [train, test, orig]:
            v = df[c].fillna(0).astype(np.float64)
            int_part = np.floor(np.abs(v.values)).astype(np.int64)
            for i in range(3):
                df[f"{c}_digit{i}"] = ((int_part // (10 ** i)) % 10).astype(np.int8)
            frac = np.abs(v.values) - np.floor(np.abs(v.values))
            df[f"{c}_dec0"] = (np.floor(frac * 10) % 10).astype(np.int8)

    # 2. round features
    for c in FORMULA_NUMS:
        for df in [train, test, orig]:
            df[f"{c}_round1"] = np.round(df[c], 0).astype(np.float32)
            df[f"{c}_round5"] = (np.round(df[c].values / 5) * 5).astype(np.float32)

    # 3. categorical pairs
    cat_pairs = [
        ("Crop_Growth_Stage", "Mulching_Used"), ("Crop_Growth_Stage", "Season"),
        ("Crop_Type", "Soil_Type"), ("Crop_Type", "Season"),
        ("Crop_Type", "Crop_Growth_Stage"), ("Region", "Season"),
    ]
    for a, b in cat_pairs:
        for df in [train, test, orig]:
            df[f"{a}__{b}"] = df[a].astype(str) + "_" + df[b].astype(str)

    # 4. formula features
    for df in [train, test, orig]:
        SM = df["Soil_Moisture"]; RM = df["Rainfall_mm"]; TC = df["Temperature_C"]
        WS = df["Wind_Speed_kmh"]; CGS = df["Crop_Growth_Stage"]; MU = df["Mulching_Used"]
        score = ((SM < 25) * 2 + (RM < 300) * 2 + (TC > 30) + (WS > 10) - (CGS == "Harvest") * 2 - (CGS == "Sowing") * 2 - (MU == "Yes"))
        df["formula_score"] = score.astype(np.float32)
        df["is_low"] = (score <= 0).astype(np.float32)
        df["is_high"] = (score > 3).astype(np.float32)

    # 5. per-class original priors
    all_cols_for_prior = CATS + [f"{a}__{b}" for a, b in cat_pairs]
    for c in all_cols_for_prior:
        for k in range(3):
            m = orig.groupby(c).apply(lambda g, kk=k: (g[LABEL] == kk).mean())
            gm = float((y_orig == k).mean())
            for df in [train, test]:
                df[f"ORIG_k{k}_{c}"] = df[c].map(m).fillna(gm).astype(np.float32)

    # 6. value frequencies
    for c in NUMS:
        freq = pd.concat([train[c], test[c]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f"FREQ_{c}"] = df[c].map(freq).fillna(0).astype(np.float32)

    # 7. category counts / rare flags
    for c in CATS:
        vc = pd.concat([train[c], test[c]]).value_counts()
        for df in [train, test]:
            df[f"CNT_{c}"] = df[c].map(vc).fillna(0).astype(np.float32)
            df[f"RARE_{c}"] = (df[f"CNT_{c}"] < 50).astype(np.float32)

    # 8. drift ratio
    for c in CATS:
        p_tr = train[c].value_counts(normalize=True)
        p_or = orig[c].value_counts(normalize=True)
        for df in [train, test]:
            df[f"DRIFT_{c}"] = np.log1p(df[c].map(p_tr).fillna(0) / (df[c].map(p_or).fillna(1e-6) + 1e-6)).astype(np.float32)


def main():
    train = read_train()
    test = read_test()
    orig = read_original()

    train[LABEL] = train[TARGET].map(TARGET_MAP).astype(np.int32)
    orig[LABEL] = orig[TARGET].map(TARGET_MAP).astype(np.int32)
    y = train[LABEL].values
    y_orig = orig[LABEL].values

    for c in [ID, TARGET]:
        for df in [train, test]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
    print(f"Train: {train.shape}, Test: {test.shape}")

    build_features(train, test, orig, y_orig)

    features = [c for c in train.columns if c not in [LABEL, TARGET]]
    cat_cols = [c for c in features if train[c].dtype.name in ["object", "string", "category"]]
    print(f"Features: {len(features)}, Cat cols: {len(cat_cols)}")

    fold_id = folds(y)
    oof_preds = np.zeros((len(train), 3), dtype=np.float32)
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    fold_bacs = []

    test_fe = test[features].copy()

    for fold in range(1, N_FOLDS + 1):
        tr_idx, va_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        t0 = time.time()

        X_tr = train.iloc[tr_idx][features].copy()
        X_va = train.iloc[va_idx][features].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(
            X_tr, y_tr,
            X_val=X_va, y_val=y_va,
            cat_col_names=cat_cols,
        )

        va_probs = model.predict_proba(X_va).astype(np.float32)
        oof_preds[va_idx] = va_probs

        bac = balanced_accuracy_score(y_va, np.argmax(va_probs, axis=1))
        fold_bacs.append(bac)
        print(f"  Fold {fold} BAC={bac:.5f} ({time.time() - t0:.0f}s)")

        te_probs = model.predict_proba(test_fe).astype(np.float32)
        test_preds += te_probs / N_FOLDS

        del model
        torch.cuda.empty_cache()
        gc.collect()

    oof_bac = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
    print(f'\nFold BACs: {[f"{b:.5f}" for b in fold_bacs]}')
    print(f"OOF BAC: {oof_bac:.5f}")
    save_pair(model_dir(FOLDER), PAIR, oof_preds, test_preds)


if __name__ == "__main__":
    main()
