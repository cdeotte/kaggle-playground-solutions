#!/usr/bin/env python
"""TabM-mini-normal on frequency / ratio / original-prior / distribution / digit features plus fold target encodings of
numeric-as-category values, categoricals and n-grams (pair tabm_original_priors_te_ngram_a).

Features (competition rows only; numeric NaN -> train median):
- the 11 numerics; their value frequency over train + original + test;
- water balance, rain / sunlight, previous irrigation / field area, moisture / humidity, temperature x humidity,
  conductivity x pH;
- per-class original target rate of the 8 categoricals and of Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture,
  Humidity rounded to 2 decimals (36 columns);
- Previous_Irrigation_mm and Rainfall_mm: percentile rank and z-score within the original Low / Medium / High rows,
  percentile rank within all original rows;
- Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Humidity: integer part mod 10 / mod 100, the two decimals,
  is-integer flag;
- per fold: scikit-learn TargetEncoder (multiclass, smooth auto, 5-fold cross fitting, shuffle, seed 42) of the 11
  numerics as category strings (rounded to 2 decimals for Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture,
  Humidity, Temperature_C, to 1 for the others) and the 8 categoricals (57 columns), and of 15 bigrams of Crop_Type,
  Season, Irrigation_Type, Water_Source, Region, Soil_Type and 4 trigrams of the first 4 (57 columns).
All 208 numeric columns are standardized on the fold's training rows; the 8 categoricals are ordinal-encoded
(unseen -> -1).
Model: pytabkit TabM_D_Classifier, tabm-mini-normal, k 32, 3 blocks x 256, piecewise-linear numeric embeddings
(d 24), dropout 0.2, lr 1e-3, weight decay 1e-3, batch 512, up to 50 epochs, patience 10 on the validation fold,
random_state 42. Test = mean of the 5 fold models.
Usage: python train_original_priors_te_ngram_a.py"""
import gc
import random
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import TabM_D_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "tabm"
PAIR = "tabm_original_priors_te_ngram_a"
SEED = 42
INNER_FOLDS = 5
TABM_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbosity": 0,
    "arch_type": "tabm-mini-normal",
    "tabm_k": 32,
    "num_emb_type": "pwl",
    "d_embedding": 24,
    "batch_size": 512,
    "lr": 1e-3,
    "n_epochs": 50,
    "dropout": 0.2,
    "d_block": 256,
    "n_blocks": 3,
    "patience": 10,
    "weight_decay": 1e-3,
    "random_state": 42,
}

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

TOP_CATS_FOR_NGRAM = [
    "Crop_Type",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Region",
    "Soil_Type",
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


def load_data():
    print("Loading datasets...")
    train = read_train()
    test = read_test()
    orig = read_original()

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)

    for df in [train, test, orig]:
        for col in NUMS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    train_num_medians = train[NUMS].median()
    for df in [train, test, orig]:
        for col in NUMS:
            df[col] = df[col].fillna(train_num_medians[col])

    print(f"Train: {train.shape}")
    print(f"Test : {test.shape}")
    print(f"Orig : {orig.shape}")
    return train, test, orig


def add_base_features(train, test, orig):
    new_nums = []

    print("[1/6] Frequency Encoding...")
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    print("[2/6] Arithmetic Features...")
    for df in [train, test, orig]:
        df["water_balance"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"]).astype("float32")
        df["rain_to_sun_ratio"] = (df["Rainfall_mm"] / (df["Sunlight_Hours"] + 1)).astype("float32")
        df["prev_to_area_ratio"] = (df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 1)).astype("float32")
        df["moisture_to_humidity"] = (df["Soil_Moisture"] / (df["Humidity"] + 1e-3)).astype("float32")
        df["temp_humidity_interaction"] = (df["Temperature_C"] * df["Humidity"]).astype("float32")
        df["ec_ph_interaction"] = (df["Electrical_Conductivity"] * df["Soil_pH"]).astype("float32")
    new_nums += [
        "water_balance",
        "rain_to_sun_ratio",
        "prev_to_area_ratio",
        "moisture_to_humidity",
        "temp_humidity_interaction",
        "ec_ph_interaction",
    ]

    print("[3/6] Original Probability Mapping...")
    key_num_cols = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity"]
    for df in [train, test, orig]:
        for col in key_num_cols:
            df[f"MAP_{col}"] = df[col].round(2).astype(str)

    map_cols = CATS + [f"MAP_{col}" for col in key_num_cols]
    orig_onehot = pd.get_dummies(orig[TARGET], prefix="cls")
    orig_tmp = pd.concat([orig[map_cols], orig_onehot], axis=1)
    class_names = list(orig_onehot.columns)

    for col in map_cols:
        gp = orig_tmp.groupby(col)[class_names].mean().reset_index()
        rename_map = {cls: f"ORIG_proba_{col}_{cls}" for cls in class_names}
        gp = gp.rename(columns=rename_map)
        train = train.merge(gp, on=col, how="left")
        test = test.merge(gp, on=col, how="left")
        for feat in rename_map.values():
            train[feat] = train[feat].fillna(1.0 / len(class_names)).astype("float32")
            test[feat] = test[feat].fillna(1.0 / len(class_names)).astype("float32")
            new_nums.append(feat)

    print("[4/6] Distribution Features...")
    key_dist_cols = ["Previous_Irrigation_mm", "Rainfall_mm"]
    for col in key_dist_cols:
        orig_all = orig[col].values
        for cls_id, cls_name in enumerate(CLASSES):
            ref = orig.loc[orig[TARGET] == cls_id, col].values
            for df in [train, test]:
                vals = df[col].values
                df[f"pctrank_{cls_name}_{col}"] = pctrank_against(vals, ref)
                df[f"zscore_{cls_name}_{col}"] = zscore_against(vals, ref)
            new_nums += [f"pctrank_{cls_name}_{col}", f"zscore_{cls_name}_{col}"]
        for df in [train, test]:
            vals = df[col].values
            df[f"pctrank_orig_{col}"] = pctrank_against(vals, orig_all)
        new_nums.append(f"pctrank_orig_{col}")

    print("[5/6] Digit Features...")
    digit_cols = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity"]
    for df in [train, test]:
        for col in digit_cols:
            prefix = col.replace("_", "")
            intval = np.floor(df[col]).astype(int)
            frac = (df[col] - np.floor(df[col])).astype("float32")
            df[f"{prefix}_mod10"] = (intval % 10).astype("float32")
            df[f"{prefix}_mod100"] = (intval % 100).astype("float32")
            df[f"{prefix}_frac100"] = np.round(frac * 100).astype("float32")
            df[f"{prefix}_is_round"] = (frac < 0.005).astype("float32")
            new_nums += [
                f"{prefix}_mod10",
                f"{prefix}_mod100",
                f"{prefix}_frac100",
                f"{prefix}_is_round",
            ]

    print("[6/6] N-gram Features...")
    bigram_cols, trigram_cols = [], []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        bigram_cols.append(col_name)

    top4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        trigram_cols.append(col_name)
    ngram_cols = bigram_cols + trigram_cols

    num_as_cat = []
    for col in NUMS:
        new_col = f"CAT_{col}"
        num_as_cat.append(new_col)
        round_digits = 2 if col in {"Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"} else 1
        for df in [train, test]:
            df[new_col] = df[col].round(round_digits).astype(str)

    te_columns = num_as_cat + CATS

    new_nums = list(dict.fromkeys(new_nums))
    print(f"Numerical features total: {len(NUMS + new_nums)}")
    print(f"Categorical TE columns   : {len(te_columns)}")
    print(f"N-gram columns           : {len(ngram_cols)}")
    return train, test, new_nums, te_columns, ngram_cols


def run_training(train, test, new_nums, te_columns, ngram_cols):
    print("=" * 70)
    print("TRAINING TABM WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    y_all = train[TARGET].values
    fold_id = folds(y_all)
    oof = np.zeros((len(train), 3), dtype="float32")
    pred = np.zeros((len(test), 3), dtype="float32")
    fold_scores = []

    for fold_i in range(1, N_FOLDS + 1):
        train_idx, val_idx = np.where(fold_id != fold_i - 1)[0], np.where(fold_id == fold_i - 1)[0]
        t0 = time.time()
        print(f"\nFold {fold_i}/{N_FOLDS}")

        X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr = y_all[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te = test.copy()

        te = TargetEncoder(cv=INNER_FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_te = te.fit_transform(X_tr[te_columns], y_tr).astype("float32")
        val_te = te.transform(X_val[te_columns]).astype("float32")
        te_te = te.transform(X_te[te_columns]).astype("float32")
        te_feat_names = [f"TE_{i}" for i in range(tr_te.shape[1])]
        X_tr[te_feat_names] = tr_te
        X_val[te_feat_names] = val_te
        X_te[te_feat_names] = te_te

        ng_te = TargetEncoder(cv=INNER_FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_ng = ng_te.fit_transform(X_tr[ngram_cols], y_tr).astype("float32")
        val_ng = ng_te.transform(X_val[ngram_cols]).astype("float32")
        te_ng = ng_te.transform(X_te[ngram_cols]).astype("float32")
        ng_feat_names = [f"NGTE_{i}" for i in range(tr_ng.shape[1])]
        X_tr[ng_feat_names] = tr_ng
        X_val[ng_feat_names] = val_ng
        X_te[ng_feat_names] = te_ng

        all_nums_final = NUMS + new_nums + te_feat_names + ng_feat_names
        all_cats_final = CATS

        if fold_i == 1:
            print(f"Numeric features    : {len(all_nums_final)}")
            print(f"Categorical features: {len(all_cats_final)}")

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(X_tr[all_cats_final].astype(str))
        X_tr_cat = encoder.transform(X_tr[all_cats_final].astype(str))
        X_val_cat = encoder.transform(X_val[all_cats_final].astype(str))
        X_te_cat = encoder.transform(X_te[all_cats_final].astype(str))

        for df in [X_tr, X_val, X_te]:
            df[all_nums_final] = df[all_nums_final].fillna(0).astype("float32")

        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[all_nums_final])
        X_val_num = scaler.transform(X_val[all_nums_final])
        X_te_num = scaler.transform(X_te[all_nums_final])

        all_cols = all_nums_final + all_cats_final
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=all_cols)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=all_cols)
        X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=all_cols)

        for c in all_cats_final:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c] = X_te_final[c].astype(int)

        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(
            X_tr_final,
            y_tr,
            X_val=X_val_final,
            y_val=y_val,
            cat_col_names=all_cats_final,
        )

        val_probs = model.predict_proba(X_val_final).astype("float32")
        test_probs = model.predict_proba(X_te_final).astype("float32")

        oof[val_idx] = val_probs
        pred += test_probs / N_FOLDS

        fold_ba = balanced_accuracy_score(y_val, val_probs.argmax(axis=1))
        fold_scores.append(fold_ba)
        print(f"Fold {fold_i} Balanced Accuracy: {fold_ba:.6f} | Time: {(time.time() - t0) / 60:.1f} min")

        del model, X_tr, X_val, X_te, X_tr_final, X_val_final, X_te_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_ba = balanced_accuracy_score(y_all, oof.argmax(axis=1))
    print("\n" + "=" * 70)
    print(f"OOF Balanced Accuracy:  {overall_ba:.6f}")
    print(f"Mean BA: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
    print("=" * 70)
    return oof, pred


def main():
    seed_everything(SEED)
    train, test, orig = load_data()
    train, test, new_nums, te_columns, ngram_cols = add_base_features(train, test, orig)
    oof, pred = run_training(train, test, new_nums, te_columns, ngram_cols)
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
