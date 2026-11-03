#!/usr/bin/env python
"""CatBoost on numeric transforms and native value categories, original rows appended
(pair catboost_numcat_crosses_original_rows_a).

Label-free features (frequencies over the train + test + original rows):
  - each of the 11 numerics: value, log1p(|x|), frequency, integer part mod 100 and mod 10, fractional part
  - rainfall + previous irrigation, rainfall - previous irrigation, their sum / (area + 1), soil moisture - humidity,
    sunlight x temperature, conductivity / (pH + 1), and a 3-level flag of |rainfall - previous irrigation|
    (< 10, < 60, >= 60)
  - frequency of each of the 8 categoricals
  - 53 native CatBoost categories: the flag, the 8 categoricals, every numeric rounded to 2 decimals / floored /
    floored to an even integer, rainfall | previous irrigation | soil moisture (2 decimals), and the 10 pairs of
    Crop_Type, Season, Irrigation_Type, Water_Source, Region
Per fold: all 10,000 original rows are appended to the training rows; the numeric columns are ranked by their variance
on these rows and the top 120 are kept (all 81 pass), followed by the categories.
Model: CatBoost GPU, MultiClass, depth 6, lr 0.03, l2 3, random_strength 0.5, Bernoulli subsample 0.8, 254 borders,
one_hot_max_size 4, max_ctr_complexity 1, up to 5000 iterations with early stopping (300) on the validation fold,
seed 42. Test = mean of the 5 fold models.
Usage: python train_numcat_crosses.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "catboost_numcat_crosses_original_rows_a"
SEED = 42
N_FOLDS = 5
N_KEEP_NUMERIC = 120
CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    border_count=254,
    grow_policy="SymmetricTree",
    eval_metric="MultiClass",
    loss_function="MultiClass",
    task_type="GPU",
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=300,
    use_best_model=True,
    one_hot_max_size=4,
    max_ctr_complexity=1,
)
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)
    for df in [train, test, orig]:
        for c in NUM_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    log(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")
    return train, test, orig


def get_column_types(train):
    all_cols = [c for c in train.columns if c != TARGET]
    num_cols = [c for c in NUM_COLS if c in all_cols]
    cat_cols = [c for c in all_cols if c not in num_cols + [ID]]
    return num_cols, cat_cols


def build_features(train, test, orig):
    log("Building features...")
    num_cols, cat_cols = get_column_types(train)

    combined = pd.concat([
        train.drop(columns=[TARGET], errors="ignore"),
        test.drop(columns=[TARGET], errors="ignore"),
        orig.drop(columns=[TARGET], errors="ignore"),
    ], axis=0, ignore_index=True)

    n_train = len(train)
    n_test = len(test)
    y_train = train[TARGET].values
    y_orig = orig[TARGET].values

    feat = pd.DataFrame(index=combined.index)
    for c in num_cols:
        vals = combined[c].values.astype(np.float64)
        feat[c] = vals
        feat[f"{c}_log1p"] = np.log1p(np.abs(vals))
        feat[f"{c}_freq"] = combined[c].map(combined[c].value_counts()).astype(np.float32).values
        int_part = np.floor(np.abs(vals)).astype(np.int64)
        feat[f"{c}_mod100"] = (int_part % 100).astype(np.int16)
        feat[f"{c}_mod10"] = (int_part % 10).astype(np.int8)
        feat[f"{c}_frac"] = (vals - np.floor(vals)).astype(np.float32)

    rain = combined["Rainfall_mm"].values.astype(np.float64)
    prev = combined["Previous_Irrigation_mm"].values.astype(np.float64)
    moist = combined["Soil_Moisture"].values.astype(np.float64)
    hum = combined["Humidity"].values.astype(np.float64)
    temp = combined["Temperature_C"].values.astype(np.float64)
    sun = combined["Sunlight_Hours"].values.astype(np.float64)
    area = combined["Field_Area_hectare"].values.astype(np.float64)
    ec = combined["Electrical_Conductivity"].values.astype(np.float64)
    ph = combined["Soil_pH"].values.astype(np.float64)

    feat["water_total"] = rain + prev
    feat["water_gap"] = rain - prev
    feat["water_per_area"] = feat["water_total"] / (area + 1.0)
    feat["moisture_humidity_gap"] = moist - hum
    feat["sun_temp_x"] = sun * temp
    feat["conductivity_ph_ratio"] = ec / (ph + 1.0)
    feat["water_stress_flag"] = pd.cut(
        np.abs(feat["water_gap"]),
        bins=[-np.inf, 10.0, 60.0, np.inf],
        labels=[0, 1, 2],
        right=False,
    ).astype(np.int8)

    for c in cat_cols:
        feat[f"{c}_freq"] = combined[c].astype(str).map(combined[c].astype(str).value_counts()).astype(np.float32).values

    cat_native = pd.DataFrame(index=combined.index)
    cat_native["water_stress_flag_str"] = feat["water_stress_flag"].astype(str)
    for c in cat_cols:
        cat_native[f"RAW_{c}"] = combined[c].astype(str).fillna("__NA__")
    for c in num_cols:
        cat_native[f"NUMCAT_{c}_exact"] = np.round(combined[c], 2).astype(str)
        cat_native[f"NUMCAT_{c}_int"] = np.floor(combined[c].values).astype(int).astype(str)
        cat_native[f"NUMCAT_{c}_div2"] = (np.floor(combined[c].values / 2) * 2).astype(int).astype(str)
    cat_native["CROSS_exact_all"] = (
        np.round(combined["Rainfall_mm"], 2).astype(str) + "_"
        + np.round(combined["Previous_Irrigation_mm"], 2).astype(str) + "_"
        + np.round(combined["Soil_Moisture"], 2).astype(str)
    )
    top_cats = [c for c in ["Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region"] if c in cat_cols]
    for i, c1 in enumerate(top_cats):
        for c2 in top_cats[i + 1:]:
            cat_native[f"BI_{c1}_{c2}"] = combined[c1].astype(str) + "_" + combined[c2].astype(str)

    train_feat = feat.iloc[:n_train].reset_index(drop=True)
    test_feat = feat.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_feat = feat.iloc[n_train + n_test:].reset_index(drop=True)
    train_cat_native = cat_native.iloc[:n_train].reset_index(drop=True)
    test_cat_native = cat_native.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_cat_native = cat_native.iloc[n_train + n_test:].reset_index(drop=True)

    train_all = pd.concat([train_feat, train_cat_native], axis=1)
    test_all = pd.concat([test_feat, test_cat_native], axis=1)
    orig_all = pd.concat([orig_feat, orig_cat_native], axis=1)
    all_features = train_all.columns.tolist()
    cat_features = train_cat_native.columns.tolist()
    log(f"Features: {len(all_features)} ({len(cat_features)} categorical)")
    return train_all, test_all, orig_all, all_features, cat_features, y_train, y_orig


def main():
    train, test, orig = load_data()
    train_all, test_all, orig_all, all_features, cat_features, y_train, y_orig = build_features(train, test, orig)

    fold_ids = folds(y_train)
    oof = np.zeros((len(train_all), 3), dtype=np.float32)
    test_pred = np.zeros((len(test_all), 3), dtype=np.float32)
    fold_scores = []
    for k in range(N_FOLDS):
        train_idx = np.where(fold_ids != k)[0]
        val_idx = np.where(fold_ids == k)[0]
        log(f"Fold {k}")
        X_tr_base = train_all.iloc[train_idx].reset_index(drop=True)
        y_tr_base = y_train[train_idx]
        X_val = train_all.iloc[val_idx].reset_index(drop=True)
        y_val = y_train[val_idx]

        X_train = pd.concat([X_tr_base, orig_all.reset_index(drop=True)], axis=0, ignore_index=True)
        y_train_fold = np.concatenate([y_tr_base, y_orig])

        # numeric columns ranked by variance on the fold's training rows (+ original rows), top 120 kept
        numeric_candidates = [c for c in all_features if c not in cat_features]
        var_scores = X_train[numeric_candidates].var().sort_values(ascending=False)
        keep_num = var_scores.head(min(N_KEEP_NUMERIC, len(var_scores))).index.tolist()
        keep_features = keep_num + cat_features

        X_train = X_train[keep_features]
        X_val = X_val[keep_features]
        X_test_fold = test_all[keep_features]
        cat_features_fold = [c for c in cat_features if c in keep_features]

        train_pool = Pool(X_train, label=y_train_fold, cat_features=cat_features_fold)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_features_fold)
        test_pool = Pool(X_test_fold, cat_features=cat_features_fold)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=500)

        val_preds = model.predict_proba(val_pool).astype(np.float32)
        test_pred += model.predict_proba(test_pool).astype(np.float32) / N_FOLDS
        oof[val_idx] = val_preds
        fold_ba = balanced_accuracy_score(y_val, val_preds.argmax(axis=1))
        fold_scores.append(fold_ba)
        log(f"Fold {k} balanced accuracy: {fold_ba:.6f}")

        del X_tr_base, y_tr_base, X_val, y_val, X_train, y_train_fold, X_test_fold
        del train_pool, val_pool, test_pool, model, val_preds
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y_train, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == "__main__":
    main()
