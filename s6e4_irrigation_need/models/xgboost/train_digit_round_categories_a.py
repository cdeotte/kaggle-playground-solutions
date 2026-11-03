#!/usr/bin/env python
"""GPU XGBoost with native categoricals on digit positions, rounded numerics as categories and category triples
(pair xgb_digit_round_categories_a).

Features (built on train + test): the 19 raw columns (the 8 categoricals as native categoricals); 6 arithmetic features
(rain - previous irrigation, rain + previous irrigation, moisture - humidity, temperature x sunlight, water per field
area, moisture / rain); Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Humidity, Temperature_C and
Field_Area_hectare rounded to 1 and to 0 decimals as categories; 3 category triples (crop x season x irrigation type,
soil x crop x region, water source x irrigation type x region); the integer digit positions and the first 2 decimal
digits of the same 6 numerics as categories (27). 67 columns.
Model: XGBoost hist on GPU, depth 3, lr 0.03, subsample / colsample_bytree 0.8, max_bin 16,000, up to 20,000 rounds,
early stopping 200 (mlogloss) on the validation fold, balanced sample weights, random_state 42. Test = mean of the
5 fold models. Competition rows only.
Usage: python train_digit_round_categories_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, TARGET, TARGET_MAP, folds, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.simplefilter("ignore")

PAIR = "xgb_digit_round_categories_a"
SEED = 42
N_SPLITS = 5
LABELS = list(CLASSES)

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

DIGIT_BASE_COLS = [
    "Rainfall_mm",
    "Previous_Irrigation_mm",
    "Soil_Moisture",
    "Humidity",
    "Temperature_C",
    "Field_Area_hectare",
]

xgb_params = {
    "n_estimators": 20000,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 16000,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_jobs": -1,
    "random_state": SEED,
    "early_stopping_rounds": 200,
    "device": "cuda",
    "enable_categorical": True,
    "tree_method": "hist",
}


def add_digit_features(df, base_cols, max_dec_digits=2):
    digit_cols = []
    for col in base_cols:
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        fill_val = float(vals.median()) if vals.notna().any() else 0.0
        vals = vals.fillna(fill_val)

        max_val = float(vals.abs().max())
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1

        abs_int = vals.abs().astype(np.int64)
        for i in range(max_int_digits):
            new_col = f"{col}_digit_pos_{i+1}"
            df[new_col] = pd.Categorical(((abs_int // (10 ** i)) % 10).astype(str))
            digit_cols.append(new_col)

        frac = (vals.abs() - np.floor(vals.abs()))
        scaled = (frac * (10 ** max_dec_digits)).round().astype(np.int64)
        for i in range(max_dec_digits):
            new_col = f"{col}_decimal_digit_{i+1}"
            df[new_col] = pd.Categorical(((scaled // (10 ** (max_dec_digits - i - 1))) % 10).astype(str))
            digit_cols.append(new_col)

    return digit_cols


def add_artifact_features(df):
    df = df.copy()

    for col in NUMS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rain_minus_prev_irrig"] = df["Rainfall_mm"] - df["Previous_Irrigation_mm"]
    df["water_total"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["moisture_humidity_gap"] = df["Soil_Moisture"] - df["Humidity"]
    df["heat_x_sun"] = df["Temperature_C"] * df["Sunlight_Hours"]
    df["water_per_area"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"]) / (df["Field_Area_hectare"] + 1e-3)
    df["moisture_per_rain"] = df["Soil_Moisture"] / (df["Rainfall_mm"] + 1.0)

    for col in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C", "Field_Area_hectare"]:
        df[f"{col}_round1_cat"] = pd.Categorical(df[col].round(1).astype(str))
        df[f"{col}_round0_cat"] = pd.Categorical(df[col].round(0).astype(int).astype(str))

    combo_specs = [
        ("Crop_Type", "Season", "Irrigation_Type"),
        ("Soil_Type", "Crop_Type", "Region"),
        ("Water_Source", "Irrigation_Type", "Region"),
    ]
    for cols in combo_specs:
        name = "_".join(cols)
        df[name] = pd.Categorical(df[list(cols)].astype(str).agg("_".join, axis=1))

    digit_cols = add_digit_features(df, DIGIT_BASE_COLS)

    cat_cols = [c for c in df.columns if str(df[c].dtype) == "category" or df[c].dtype == "object"]
    feature_cols = [c for c in df.columns if c not in ["id", TARGET]]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df, feature_cols, digit_cols


def build_features():
    train = read_train()
    test = read_test()
    print(f"train shape = {train.shape}, test shape = {test.shape}")

    train[TARGET] = train[TARGET].map(TARGET_MAP)
    combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    combined_fe, feature_cols, digit_cols = add_artifact_features(combined)

    train_fe = combined_fe.iloc[: len(train)].copy()
    test_fe = combined_fe.iloc[len(train) :].copy().reset_index(drop=True)
    train_fe[TARGET] = train[TARGET].values
    y = train_fe[TARGET].values

    print(f"feature count = {len(feature_cols)}")
    print(f"digit feature count = {len(digit_cols)}")
    return train_fe, test_fe, y, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train_fe, test_fe, y, feature_cols = build_features()
        fold_id = folds(y)
        for k in todo:
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            print(f"\n### Fold {k}", flush=True)
            X_train = train_fe.iloc[tr_idx][feature_cols].copy()
            X_valid = train_fe.iloc[va_idx][feature_cols].copy()
            y_train = y[tr_idx]
            y_valid = y[va_idx]
            X_test = test_fe[feature_cols].copy()

            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

            model = XGBClassifier(**xgb_params)
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_valid, y_valid)],
                verbose=200,
            )

            p_va = model.predict_proba(X_valid)
            p_te = model.predict_proba(X_test)
            save_part(HERE, PAIR, k, p_va, p_te)
            fold_score = balanced_accuracy_score(y_valid, p_va.argmax(axis=1))
            print(f"{PAIR} fold {k} | Balanced Accuracy: {fold_score:.5f}", flush=True)

            del X_train, X_valid, X_test, model
            gc.collect()
    y = read_train()[TARGET].map(TARGET_MAP).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
