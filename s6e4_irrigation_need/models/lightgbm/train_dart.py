#!/usr/bin/env python
"""DART LightGBM on frequencies, original-class distributions and priors, digits and n-gram target encodings
(pair lgbm_dart_priors_ngrams_sklearn_te_a).

Numerics are filled with the train medians. Label-free features (124 numeric columns):
  - the 11 numerics
  - frequency over train + test + original of every categorical and of every numeric rounded to 2 decimals
  - rainfall + previous irrigation - soil moisture, rainfall / sunlight, previous irrigation / area,
    soil moisture / humidity, temperature x humidity, conductivity x pH
  - rainfall, previous irrigation, soil moisture, humidity against the original rows: percentile rank and z-score
    within each class, percentile rank among all original rows
  - integer part mod 10 and mod 100, the 2 first decimals as an integer, and a whole-number flag of the same 4 numerics
  - per-class original frequency of the 8 categoricals and of the 4 numerics rounded to 2 decimals (missing: 1/3)
  - the 8 categoricals factorized over train + test + original
Target-encoded keys (36): the 8 categoricals, 7 numerics rounded to 2 decimals (rainfall, previous irrigation, soil
moisture, humidity, temperature, sunlight, area), <= 500 quantile bins of rainfall and previous irrigation (edges from
train + test), the 15 pairs of Crop_Type, Season, Irrigation_Type, Water_Source, Region, Soil_Type and the 4 triples
of the first 4. Per fold: scikit-learn TargetEncoder (multiclass, cv 5, shuffle, seed 42, smooth 'auto') fitted on
the training rows (out-of-fold for them), 108 columns.
Model: LightGBM DART multiclass, 3000 rounds (the early-stopping callback is inactive with DART, all rounds are used),
lr 0.03, 127 leaves, min_data_in_leaf 50, feature_fraction 0.7, bagging 0.8 every round, lambda_l1 1, lambda_l2 3,
max_bin 255, seed 42, 32 threads. Test = mean of the 5 fold models.
Usage: python train_dart.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

PAIR = "lgbm_dart_priors_ngrams_sklearn_te_a"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
TARGET_NAMES = ["Low", "Medium", "High"]

RAW_CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
RAW_NUMS = [
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
NUMERIC_ROUND_MAP = {
    "Rainfall_mm": 2,
    "Previous_Irrigation_mm": 2,
    "Soil_Moisture": 2,
    "Humidity": 2,
    "Temperature_C": 2,
    "Sunlight_Hours": 2,
    "Field_Area_hectare": 2,
    "Soil_pH": 2,
}
KEY_NUMS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity"]

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "dart",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 3.0,
    "max_bin": 255,
    "seed": SEED,
    "verbosity": -1,
    "n_jobs": 32,
}


def log(msg):
    print(msg, flush=True)


def pctrank_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


def sanitize_str(df, cols):
    for c in cols:
        df[c] = df[c].astype("string").fillna("missing").str.strip()
    return df


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype("int8")
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype("int8")
    for df in [train, test, orig]:
        df = sanitize_str(df, RAW_CATS)
        for c in RAW_NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    train_medians = train[RAW_NUMS].median()
    for df in [train, test, orig]:
        for c in RAW_NUMS:
            df[c] = df[c].fillna(train_medians[c])

    y = train[TARGET].values
    X_train = train.drop(columns=[TARGET]).copy()
    X_test = test.copy()
    log(f"Train shape: {train.shape} | Test shape: {test.shape} | Orig shape: {orig.shape}")

    combined = pd.concat([
        X_train.drop(columns=[ID]),
        X_test.drop(columns=[ID]),
        orig.drop(columns=[TARGET]),
    ], axis=0, ignore_index=True)
    for c in RAW_CATS:
        combined[c] = combined[c].astype("string").fillna("missing").str.strip()

    log("Feature engineering...")
    # frequencies over train + test + original (numerics rounded to 2 decimals)
    for c in RAW_CATS + RAW_NUMS:
        if c in RAW_NUMS:
            key = combined[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
            tr_key = X_train[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
            te_key = X_test[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
        else:
            key = combined[c].astype(str)
            tr_key = X_train[c].astype(str)
            te_key = X_test[c].astype(str)
        vc = key.value_counts(normalize=True)
        fname = f"FREQ_{c}"
        X_train[fname] = tr_key.map(vc).fillna(0).astype("float32")
        X_test[fname] = te_key.map(vc).fillna(0).astype("float32")

    # arithmetic interactions
    for df in [X_train, X_test]:
        df["water_balance"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"]).astype("float32")
        df["rain_to_sun_ratio"] = (df["Rainfall_mm"] / (df["Sunlight_Hours"] + 1e-3)).astype("float32")
        df["prev_to_area_ratio"] = (df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 1e-3)).astype("float32")
        df["moisture_to_humidity"] = (df["Soil_Moisture"] / (df["Humidity"] + 1e-3)).astype("float32")
        df["temp_humidity_interaction"] = (df["Temperature_C"] * df["Humidity"]).astype("float32")
        df["ec_ph_interaction"] = (df["Electrical_Conductivity"] * df["Soil_pH"]).astype("float32")

    # distribution against the original rows of each class
    for col in KEY_NUMS:
        orig_all = orig[col].values
        for cls_id, cls_name in enumerate(TARGET_NAMES):
            ref = orig.loc[orig[TARGET] == cls_id, col].values
            p_name = f"pctrank_{cls_name}_{col}"
            z_name = f"zscore_{cls_name}_{col}"
            X_train[p_name] = pctrank_against(X_train[col].values, ref)
            X_test[p_name] = pctrank_against(X_test[col].values, ref)
            X_train[z_name] = zscore_against(X_train[col].values, ref)
            X_test[z_name] = zscore_against(X_test[col].values, ref)
        o_name = f"pctrank_orig_{col}"
        X_train[o_name] = pctrank_against(X_train[col].values, orig_all)
        X_test[o_name] = pctrank_against(X_test[col].values, orig_all)

    # numerics as keys, digits, quantile bins
    te_source_cols = []
    for col in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C", "Sunlight_Hours", "Field_Area_hectare"]:
        round_digits = NUMERIC_ROUND_MAP.get(col, 2)
        cat_name = f"{col}_cat_te"
        X_train[cat_name] = X_train[col].round(round_digits).astype(str)
        X_test[cat_name] = X_test[col].round(round_digits).astype(str)
        te_source_cols.append(cat_name)

    for col in KEY_NUMS:
        vals_tr = X_train[col].values
        vals_te = X_test[col].values
        prefix = col.replace("_", "")
        X_train[f"{prefix}_mod10"] = (np.floor(vals_tr).astype(int) % 10).astype("float32")
        X_test[f"{prefix}_mod10"] = (np.floor(vals_te).astype(int) % 10).astype("float32")
        X_train[f"{prefix}_mod100"] = (np.floor(vals_tr).astype(int) % 100).astype("float32")
        X_test[f"{prefix}_mod100"] = (np.floor(vals_te).astype(int) % 100).astype("float32")
        X_train[f"{prefix}_frac100"] = np.round((vals_tr - np.floor(vals_tr)) * 100).astype("float32")
        X_test[f"{prefix}_frac100"] = np.round((vals_te - np.floor(vals_te)) * 100).astype("float32")
        X_train[f"{prefix}_is_round"] = ((vals_tr - np.floor(vals_tr)) < 0.005).astype("float32")
        X_test[f"{prefix}_is_round"] = ((vals_te - np.floor(vals_te)) < 0.005).astype("float32")

    for col in ["Rainfall_mm", "Previous_Irrigation_mm"]:
        q_name = f"{col}_qcut_te"
        combined_num = pd.concat([X_train[col], X_test[col]], axis=0)
        try:
            bins = min(500, max(10, combined_num.nunique()))
            _, edges = pd.qcut(combined_num, q=bins, retbins=True, duplicates="drop")
            X_train[q_name] = pd.cut(X_train[col], bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)
            X_test[q_name] = pd.cut(X_test[col], bins=edges, labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)
        except Exception:
            X_train[q_name] = "0"
            X_test[q_name] = "0"
        te_source_cols.append(q_name)

    # per-class original frequencies
    orig_tmp = orig.copy()
    for c in KEY_NUMS:
        orig_tmp[f"MAP_{c}"] = orig_tmp[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
        X_train[f"MAP_{c}"] = X_train[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
        X_test[f"MAP_{c}"] = X_test[c].round(NUMERIC_ROUND_MAP.get(c, 2)).astype(str)
    prior_cols = RAW_CATS + [f"MAP_{c}" for c in KEY_NUMS]
    orig_onehot = pd.get_dummies(orig_tmp[TARGET], prefix="cls")
    orig_gp_df = pd.concat([orig_tmp[prior_cols], orig_onehot], axis=1)
    orig_class_cols = list(orig_onehot.columns)
    for col in prior_cols:
        gp = orig_gp_df.groupby(col)[orig_class_cols].mean().reset_index()
        rename_map = {cc: f"ORIGP_{col}_{cc}" for cc in orig_class_cols}
        gp = gp.rename(columns=rename_map)
        X_train = X_train.merge(gp, on=col, how="left")
        X_test = X_test.merge(gp, on=col, how="left")
        for feat in rename_map.values():
            X_train[feat] = X_train[feat].fillna(1 / 3).astype("float32")
            X_test[feat] = X_test[feat].fillna(1 / 3).astype("float32")

    # category n-grams
    ngram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        n = f"BG_{c1}_{c2}_te"
        X_train[n] = X_train[c1].astype(str) + "_" + X_train[c2].astype(str)
        X_test[n] = X_test[c1].astype(str) + "_" + X_test[c2].astype(str)
        ngram_cols.append(n)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        n = f"TG_{c1}_{c2}_{c3}_te"
        X_train[n] = X_train[c1].astype(str) + "_" + X_train[c2].astype(str) + "_" + X_train[c3].astype(str)
        X_test[n] = X_test[c1].astype(str) + "_" + X_test[c2].astype(str) + "_" + X_test[c3].astype(str)
        ngram_cols.append(n)

    te_cols = list(dict.fromkeys(RAW_CATS + te_source_cols + ngram_cols))

    # categoricals factorized over train + test + original
    for c in RAW_CATS:
        key = pd.concat([X_train[c], X_test[c], orig[c]], axis=0, ignore_index=True).astype(str)
        codes, _ = pd.factorize(key, sort=False)
        new_c = f"CATINT_{c}"
        X_train[new_c] = codes[:len(X_train)].astype("int32")
        X_test[new_c] = codes[len(X_train):len(X_train) + len(X_test)].astype("int32")

    numeric_feature_cols = [
        c for c in X_train.columns
        if c not in RAW_CATS + te_cols + [ID] + [f"MAP_{c}" for c in KEY_NUMS]
    ]
    numeric_feature_cols = list(dict.fromkeys(numeric_feature_cols))
    log(f"TE columns: {len(te_cols)} | numeric feature cols before TE: {len(numeric_feature_cols)}")
    return X_train, X_test, y, te_cols, numeric_feature_cols


def main():
    X_train, X_test, y, te_cols, numeric_feature_cols = build_features()

    fold_ids = folds(y)
    oof = np.zeros((len(X_train), 3), dtype=np.float32)
    pred = np.zeros((len(X_test), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        t0 = time.time()
        log(f"===== Fold {k} =====")

        X_tr = X_train.iloc[tr_idx].copy().reset_index(drop=True)
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx].copy().reset_index(drop=True)
        y_va = y[va_idx]
        X_te = X_test.copy().reset_index(drop=True)

        te = TargetEncoder(cv=INNER_FOLDS, smooth="auto", shuffle=True, random_state=SEED)
        tr_te = te.fit_transform(X_tr[te_cols], y_tr).astype("float32")
        va_te = te.transform(X_va[te_cols]).astype("float32")
        te_te = te.transform(X_te[te_cols]).astype("float32")
        te_feature_names = [f"TEF_{i}" for i in range(tr_te.shape[1])]

        X_tr_model = pd.concat([X_tr[numeric_feature_cols].reset_index(drop=True), pd.DataFrame(tr_te, columns=te_feature_names)], axis=1)
        X_va_model = pd.concat([X_va[numeric_feature_cols].reset_index(drop=True), pd.DataFrame(va_te, columns=te_feature_names)], axis=1)
        X_te_model = pd.concat([X_te[numeric_feature_cols].reset_index(drop=True), pd.DataFrame(te_te, columns=te_feature_names)], axis=1)
        for df in [X_tr_model, X_va_model, X_te_model]:
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

        dtrain = lgb.Dataset(X_tr_model, label=y_tr)
        dvalid = lgb.Dataset(X_va_model, label=y_va, reference=dtrain)
        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(200)],
        )

        va_pred = model.predict(X_va_model, num_iteration=model.best_iteration).astype(np.float32)
        te_pred = model.predict(X_te_model, num_iteration=model.best_iteration).astype(np.float32)
        oof[va_idx] = va_pred
        pred += te_pred / N_FOLDS

        fold_ba = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
        log(f"Fold {k} balanced accuracy: {fold_ba:.6f} | time={(time.time() - t0) / 60:.1f} min")

        del X_tr, X_va, X_te, X_tr_model, X_va_model, X_te_model, dtrain, dvalid, model
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, pred)


if __name__ == "__main__":
    main()
