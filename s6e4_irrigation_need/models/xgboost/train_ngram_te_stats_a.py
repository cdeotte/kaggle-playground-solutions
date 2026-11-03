#!/usr/bin/env python
"""GPU XGBoost on original-data distribution features, numerics-as-categories, categorical n-grams and nested target
statistics (pair xgb_ngram_te_stats_a).

Label-free features (train / test / original rows):
- frequency of each numeric value (train + original + test); water-balance gap (rain + previous irrigation - soil
  moisture), moisture / water ratio, sunlight x temperature, humidity - temperature; managed-water count, mulch and
  rainfed flags;
- original-data class rates of every value of the 19 columns (unseen 1/3);
- Rainfall_mm percentile rank within the original Low / Medium / High / all rows, the |z| gap to the original High vs Low
  rows, Soil_Moisture minus the original Crop_Type mean; distances of Rainfall_mm to the 25 / 50 / 75 % quantiles of the
  original High and Low rows;
- the 11 numerics rounded to 2 decimals as categories, 15 bigrams of 6 categoricals and 4 trigrams.
Per fold (inner StratifiedKFold(5, shuffle, 42) of the training rows for the training-row values, all training rows for
the validation / test rows): std / min / max of the class id per level of the 19 categorical-like columns; class rates
of the 19 n-grams (unseen 1/3); sklearn TargetEncoder (cv 5, smooth 'auto', one binary encoder per class) class rates
of the 19 categorical-like columns; then the categorical-like columns and n-grams are dropped (272 model columns).
Model: XGBoost on GPU, lr 0.01, depth 5, subsample 0.81, colsample_bytree 0.32, min_child_weight 6, alpha 3.5,
lambda 1.29, gamma 0.79, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation fold, random_state 42.
Test = mean of the 5 fold models. Competition rows only (the original rows only feed the statistics).
The feature build takes about a minute; each fold about 3.5 minutes on one GPU.
Usage: python train_ngram_te_stats_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

PAIR = "xgb_ngram_te_stats_a"


class CFG:
    TARGET = TARGET
    ID_COL = ID

    N_FOLDS = 5
    INNER_FOLDS = 5
    RANDOM_SEED = 42

    XGB_PARAMS = {
        "n_estimators": 50000,
        "learning_rate": 0.01,
        "max_depth": 5,
        "subsample": 0.81,
        "colsample_bytree": 0.32,
        "min_child_weight": 6,
        "reg_alpha": 3.5,
        "reg_lambda": 1.29,
        "gamma": 0.79,
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 500,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "enable_categorical": True,
        "device": "cuda",
        "verbosity": 0,
    }


CLASS_ORDER = list(CLASSES)

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
MANAGEMENT_COLS = ["Irrigation_Type", "Water_Source", "Mulching_Used"]
STATS = ["std", "min", "max"]
TOP_CATS_FOR_NGRAM = ["Crop_Type", "Season", "Irrigation_Type", "Water_Source", "Region", "Soil_Type"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2


def load_data():
    log("Loading train/test/orig...")
    train = read_train()
    test = read_test()
    orig = read_original()

    train[CFG.TARGET] = train[CFG.TARGET].map(TARGET_MAP).astype("int8")
    orig[CFG.TARGET] = orig[CFG.TARGET].map(TARGET_MAP).astype("int8")

    train_ids = train[CFG.ID_COL].copy()
    test_ids = test[CFG.ID_COL].copy()

    log(f"train shape = {train.shape}, mem = {mem_mb(train):.1f} MB")
    log(f"test  shape = {test.shape}, mem = {mem_mb(test):.1f} MB")
    log(f"orig  shape = {orig.shape}, mem = {mem_mb(orig):.1f} MB")
    log(f"class counts = {train[CFG.TARGET].value_counts().to_dict()}")
    return train, test, orig, train_ids, test_ids


def cast_base_types(train, test, orig):
    log("Casting base column types...")
    for df in [train, test, orig]:
        for c in NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c in CATS:
            df[c] = df[c].astype(str)
    return train, test, orig


def add_block(df: pd.DataFrame, cols_dict: dict) -> pd.DataFrame:
    block = pd.DataFrame(cols_dict, index=df.index)
    return pd.concat([df, block], axis=1)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


def class_prob_map(orig: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = orig.groupby(col, observed=False)[CFG.TARGET].value_counts(normalize=True).unstack(fill_value=0.0)
    tmp = tmp.reindex(columns=[0, 1, 2], fill_value=0.0)
    tmp.columns = [f"ORIG_proba_{col}_{c}" for c in CLASS_ORDER]
    return tmp.astype("float32")


def build_features(train, test, orig):
    log("Building engineered features...")
    new_nums = []
    num_as_cat = []

    log("[1/8] Frequency encoding...")
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    log("[2/8] Arithmetic interactions...")
    for name, df in [("train", train), ("test", test), ("orig", orig)]:
        df2 = add_block(
            df,
            {
                "water_balance_gap": (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"]).astype("float32"),
                "moisture_to_water_ratio": (df["Soil_Moisture"] / (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] + 1)).astype("float32"),
                "sun_temp_interaction": (df["Sunlight_Hours"] * df["Temperature_C"]).astype("float32"),
                "humidity_temp_gap": (df["Humidity"] - df["Temperature_C"]).astype("float32"),
            },
        )
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["water_balance_gap", "moisture_to_water_ratio", "sun_temp_interaction", "humidity_temp_gap"]

    log("[3/8] Management count features...")
    for name, df in [("train", train), ("test", test), ("orig", orig)]:
        df2 = add_block(
            df,
            {
                "managed_water_count": (df[MANAGEMENT_COLS].astype(str) != "No").sum(axis=1).astype("float32"),
                "uses_mulch": (df["Mulching_Used"] == "Yes").astype("float32"),
                "rainfed_flag": (df["Irrigation_Type"] == "Rainfed").astype("float32"),
            },
        )
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["managed_water_count", "uses_mulch", "rainfed_flag"]

    log("[4/8] ORIG probability mappings...")
    for col in CATS + NUMS:
        tmp = class_prob_map(orig, col)
        for cls_col in tmp.columns:
            train[cls_col] = train[col].map(tmp[cls_col]).fillna(1 / 3).astype("float32")
            test[cls_col] = test[col].map(tmp[cls_col]).fillna(1 / 3).astype("float32")
            new_nums.append(cls_col)

    log("[5/8] Distribution features...")
    orig_high = orig.loc[orig[CFG.TARGET] == 2, "Rainfall_mm"].values
    orig_med = orig.loc[orig[CFG.TARGET] == 1, "Rainfall_mm"].values
    orig_low = orig.loc[orig[CFG.TARGET] == 0, "Rainfall_mm"].values
    orig_all = orig["Rainfall_mm"].values
    orig_crop_moist = orig.groupby("Crop_Type", observed=False)["Soil_Moisture"].mean()

    for name, df in [("train", train), ("test", test)]:
        rain = df["Rainfall_mm"].values
        vals = {
            "pctrank_low_rain": pctrank_against(rain, orig_low),
            "pctrank_med_rain": pctrank_against(rain, orig_med),
            "pctrank_high_rain": pctrank_against(rain, orig_high),
            "pctrank_orig_rain": pctrank_against(rain, orig_all),
            "zscore_high_gap_rain": (
                np.abs(zscore_against(rain, orig_high)) - np.abs(zscore_against(rain, orig_low))
            ).astype("float32"),
            "resid_crop_moisture": (df["Soil_Moisture"].values - df["Crop_Type"].map(orig_crop_moist).fillna(0).values).astype("float32"),
        }
        df2 = add_block(df, vals)
        if name == "train":
            train = df2
        else:
            test = df2
    new_nums += ["pctrank_low_rain", "pctrank_med_rain", "pctrank_high_rain", "pctrank_orig_rain", "zscore_high_gap_rain", "resid_crop_moisture"]

    log("[6/8] Quantile distance features...")
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        hi_q = np.quantile(orig_high, q_val)
        lo_q = np.quantile(orig_low, q_val)
        for name, df in [("train", train), ("test", test)]:
            vals = {
                f"dist_to_hi_{q_label}": np.abs(df["Rainfall_mm"] - hi_q).astype("float32"),
                f"dist_to_lo_{q_label}": np.abs(df["Rainfall_mm"] - lo_q).astype("float32"),
                f"qdist_gap_{q_label}": (np.abs(df["Rainfall_mm"] - lo_q) - np.abs(df["Rainfall_mm"] - hi_q)).astype("float32"),
            }
            df2 = add_block(df, vals)
            if name == "train":
                train = df2
            else:
                test = df2
    new_nums += [
        "dist_to_hi_q25", "dist_to_lo_q25", "qdist_gap_q25",
        "dist_to_hi_q50", "dist_to_lo_q50", "qdist_gap_q50",
        "dist_to_hi_q75", "dist_to_lo_q75", "qdist_gap_q75",
    ]

    log("[7/8] Numericals as categories...")
    for col in NUMS:
        new_col = f"CAT_{col}"
        train[new_col] = train[col].round(2).astype(str).astype("category")
        test[new_col] = test[col].round(2).astype(str).astype("category")
        orig[new_col] = orig[col].round(2).astype(str).astype("category")
        num_as_cat.append(new_col)

    log("[8/8] Bi-grams / tri-grams...")
    bigram_cols = []
    trigram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        train[col_name] = (train[c1].astype(str) + "_" + train[c2].astype(str)).astype("category")
        test[col_name] = (test[c1].astype(str) + "_" + test[c2].astype(str)).astype("category")
        bigram_cols.append(col_name)

    tri_defs = [
        ("Crop_Type", "Season", "Irrigation_Type"),
        ("Soil_Type", "Crop_Type", "Region"),
        ("Water_Source", "Irrigation_Type", "Region"),
        ("Crop_Growth_Stage", "Season", "Crop_Type"),
    ]
    for c1, c2, c3 in tri_defs:
        col_name = f"TG_{c1}_{c2}_{c3}"
        train[col_name] = (train[c1].astype(str) + "_" + train[c2].astype(str) + "_" + train[c3].astype(str)).astype("category")
        test[col_name] = (test[c1].astype(str) + "_" + test[c2].astype(str) + "_" + test[c3].astype(str)).astype("category")
        trigram_cols.append(col_name)

    ngram_cols = bigram_cols + trigram_cols

    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    te_columns = num_as_cat + CATS
    te_ngram_columns = ngram_cols
    to_remove = num_as_cat + CATS + ngram_cols

    log("Feature engineering complete")
    log(f"  numerical base   = {len(NUMS)}")
    log(f"  categorical base = {len(CATS)}")
    log(f"  engineered nums  = {len(new_nums)}")
    log(f"  num-as-cat       = {len(num_as_cat)}")
    log(f"  bigrams          = {len(bigram_cols)}")
    log(f"  trigrams         = {len(trigram_cols)}")
    log(f"  total FEATURES   = {len(features)}")

    return train, test, orig, features, te_columns, te_ngram_columns, to_remove, bigram_cols, trigram_cols


def multiclass_stats(df: pd.DataFrame, col: str):
    tmp = df.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
    tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
    return tmp.astype("float32")


def run_fold(fold, train_idx, val_idx, train, test, features, te_columns, te_ngram_columns, to_remove):
    """fold: 1..5."""
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    t0 = time.time()
    log("=" * 70)
    log(f"Fold {fold}/{CFG.N_FOLDS}")
    log("=" * 70)

    x_tr = train.loc[train_idx, features + [CFG.TARGET]].reset_index(drop=True).copy()
    y_tr = train.loc[train_idx, CFG.TARGET].values
    x_val = train.loc[val_idx, features].reset_index(drop=True).copy()
    y_val = train.loc[val_idx, CFG.TARGET].values
    x_te = test[features].reset_index(drop=True).copy()

    for in_tr, in_va in skf_inner.split(x_tr, y_tr):
        x_tr2 = x_tr.loc[in_tr, features + [CFG.TARGET]].copy()
        x_va2 = x_tr.loc[in_va, features].copy()
        for col in te_columns:
            tmp = multiclass_stats(x_tr2, col)
            x_va2 = x_va2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                x_tr.loc[in_va, c] = x_va2[c].values.astype("float32")

    for col in te_columns:
        tmp = multiclass_stats(x_tr, col)
        x_val = x_val.merge(tmp, on=col, how="left")
        x_te = x_te.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            x_tr[c] = x_tr[c].fillna(0).astype("float32")
            x_val[c] = x_val[c].fillna(0).astype("float32")
            x_te[c] = x_te[c].fillna(0).astype("float32")

    for in_tr, in_va in skf_inner.split(x_tr, y_tr):
        x_tr2 = x_tr.loc[in_tr].copy()
        x_va2 = x_tr.loc[in_va].copy()
        for col in te_ngram_columns:
            ng_name = f"TE_ng_{col}"
            for cls in [0, 1, 2]:
                cls_map = (x_tr2[CFG.TARGET] == cls).groupby(x_tr2[col], observed=False).mean()
                mapped = x_va2[col].astype(str).map(cls_map)
                x_tr.loc[in_va, f"{ng_name}_{cls}"] = pd.to_numeric(mapped, errors="coerce").fillna(1 / 3).astype("float32").values

    for col in te_ngram_columns:
        ng_name = f"TE_ng_{col}"
        for cls in [0, 1, 2]:
            cls_map = (x_tr[CFG.TARGET] == cls).groupby(x_tr[col], observed=False).mean()
            x_val[f"{ng_name}_{cls}"] = pd.to_numeric(x_val[col].astype(str).map(cls_map), errors="coerce").fillna(1 / 3).astype("float32")
            x_te[f"{ng_name}_{cls}"] = pd.to_numeric(x_te[col].astype(str).map(cls_map), errors="coerce").fillna(1 / 3).astype("float32")
            if f"{ng_name}_{cls}" not in x_tr.columns:
                x_tr[f"{ng_name}_{cls}"] = np.float32(1 / 3)
            else:
                x_tr[f"{ng_name}_{cls}"] = pd.to_numeric(x_tr[f"{ng_name}_{cls}"], errors="coerce").fillna(1 / 3).astype("float32")

    te_mean_blocks = []
    for cls in [0, 1, 2]:
        te = TargetEncoder(
            cv=CFG.INNER_FOLDS,
            shuffle=True,
            smooth="auto",
            target_type="binary",
            random_state=CFG.RANDOM_SEED,
        )
        y_bin = (y_tr == cls).astype(np.int8)
        te_cols = [f"TE_{col}_{cls}" for col in te_columns]
        x_tr[te_cols] = te.fit_transform(x_tr[te_columns], y_bin)
        x_val[te_cols] = te.transform(x_val[te_columns])
        x_te[te_cols] = te.transform(x_te[te_columns])
        te_mean_blocks.extend(te_cols)

    for df in [x_tr, x_val, x_te]:
        for c in CATS + [f"CAT_{x}" for x in NUMS]:
            if c in df.columns:
                df[c] = df[c].astype(str).astype("category")
        df.drop(columns=[c for c in to_remove if c in df.columns], inplace=True, errors="ignore")

    x_tr.drop(columns=[CFG.TARGET], inplace=True, errors="ignore")
    cols_xgb = x_tr.columns.tolist()

    if fold == 1:
        n_ngram_feats = sum(c.startswith("TE_ng_") for c in cols_xgb)
        log(f"XGB feature count = {len(cols_xgb)}")
        log(f"N-gram TE feature count = {n_ngram_feats}")
        log(f"Class order for saved probabilities = {CLASS_ORDER}")

    model = xgb.XGBClassifier(**CFG.XGB_PARAMS)
    model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=1000)

    p_val = model.predict_proba(x_val).astype("float32")
    p_te = model.predict_proba(x_te[cols_xgb]).astype("float32")

    fold_bac = balanced_accuracy_score(y_val, p_val.argmax(axis=1))
    log(f"{PAIR} fold {fold - 1}: BAC = {fold_bac:.6f} | elapsed = {(time.time() - t0) / 60:.1f} min")
    del x_tr, x_val, x_te, y_tr, y_val, model
    gc.collect()
    return p_val, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, orig, train_ids, test_ids = load_data()
        train, test, orig = cast_base_types(train, test, orig)
        train, test, orig, features, te_columns, te_ngram_columns, to_remove, _, _ = build_features(train, test, orig)
        log("Training XGBoost with nested TE...")
        np.random.seed(CFG.RANDOM_SEED)
        fold_id = folds(train[CFG.TARGET].values)
        for k in todo:
            p_val, p_te = run_fold(k + 1, np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k), train, test,
                                   features, te_columns, te_ngram_columns, to_remove)
            save_part(HERE, PAIR, k, p_val, p_te)
    y = read_train()[TARGET].map(TARGET_MAP).astype("int8").values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
