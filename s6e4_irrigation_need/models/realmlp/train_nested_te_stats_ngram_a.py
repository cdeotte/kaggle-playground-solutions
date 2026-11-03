#!/usr/bin/env python
"""RealMLP on engineered numerics, numeric-as-category values, categorical bi- / tri-grams and nested multiclass target
encodings (pair realmlp_nested_te_stats_ngram_a).

Features (competition rows only in training; the original rows feed priors and distribution features):
- the 19 raw columns; frequency of the 11 numeric values over train + original + test;
- 5 arithmetic interactions (water deviation, rain / previous irrigation, water per area, temperature x sunlight,
  humidity x moisture) and 3 practice flags (mulch count, mulching, surface water source);
- original target mean (labels 0 / 1 / 2) per value of the 19 raw columns;
- Previous_Irrigation_mm against the original rows: percentile rank within the Low / Medium / High / all values,
  z-score gap High vs Low, distances to the per-class quartiles (q25 / q50 / q75) and their Low - High gaps;
  Rainfall_mm minus the original Crop_Type mean rainfall;
- digit features (first / second / last digit, mod 10, mod 100, number of digits, rounded to 10, fraction, distance to
  the rounded value) of Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Temperature_C;
- Rainfall_mm, Previous_Irrigation_mm, Soil_Moisture, Humidity, Temperature_C, Sunlight_Hours rounded to 2 decimals as
  categories; 15 bigrams of 6 categoricals and 4 trigrams of the first 4 of them, as categories.
Per fold, target encodings fitted out-of-fold over 5 inner folds (seed 42) for the training rows and on all training
rows for the validation and test rows: std / min / max of the label and Laplace-smoothed class probabilities for the
6 numeric-as-category columns and the 8 categoricals, class probabilities for the 19 n-grams (141 columns).
Model: pytabkit RealMLP_TD_Classifier, 4 x 512 mish, PBLD numeric embeddings, n_ens 1, lr 0.13, batch 1024, up to
64 epochs with early stopping on the validation fold (classification error), random_state 42.
Test = mean of the 5 fold models.
Usage: python train_nested_te_stats_ngram_a.py"""
import gc
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import (CLASSES, N_FOLDS, TARGET, TARGET_MAP, folds, model_dir, read_original, read_test,  # noqa: E402
                         read_train)
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_nested_te_stats_ngram_a"
INNER_FOLDS = 5
RANDOM_SEED = 42
CLASS_NAMES = list(CLASSES)

REALMLP_PARAMS = {
    "act": "mish",
    "ls_eps_sched": "coslog4",
    "p_drop_sched": "flat_cos",
    "wd_sched": "flat_cos",
    "embedding_size": 4,
    "bias_wd_factor": 0.0,
    "weight_param": "ntk",
    "bias_lr_factor": 0.1,
    "use_parametric_act": True,
    "add_front_scale": True,
    "act_lr_factor": 0.1,
    "block_str": "w-b-a-d",
    "hidden_sizes": "rectangular",
    "bias_init_mode": "he+5",
    "weight_init_mode": "std",
    "val_metric_name": "class_error",
    "num_emb_type": "pbld",
    "verbosity": 2,
    "batch_size": 1024,
    "random_state": RANDOM_SEED,
    "n_epochs": 64,
    "early_stopping_additive_patience": 24,
    "early_stopping_multiplicative_patience": 2,
    "n_ens": 1,
    "ens_av_before_softmax": False,
    "n_hidden_layers": 4,
    "hidden_width": 512,
    "lr": 0.1301052035689966,
    "wd": 0.04914969178803988,
    "p_drop": 0.02682248660489578,
    "scale_lr_factor": 2.1896500647074553,
    "use_ls": False,
    "ls_eps": 0.04088400565558735,
    "plr_sigma": 1.668950443527603,
    "plr_lr_factor": 0.27098419375438565,
    "plr_hidden_1": 8,
    "plr_hidden_2": 8,
    "sq_mom": 0.9855686567493073,
    "first_layer_lr_factor": 1.3750227365258307,
    "use_early_stopping": True,
    "max_one_hot_cat_size": 15,
}

TOP_CATS_FOR_NGRAM = [
    "Crop_Type", "Season", "Irrigation_Type",
    "Water_Source", "Region", "Soil_Type",
]

CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
STATS = ["std", "min", "max"]
ARTIFACT_NUMS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C", "Sunlight_Hours"]
DIST_NUM = "Previous_Irrigation_mm"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    log("Loading train/test/orig...")
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype("int8")
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype("int8")
    log(f"train shape = {train.shape}")
    log(f"test  shape = {test.shape}")
    log(f"orig  shape = {orig.shape}")
    print("Class order:", CLASS_NAMES)
    return train, test, orig


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
    return (np.searchsorted(ref_sorted, values) / max(len(reference), 1)).astype("float32")


def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


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
        df2 = add_block(df, {
            "water_deviation": (df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["Soil_Moisture"] * 2.0).astype("float32"),
            "rain_to_prev_ratio": (df["Rainfall_mm"] / (df["Previous_Irrigation_mm"] + 1)).astype("float32"),
            "avg_water_per_area": ((df["Rainfall_mm"] + df["Previous_Irrigation_mm"]) / (df["Field_Area_hectare"] + 1)).astype("float32"),
            "temp_x_sun": (df["Temperature_C"] * df["Sunlight_Hours"]).astype("float32"),
            "humidity_x_moisture": (df["Humidity"] * df["Soil_Moisture"]).astype("float32"),
        })
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["water_deviation", "rain_to_prev_ratio", "avg_water_per_area", "temp_x_sun", "humidity_x_moisture"]

    log("[3/8] Practice flags...")
    for name, df in [("train", train), ("test", test), ("orig", orig)]:
        df2 = add_block(df, {
            "mulch_count": (df["Mulching_Used"].eq("Yes").astype(int) + 1).astype("float32"),
            "has_mulch": (df["Mulching_Used"] == "Yes").astype("float32"),
            "has_surface_water": df["Water_Source"].isin(["River", "Canal", "Pond"]).astype("float32"),
        })
        if name == "train":
            train = df2
        elif name == "test":
            test = df2
        else:
            orig = df2
    new_nums += ["mulch_count", "has_mulch", "has_surface_water"]

    log("[4/8] ORIG_proba mappings...")
    for col in CATS + NUMS:
        tmp = orig.groupby(col, observed=False)[TARGET].mean()
        new_col = f"ORIG_proba_{col}"
        train[new_col] = train[col].map(tmp).fillna(orig[TARGET].mean()).astype("float32")
        test[new_col] = test[col].map(tmp).fillna(orig[TARGET].mean()).astype("float32")
        new_nums.append(new_col)

    log("[5/8] Distribution features...")
    orig_low = orig.loc[orig[TARGET] == 0, DIST_NUM].values
    orig_med = orig.loc[orig[TARGET] == 1, DIST_NUM].values
    orig_high = orig.loc[orig[TARGET] == 2, DIST_NUM].values
    orig_all = orig[DIST_NUM].values
    orig_crop_mean = orig.groupby("Crop_Type", observed=False)["Rainfall_mm"].mean()
    for name, df in [("train", train), ("test", test)]:
        vals = df[DIST_NUM].values
        out = {
            "pctrank_low_dist": pctrank_against(vals, orig_low),
            "pctrank_med_dist": pctrank_against(vals, orig_med),
            "pctrank_high_dist": pctrank_against(vals, orig_high),
            "pctrank_orig_dist": pctrank_against(vals, orig_all),
            "zscore_high_gap": (np.abs(zscore_against(vals, orig_high)) - np.abs(zscore_against(vals, orig_low))).astype("float32"),
            "resid_crop_rain": (df["Rainfall_mm"].values - df["Crop_Type"].map(orig_crop_mean).fillna(orig["Rainfall_mm"].mean()).values).astype("float32"),
        }
        df2 = add_block(df, out)
        if name == "train":
            train = df2
        else:
            test = df2
    new_nums += ["pctrank_low_dist", "pctrank_med_dist", "pctrank_high_dist", "pctrank_orig_dist", "zscore_high_gap", "resid_crop_rain"]

    log("[6/8] Quantile distance features...")
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        low_q = np.quantile(orig_low, q_val)
        high_q = np.quantile(orig_high, q_val)
        med_q = np.quantile(orig_med, q_val)
        for name, df in [("train", train), ("test", test)]:
            df2 = add_block(df, {
                f"dist_To_low_{q_label}": np.abs(df[DIST_NUM] - low_q).astype("float32"),
                f"dist_To_med_{q_label}": np.abs(df[DIST_NUM] - med_q).astype("float32"),
                f"dist_To_high_{q_label}": np.abs(df[DIST_NUM] - high_q).astype("float32"),
                f"qdist_gap_To_{q_label}": (np.abs(df[DIST_NUM] - low_q) - np.abs(df[DIST_NUM] - high_q)).astype("float32"),
            })
            if name == "train":
                train = df2
            else:
                test = df2
    new_nums += [
        "dist_To_low_q25", "dist_To_med_q25", "dist_To_high_q25", "qdist_gap_To_q25",
        "dist_To_low_q50", "dist_To_med_q50", "dist_To_high_q50", "qdist_gap_To_q50",
        "dist_To_low_q75", "dist_To_med_q75", "dist_To_high_q75", "qdist_gap_To_q75",
    ]

    log("[7/8] Numericals-as-categories...")
    for col in ARTIFACT_NUMS:
        new_col = f"CAT_{col}"
        train[new_col] = train[col].round(2).astype(str).astype("category")
        test[new_col] = test[col].round(2).astype(str).astype("category")
        num_as_cat.append(new_col)

    log("[8/8] Digit features...")
    digit_features = []
    for base_name, col in [("rain", "Rainfall_mm"), ("prev", "Previous_Irrigation_mm"), ("moist", "Soil_Moisture"), ("temp", "Temperature_C")]:
        digit_features += [
            f"{base_name}_first_digit", f"{base_name}_last_digit", f"{base_name}_second_digit",
            f"{base_name}_mod10", f"{base_name}_mod100", f"{base_name}_num_digits",
            f"{base_name}_rounded10", f"{base_name}_fractional", f"{base_name}_dev_round10",
        ]
    for name, df in [("train", train), ("test", test)]:
        vals = {}
        for base_name, col in [("rain", "Rainfall_mm"), ("prev", "Previous_Irrigation_mm"), ("moist", "Soil_Moisture"), ("temp", "Temperature_C")]:
            floor = np.floor(df[col].fillna(0)).astype(int)
            s = df[col].fillna(0).astype(str).str.replace(".", "", regex=False)
            vals[f"{base_name}_first_digit"] = s.str[0].replace("-", "0").fillna("0").astype(int)
            vals[f"{base_name}_last_digit"] = s.str[-1].replace("-", "0").fillna("0").astype(int)
            vals[f"{base_name}_second_digit"] = s.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype("int16")
            vals[f"{base_name}_mod10"] = (floor % 10).astype("float32")
            vals[f"{base_name}_mod100"] = (floor % 100).astype("float32")
            vals[f"{base_name}_num_digits"] = floor.astype(str).str.len().astype("int16")
            vals[f"{base_name}_rounded10"] = (np.round(df[col] / 10) * 10).astype("float32")
            vals[f"{base_name}_fractional"] = (df[col] - np.floor(df[col])).astype("float32")
            vals[f"{base_name}_dev_round10"] = np.abs(df[col] - np.round(df[col] / 10) * 10).astype("float32")
        df2 = add_block(df, vals)
        if name == "train":
            train = df2
        else:
            test = df2
    new_nums += digit_features

    log("Creating bi-gram / tri-gram categorical features...")
    bigram_cols = []
    trigram_cols = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        train[col_name] = (train[c1].astype(str) + "_" + train[c2].astype(str)).astype("category")
        test[col_name] = (test[c1].astype(str) + "_" + test[c2].astype(str)).astype("category")
        bigram_cols.append(col_name)
    top4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        train[col_name] = (train[c1].astype(str) + "_" + train[c2].astype(str) + "_" + train[c3].astype(str)).astype("category")
        test[col_name] = (test[c1].astype(str) + "_" + test[c2].astype(str) + "_" + test[c3].astype(str)).astype("category")
        trigram_cols.append(col_name)

    ngram_cols = bigram_cols + trigram_cols
    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    te_columns = num_as_cat + CATS
    te_ngram_columns = ngram_cols
    log(f"total FEATURES = {len(features)}")
    return train, test, orig, features, te_columns, te_ngram_columns


def make_realmlp_params():
    params = dict(REALMLP_PARAMS)
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return params


def get_cat_cols_for_realmlp(features, te_ngram_columns):
    out = []
    seen = set()
    for c in (CATS + [f"CAT_{x}" for x in ARTIFACT_NUMS] + te_ngram_columns):
        if c in features and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def finalize_fold_frames(X_tr, X_val, X_te, features, cat_cols):
    use_cols = []
    seen = set()
    for c in features:
        if c not in seen:
            use_cols.append(c)
            seen.add(c)
    X_tr = X_tr[use_cols].copy()
    X_val = X_val[use_cols].copy()
    X_te = X_te[use_cols].copy()
    for c in use_cols:
        if c in cat_cols:
            X_tr[c] = X_tr[c].astype(str).fillna("missing").astype("category")
            X_val[c] = X_val[c].astype(str).fillna("missing").astype("category")
            X_te[c] = X_te[c].astype(str).fillna("missing").astype("category")
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype("float32").fillna(0)
            X_val[c] = pd.to_numeric(X_val[c], errors="coerce").astype("float32").fillna(0)
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype("float32").fillna(0)
    return X_tr, X_val, X_te


def fit_class_maps(series, y):
    """Laplace-smoothed class probabilities per key (+1 per class) and the global class frequencies."""
    df = pd.DataFrame({"key": series.astype(str).values, "y": y})
    counts = df.groupby(["key", "y"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(len(CLASS_NAMES)), fill_value=0)
    probs = (counts + 1.0).div(counts.sum(axis=1) + len(CLASS_NAMES), axis=0)
    global_probs = pd.Series(np.bincount(y, minlength=len(CLASS_NAMES)) / len(y), index=range(len(CLASS_NAMES)))
    return probs, global_probs


def apply_class_maps(series, probs, global_probs, prefix):
    mapped = pd.DataFrame(index=series.index)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        col = probs[cls_idx].to_dict()
        mapped[f"{prefix}_{cls_name}"] = series.astype(str).map(col).fillna(global_probs[cls_idx]).astype("float32")
    return mapped


def train_model(train, test, features, te_columns, te_ngram_columns):
    log("Training RealMLP with nested multiclass TE pipeline...")
    np.random.seed(RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cat_cols = get_cat_cols_for_realmlp(features, te_ngram_columns)
    te_stat_cols = [f"TE1_{col}_{s}" for col in te_columns for s in STATS]
    te_prob_cols = [f"TE_{col}_{cls}" for col in te_columns for cls in CLASS_NAMES]
    te_ng_cols = [f"TE_ng_{col}_{cls}" for col in te_ngram_columns for cls in CLASS_NAMES]
    features_final = list(features) + te_stat_cols + te_prob_cols + te_ng_cols
    y = train[TARGET].values.astype(np.int32)
    fold_id = folds(y)
    oof = np.zeros((len(train), len(CLASS_NAMES)), dtype="float32")
    pred = np.zeros((len(test), len(CLASS_NAMES)), dtype="float32")
    params = make_realmlp_params()
    t0 = time.time()

    for fold in range(1, N_FOLDS + 1):
        train_idx, val_idx = np.where(fold_id != fold - 1)[0], np.where(fold_id == fold - 1)[0]
        log("=" * 70)
        log(f"Fold {fold}/{N_FOLDS}")
        X_tr = train.loc[train_idx, features + [TARGET]].reset_index(drop=True).copy()
        y_tr = y[train_idx]
        X_val = train.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = y[val_idx]
        X_te = test[features].reset_index(drop=True).copy()

        # inner out-of-fold encodings of the training rows: label std / min / max and class probabilities
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr, features + [TARGET]].copy()
            X_va2 = X_tr.loc[in_va, features].copy()
            for col in te_columns:
                stats_tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                stats_tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va_merge = X_va2[[col]].merge(stats_tmp, on=col, how="left")
                for c in stats_tmp.columns:
                    X_tr.loc[in_va, c] = X_va_merge[c].fillna(0).astype("float32").values
                probs, global_probs = fit_class_maps(X_tr2[col], X_tr2[TARGET].values)
                mapped = apply_class_maps(X_va2[col], probs, global_probs, f"TE_{col}")
                for c in mapped.columns:
                    X_tr.loc[in_va, c] = mapped[c].values.astype("float32")

        # validation / test rows: encodings fitted on all training rows
        for col in te_columns:
            stats_tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            stats_tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            X_val = X_val.merge(stats_tmp, on=col, how="left")
            X_te = X_te.merge(stats_tmp, on=col, how="left")
            for c in stats_tmp.columns:
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(0).astype("float32")
                X_val[c] = pd.to_numeric(X_val[c], errors="coerce").fillna(0).astype("float32")
                X_te[c] = pd.to_numeric(X_te[c], errors="coerce").fillna(0).astype("float32")
            probs, global_probs = fit_class_maps(X_tr[col], y_tr)
            val_map = apply_class_maps(X_val[col], probs, global_probs, f"TE_{col}")
            te_map = apply_class_maps(X_te[col], probs, global_probs, f"TE_{col}")
            for cls_name in CLASS_NAMES:
                c = f"TE_{col}_{cls_name}"
                if c not in X_tr.columns:
                    X_tr[c] = 0.0
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(float(global_probs[CLASS_NAMES.index(cls_name)])).astype("float32")
                X_val[c] = val_map[c].astype("float32")
                X_te[c] = te_map[c].astype("float32")

        # n-gram class probabilities: inner out-of-fold for the training rows, full fit for validation / test
        for in_tr, in_va in skf_inner.split(X_tr, y_tr):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in te_ngram_columns:
                probs, global_probs = fit_class_maps(X_tr2[col], y_tr[in_tr])
                mapped = apply_class_maps(X_va2[col], probs, global_probs, f"TE_ng_{col}")
                for c in mapped.columns:
                    X_tr.loc[in_va, c] = mapped[c].values.astype("float32")

        for col in te_ngram_columns:
            probs, global_probs = fit_class_maps(X_tr[col], y_tr)
            val_map = apply_class_maps(X_val[col], probs, global_probs, f"TE_ng_{col}")
            te_map = apply_class_maps(X_te[col], probs, global_probs, f"TE_ng_{col}")
            for cls_name in CLASS_NAMES:
                c = f"TE_ng_{col}_{cls_name}"
                if c not in X_tr.columns:
                    X_tr[c] = 0.0
                X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(float(global_probs[CLASS_NAMES.index(cls_name)])).astype("float32")
                X_val[c] = val_map[c].astype("float32")
                X_te[c] = te_map[c].astype("float32")

        X_tr, X_val, X_te = finalize_fold_frames(X_tr, X_val, X_te, features_final, cat_cols)
        log(f"Fold matrices: X_tr={X_tr.shape}, X_val={X_val.shape}, X_te={X_te.shape}")
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, cat_col_names=cat_cols)
        val_pred = model.predict_proba(X_val).astype("float32")
        test_pred = model.predict_proba(X_te).astype("float32")
        oof[val_idx] = val_pred
        pred += test_pred / N_FOLDS
        fold_bac = balanced_accuracy_score(y_val, val_pred.argmax(axis=1))
        log(f"Fold {fold} BAC = {fold_bac:.8f}")
        del model, X_tr, X_val, X_te, val_pred, test_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_bac = balanced_accuracy_score(y, oof.argmax(axis=1))
    elapsed = (time.time() - t0) / 60
    log("=" * 70)
    log(f"Overall OOF BAC = {overall_bac:.8f}")
    log(f"Total time = {elapsed:.2f} minutes")
    return oof, pred


def main():
    train, test, orig = load_data()
    train, test, orig = cast_base_types(train, test, orig)
    train, test, orig, features, te_columns, te_ngram_columns = build_features(train, test, orig)
    oof, pred = train_model(train, test, features, te_columns, te_ngram_columns)
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
