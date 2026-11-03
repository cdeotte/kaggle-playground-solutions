#!/usr/bin/env python
"""Extra-trees LightGBM on rounding / binning / digit artifacts with a two-stage nested target encoding
(pair lgbm_extra_trees_round_digit_two_stage_te_a).

Label-free features (the 8 categoricals are pandas categories over train + test + original, native in LightGBM):
  - the 11 numerics and 8 categoricals
  - rounded values cast to integers: rainfall and previous irrigation rounded to 1 / 10 / 100, soil moisture,
    temperature and sunlight to 0.1 / 1, humidity to 0.1 / 1 / 10
  - floor bins: rainfall and previous irrigation by 5 / 10 / 20 / 50, soil moisture by 5 / 10 / 20, temperature by
    2 / 5, humidity by 5 / 10, sunlight by 1 / 2
  - 3 integer and 3 decimal digits of rainfall, previous irrigation, soil moisture, temperature; 3 digit pairs within
    each of them and 2 decimal-digit pairs across them (as integers)
  - all 28 category pairs (strings)
  - for the categoricals, rounded values, bins and digit pairs: original row count and per-class original frequency
Per fold, class frequencies smoothed toward the class prior ((count x freq + 20 x prior) / (count + 20)), out-of-fold
over an inner StratifiedKFold(5, seed 42) of the training rows and from all training rows for the validation / test
rows: stage 1 encodes the 28 category pairs and 14 digit pairs and drops them; stage 2 encodes 5 numerics, the digits,
rounded values, bins and original-prior columns and keeps them.
Model: LightGBM multiclass, extra_trees, 141 leaves, max_depth 7, lr 0.01, bagging_fraction 0.7, feature_fraction 0.17,
colsample_bynode 0.925, min_data_in_leaf 63, lambda_l1 3.3, lambda_l2 5.7, max_bin 1008, seed 42, 32 threads, up to
20,000 rounds with early stopping (500) on the validation fold. Test = mean of the 5 fold models.
Usage: python train_extra_trees_two_stage_te.py"""
import gc
import itertools
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.simplefilter("ignore")

PAIR = "lgbm_extra_trees_round_digit_two_stage_te_a"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5

CAT_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
NUM_COLS = [
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

ROUND_SPECS = {
    "Rainfall_mm": [0, -1, -2],
    "Previous_Irrigation_mm": [0, -1, -2],
    "Soil_Moisture": [1, 0],
    "Temperature_C": [1, 0],
    "Humidity": [1, 0, -1],
    "Sunlight_Hours": [1, 0],
}
BIN_SPECS = {
    "Rainfall_mm": [5, 10, 20, 50],
    "Previous_Irrigation_mm": [5, 10, 20, 50],
    "Soil_Moisture": [5, 10, 20],
    "Temperature_C": [2, 5],
    "Humidity": [5, 10],
    "Sunlight_Hours": [1, 2],
}
DIGIT_BASE_COLS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C"]
MAX_CATPAIR = 28
TE_SMOOTH = 20.0

LGB_PARAMS = dict(
    objective="multiclass",
    metric="multi_logloss",
    num_class=3,
    boosting_type="gbdt",
    n_estimators=20000,
    num_leaves=141,
    max_depth=7,
    learning_rate=0.01,
    bagging_fraction=0.7,
    colsample_bynode=0.925,
    feature_fraction=0.17,
    min_data_in_leaf=63,
    lambda_l1=3.3,
    lambda_l2=5.7,
    extra_trees=True,
    max_bin=1008,
    n_jobs=32,
    verbosity=-1,
    random_state=SEED,
)
EARLY_STOPPING_ROUNDS = 500
LOG_EVERY = 500


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def label_encode_across_all(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols):
    for c in cols:
        all_vals = pd.concat(
            [train[c].astype("string"), test[c].astype("string"), orig[c].astype("string")],
            ignore_index=True,
        ).fillna("__MISSING__")
        cat = pd.Categorical(all_vals)
        train[c] = pd.Categorical(all_vals[: len(train)], categories=cat.categories)
        test[c] = pd.Categorical(all_vals[len(train): len(train) + len(test)], categories=cat.categories)
        orig[c] = pd.Categorical(all_vals[len(train) + len(test):], categories=cat.categories)
    return train, test, orig


def make_digits_from_float(s: pd.Series, keep_non=3, keep_dec=3):
    ss = pd.to_numeric(s, errors="coerce").fillna(0).astype("float64").map(lambda x: f"{x:.6f}")
    parts = ss.str.split(".", expand=True)
    non = parts[0].str.lstrip("-").str.rjust(keep_non, "0")
    dec = parts[1].str.ljust(keep_dec, "0") if parts.shape[1] > 1 else pd.Series(["0" * keep_dec] * len(ss))
    out = {}
    for i in range(keep_non):
        out[f"digit{i}"] = non.str[i].astype("int8")
    for i in range(keep_dec):
        out[f"decimal_digit{i}"] = dec.str[i].astype("int8")
    return out


def make_digit_interactions(df: pd.DataFrame, col_a: str, col_b: str):
    return ((df[col_a].astype("int16") + 1) * 11 + (df[col_b].astype("int16") + 1)).astype("int16")


def build_orig_prob_maps(orig_keys: pd.DataFrame, y_orig: np.ndarray, cols):
    temp = orig_keys.copy()
    for col in cols:
        temp[col] = temp[col].astype("string").fillna("__MISSING__")
    temp["_target"] = y_orig
    priors = [(y_orig == k).mean() for k in range(3)]
    maps = {}
    for col in cols:
        maps[col] = {}
        count_map = temp.groupby(col, dropna=False)["_target"].size().astype(np.float32)
        maps[col]["count"] = count_map
        for k in range(3):
            prob_map = temp.groupby(col, dropna=False)["_target"].apply(lambda s, kk=k: (s == kk).mean())
            maps[col][f"p{k}"] = prob_map.astype(np.float32)
        maps[col]["prior"] = priors
    return maps


def apply_orig_prob_maps(df: pd.DataFrame, maps, cols):
    out = {}
    for col in cols:
        key = df[col].astype("string").fillna("__MISSING__")
        out[f"orig_count__{col}"] = key.map(maps[col]["count"]).fillna(0).astype(np.float32)
        priors = maps[col]["prior"]
        for k in range(3):
            out[f"orig_p{k}__{col}"] = key.map(maps[col][f"p{k}"]).fillna(priors[k]).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def fit_full_multiclass_maps(s: pd.Series, y: np.ndarray, smooth: float):
    frame = pd.DataFrame({"key": s.astype("string").fillna("__MISSING__"), "y": y})
    counts = frame.groupby("key", dropna=False)["y"].size().astype(np.float32)
    priors = np.array([(y == k).mean() for k in range(3)], dtype=np.float32)
    out = {"counts": counts, "priors": priors, "maps": {}}
    for k in range(3):
        means = frame.groupby("key", dropna=False)["y"].apply(lambda z, kk=k: (z == kk).mean()).astype(np.float32)
        if smooth > 0:
            means = ((means * counts) + (priors[k] * smooth)) / (counts + smooth)
        out["maps"][k] = means.astype(np.float32)
    return out


def apply_multiclass_maps(s: pd.Series, enc):
    key = s.astype("string").fillna("__MISSING__")
    out = np.zeros((len(s), 3), dtype=np.float32)
    for k in range(3):
        out[:, k] = key.map(enc["maps"][k]).fillna(enc["priors"][k]).astype(np.float32).values
    return out


def target_encode_block(X_tr, y_tr, X_va, X_te, cols, drop_original):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for col in cols:
        oof = np.zeros((len(X_tr), 3), dtype=np.float32)
        tr_series = X_tr[col].astype("string").fillna("__MISSING__")
        for inner_tr_idx, inner_va_idx in inner.split(X_tr, y_tr):
            enc = fit_full_multiclass_maps(tr_series.iloc[inner_tr_idx], y_tr[inner_tr_idx], TE_SMOOTH)
            oof[inner_va_idx] = apply_multiclass_maps(tr_series.iloc[inner_va_idx], enc)
        enc_full = fit_full_multiclass_maps(tr_series, y_tr, TE_SMOOTH)
        va_vals = apply_multiclass_maps(X_va[col], enc_full)
        te_vals = apply_multiclass_maps(X_te[col], enc_full)
        for k in range(3):
            name = f"TE_{col}_c{k}"
            X_tr[name] = oof[:, k]
            X_va[name] = va_vals[:, k]
            X_te[name] = te_vals[:, k]
        if drop_original:
            X_tr = X_tr.drop(columns=[col])
            X_va = X_va.drop(columns=[col])
            X_te = X_te.drop(columns=[col])
    return X_tr, X_va, X_te


def build_feature_frames(train_df, test_df, orig_df):
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()
    all_dfs = [train, test, orig]

    round_cols, bin_cols = [], []
    digit_cols, digit_inter_cols, digit_cat_cols = [], [], []
    catpair_cols = []

    for col, levels in ROUND_SPECS.items():
        for level in levels:
            suffix = str(level).replace("-", "m")
            name = f"{col}_ROUND_{suffix}"
            round_cols.append(name)
            for df in all_dfs:
                df[name] = pd.to_numeric(df[col], errors="coerce").fillna(-1).round(level).astype("int32")

    for col, bin_sizes in BIN_SPECS.items():
        for size in bin_sizes:
            name = f"{col}_BIN_FLOOR_{size}"
            bin_cols.append(name)
            for df in all_dfs:
                df[name] = (pd.to_numeric(df[col], errors="coerce").fillna(-1) // size).astype("int32")

    for base_col in DIGIT_BASE_COLS:
        for df in all_dfs:
            digs = make_digits_from_float(df[base_col], keep_non=3, keep_dec=3)
            for k, v in digs.items():
                df[f"{base_col}_{k}"] = v
        for k in ["digit0", "digit1", "digit2", "decimal_digit0", "decimal_digit1", "decimal_digit2"]:
            digit_cols.append(f"{base_col}_{k}")

    for base_col in DIGIT_BASE_COLS:
        pairs = [
            (f"{base_col}_decimal_digit0", f"{base_col}_decimal_digit1"),
            (f"{base_col}_decimal_digit1", f"{base_col}_decimal_digit2"),
            (f"{base_col}_digit1", f"{base_col}_decimal_digit0"),
        ]
        for a, b in pairs:
            if a not in train.columns or b not in train.columns:
                continue
            name = f"{base_col}_DIGITINTER_{a.split('_')[-1]}_{b.split('_')[-1]}"
            for df in all_dfs:
                df[name] = make_digit_interactions(df, a, b)
            digit_inter_cols.append(name)

    cross_digit_specs = [
        ("Rainfall_mm_decimal_digit0", "Previous_Irrigation_mm_decimal_digit0", "RF_dec0_X_PI_dec0"),
        ("Soil_Moisture_decimal_digit0", "Temperature_C_decimal_digit0", "SM_dec0_X_TC_dec0"),
    ]
    for a, b, name in cross_digit_specs:
        if a in train.columns and b in train.columns:
            for df in all_dfs:
                df[name] = make_digit_interactions(df, a, b)
            digit_cat_cols.append(name)

    cat_pairs = list(itertools.combinations(CAT_COLS, 2))[:MAX_CATPAIR]
    for a, b in cat_pairs:
        name = f"{a}__X__{b}"
        catpair_cols.append(name)
        for df in all_dfs:
            df[name] = df[a].astype("string").fillna("NA") + "|" + df[b].astype("string").fillna("NA")

    orig_keys = CAT_COLS + round_cols + bin_cols + digit_inter_cols + digit_cat_cols
    y_orig = orig[TARGET].map(TARGET_MAP).astype(np.int32).values
    orig_maps = build_orig_prob_maps(orig[orig_keys], y_orig, orig_keys)
    train_orig_feats = apply_orig_prob_maps(train[orig_keys], orig_maps, orig_keys)
    test_orig_feats = apply_orig_prob_maps(test[orig_keys], orig_maps, orig_keys)

    train = pd.concat([train, train_orig_feats], axis=1)
    test = pd.concat([test, test_orig_feats], axis=1)
    orig_feat_cols = list(train_orig_feats.columns)

    feature_cols = NUM_COLS + CAT_COLS + orig_feat_cols + round_cols + digit_cols + digit_inter_cols + digit_cat_cols + bin_cols + catpair_cols
    feature_cols = list(dict.fromkeys(feature_cols))

    base_te_cols = [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C", "Humidity"] if c in feature_cols]
    base_te_cols += digit_cols + round_cols + bin_cols + orig_feat_cols
    base_te_cols = [c for c in dict.fromkeys(base_te_cols) if c in feature_cols]
    te_feat_cols = [c for c in catpair_cols + digit_inter_cols + digit_cat_cols if c in feature_cols]

    train_X = train[feature_cols].copy()
    test_X = test[feature_cols].copy()
    category_candidates = [c for c in CAT_COLS if c in train_X.columns]
    return train_X, test_X, base_te_cols, te_feat_cols, category_candidates


def finalize_for_lgbm(X_tr, X_va, X_te, category_candidates):
    all_cols = list(dict.fromkeys(list(X_tr.columns) + list(X_va.columns) + list(X_te.columns)))
    X_tr = X_tr.reindex(columns=all_cols)
    X_va = X_va.reindex(columns=all_cols)
    X_te = X_te.reindex(columns=all_cols)
    for col in all_cols:
        if col in category_candidates:
            all_vals = pd.concat(
                [X_tr[col].astype("string"), X_va[col].astype("string"), X_te[col].astype("string")],
                ignore_index=True,
            ).fillna("__MISSING__")
            cat = pd.Categorical(all_vals)
            n_tr = len(X_tr)
            n_va = len(X_va)
            X_tr[col] = pd.Categorical(all_vals[:n_tr], categories=cat.categories)
            X_va[col] = pd.Categorical(all_vals[n_tr:n_tr + n_va], categories=cat.categories)
            X_te[col] = pd.Categorical(all_vals[n_tr + n_va:], categories=cat.categories)
        else:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(0).astype(np.float32)
            X_va[col] = pd.to_numeric(X_va[col], errors="coerce").fillna(0).astype(np.float32)
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(0).astype(np.float32)
    return X_tr, X_va, X_te, [c for c in category_candidates if c in X_tr.columns]


def main():
    train = read_train()
    test = read_test()
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(np.int32).values
    train_base = train.drop(columns=[TARGET, ID]).copy()
    test_base = test.drop(columns=[ID]).copy()
    orig_base = orig.drop(columns=[TARGET]).copy()

    train_base, test_base, orig_base = label_encode_across_all(train_base, test_base, orig_base, CAT_COLS)
    orig_with_target = orig_base.copy()
    orig_with_target[TARGET] = orig[TARGET].values

    log("Building feature blocks")
    X_all, T_all, base_te_cols, te_feat_cols, category_candidates = build_feature_frames(train_base, test_base, orig_with_target)
    log(f"Base feature matrix: {X_all.shape}")
    log(f"TE1 cols: {len(te_feat_cols)} | TE2 cols: {len(base_te_cols)}")

    fold_ids = folds(y)
    oof = np.zeros((len(X_all), 3), dtype=np.float32)
    test_preds = np.zeros((len(T_all), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        log(f"fold {k}")

        X_tr = reset(X_all.iloc[tr_idx].copy())
        X_va = reset(X_all.iloc[va_idx].copy())
        X_te = reset(T_all.copy())
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        if te_feat_cols:
            X_tr, X_va, X_te = target_encode_block(X_tr, y_tr, X_va, X_te, te_feat_cols, drop_original=True)
        if base_te_cols:
            te2_cols = [c for c in base_te_cols if c in X_tr.columns]
            X_tr, X_va, X_te = target_encode_block(X_tr, y_tr, X_va, X_te, te2_cols, drop_original=False)

        X_tr, X_va, X_te, cat_feats = finalize_for_lgbm(X_tr, X_va, X_te, category_candidates)

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            categorical_feature=cat_feats,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(LOG_EVERY),
            ],
        )

        va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)
        te_pred = model.predict_proba(X_te, num_iteration=model.best_iteration_)
        oof[va_idx] = va_pred.astype(np.float32)
        test_preds += te_pred.astype(np.float32) / N_FOLDS
        log(f"fold {k} BAC = {balanced_accuracy_score(y_va, va_pred.argmax(axis=1)):.6f}")

        del X_tr, X_va, X_te, model, va_pred, te_pred
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, test_preds)


if __name__ == "__main__":
    main()
