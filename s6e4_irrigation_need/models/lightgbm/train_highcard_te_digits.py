#!/usr/bin/env python
"""LightGBM on value tokens, counts, original priors and digits with nested smoothed target encodings
(pair lgbm_highcard_te_counts_digits_a).

Label-free features (train + test rows; the original rows give the priors):
  - the 11 numerics and the 8 categoricals (pandas categories over train + test + original, native in LightGBM)
  - every column with > 7 distinct values (the 11 numerics): its value as a string key, its count over train + test,
    its original row count and per-class original frequencies, and count / original count
  - rainfall, previous irrigation, soil moisture, temperature, humidity as string keys: round(x) // 2, int(x),
    <= 5000 quantile bins, <= 5000 equal-width bins
  - 3 integer and 3 decimal digits of rainfall, previous irrigation, soil moisture, temperature, and 4 digit-pair keys
  - 6 category-pair keys
Per fold: each of the 41 string keys is replaced by its 3 class frequencies smoothed toward the class prior
((count x freq + 10 x prior) / (count + 10)), out-of-fold over an inner StratifiedKFold(5, seed 42) of the training
rows and from all training rows for the validation / test rows.
Model: LightGBM multiclass, 7500 rounds (no early stopping), lr 0.01, max_depth 4, subsample 0.5,
colsample_bytree 0.2, lambda_l2 15, lambda_l1 10, seed 42, 32 threads. Test = mean of the 5 fold models.
Usage: python train_highcard_te_digits.py"""
import gc
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

PAIR = "lgbm_highcard_te_counts_digits_a"
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
TOKEN_NUM_COLS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C", "Humidity"]
DIGIT_COLS = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Temperature_C"]
HIGH_CARD_THRESHOLD = 7
TE_SMOOTH = 10.0

LGB_PARAMS = dict(
    objective="multiclass",
    metric="multi_logloss",
    num_class=3,
    n_estimators=7500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
)


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


def create_count_features(X: pd.DataFrame, list_of_inputs):
    out = {}
    list_of_inputs = [[col] if isinstance(col, str) else list(col) for col in list_of_inputs]
    for cols in list_of_inputs:
        name = f"count({','.join(cols)})"
        out[name] = X.groupby(cols, dropna=False).transform("size")
    return pd.DataFrame(out, index=X.index)


def create_groupby_original_features(df: pd.DataFrame, orig_keys: pd.DataFrame, y_orig: np.ndarray, cols):
    priors = [(y_orig == k).mean() for k in range(3)]
    temp = orig_keys.copy()
    for col in cols:
        temp[col] = temp[col].astype("string").fillna("__MISSING__")
    temp["_target"] = y_orig
    out = {}
    for col in cols:
        count_map = temp.groupby(col, dropna=False)["_target"].size().astype(np.float32)
        key = df[col].astype("string").fillna("__MISSING__")
        out[f"orig_size__{col}"] = key.map(count_map).fillna(0).astype(np.float32)
        for k in range(3):
            mean_map = temp.groupby(col, dropna=False)["_target"].apply(lambda s, kk=k: (s == kk).mean()).astype(np.float32)
            out[f"orig_p{k}__{col}"] = key.map(mean_map).fillna(priors[k]).astype(np.float32)
    return pd.DataFrame(out, index=df.index)


def make_num_token_blocks(combined: pd.DataFrame):
    out = pd.DataFrame(index=combined.index)
    for c in TOKEN_NUM_COLS:
        s = pd.to_numeric(combined[c], errors="coerce")
        out[f"{c}_round_div2_te"] = (s.round().fillna(-1).astype(np.int64) // 2).astype(str)
        out[f"{c}_int_te"] = s.fillna(-1).astype(np.int64).astype(str)
        try:
            q = min(5000, max(2, int(s.nunique(dropna=True))))
            out[f"{c}_quantiled_te"] = pd.qcut(s, q=q, duplicates="drop").astype(str)
        except Exception:
            out[f"{c}_quantiled_te"] = "NA"
        try:
            bins = min(5000, max(2, int(s.nunique(dropna=True))))
            out[f"{c}_cut_te"] = pd.cut(s, bins=bins, duplicates="drop").astype(str)
        except Exception:
            out[f"{c}_cut_te"] = "NA"
    return out


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


def add_digit_features(combined: pd.DataFrame):
    out = pd.DataFrame(index=combined.index)
    digit_cols = []
    for c in DIGIT_COLS:
        digs = make_digits_from_float(combined[c], keep_non=3, keep_dec=3)
        for k, v in digs.items():
            name = f"{c}_{k}"
            out[name] = v
            digit_cols.append(name)
    return out, digit_cols


def add_digit_combo_features(digit_df: pd.DataFrame):
    out = pd.DataFrame(index=digit_df.index)
    specs = [
        ("Rainfall_mm_decimal_digit0", "Rainfall_mm_decimal_digit1", "RF_dec01_te"),
        ("Previous_Irrigation_mm_decimal_digit0", "Previous_Irrigation_mm_decimal_digit1", "PI_dec01_te"),
        ("Soil_Moisture_decimal_digit0", "Temperature_C_decimal_digit0", "SM_TC_dec0_te"),
        ("Rainfall_mm_decimal_digit0", "Previous_Irrigation_mm_decimal_digit0", "RF_PI_dec0_te"),
    ]
    for a, b, name in specs:
        if a in digit_df.columns and b in digit_df.columns:
            out[name] = (
                digit_df[a].astype(str).fillna("NA")
                + "|"
                + digit_df[b].astype(str).fillna("NA")
            )
    return out


def fit_multiclass_maps(s: pd.Series, y: np.ndarray):
    priors = np.bincount(y, minlength=3) / len(y)
    frame = pd.DataFrame({"key": s.astype(str).fillna("NA"), "y": y})
    counts = frame.groupby("key", dropna=False)["y"].size().astype(np.float32)
    maps = {}
    for k in range(3):
        means = frame.groupby("key", dropna=False)["y"].apply(lambda z, kk=k: (z == kk).mean()).astype(np.float32)
        means = ((means * counts) + (priors[k] * TE_SMOOTH)) / (counts + TE_SMOOTH)
        maps[k] = means
    return priors.astype(np.float32), maps


def nested_multiclass_te(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, X_te: pd.DataFrame, te_cols):
    X_tr_num = X_tr.drop(columns=te_cols).copy()
    X_va_num = X_va.drop(columns=te_cols).copy()
    X_te_num = X_te.drop(columns=te_cols).copy()
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for col in te_cols:
        tr_s = X_tr[col].astype(str).fillna("NA")
        va_s = X_va[col].astype(str).fillna("NA")
        te_s = X_te[col].astype(str).fillna("NA")
        oof_block = np.zeros((len(X_tr), 3), dtype=np.float32)
        for itr, iva in inner.split(np.zeros(len(X_tr)), y_tr):
            fit_s = tr_s.iloc[itr]
            fit_y = y_tr[itr]
            inner_priors, inner_maps = fit_multiclass_maps(fit_s, fit_y)
            apply_s = tr_s.iloc[iva]
            for k in range(3):
                oof_block[iva, k] = apply_s.map(inner_maps[k]).fillna(inner_priors[k]).to_numpy(np.float32)
        full_priors, full_maps = fit_multiclass_maps(tr_s, y_tr)
        for k in range(3):
            X_tr_num[f"{col}_p{k}"] = oof_block[:, k]
            X_va_num[f"{col}_p{k}"] = va_s.map(full_maps[k]).fillna(full_priors[k]).to_numpy(np.float32)
            X_te_num[f"{col}_p{k}"] = te_s.map(full_maps[k]).fillna(full_priors[k]).to_numpy(np.float32)
    return X_tr_num, X_va_num, X_te_num


def finalize_for_lgbm(X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame, cat_cols):
    all_cols = list(dict.fromkeys(list(X_tr.columns) + list(X_va.columns) + list(X_te.columns)))
    X_tr = X_tr.reindex(columns=all_cols)
    X_va = X_va.reindex(columns=all_cols)
    X_te = X_te.reindex(columns=all_cols)
    for c in all_cols:
        if c in cat_cols:
            all_vals = pd.concat(
                [X_tr[c].astype("string"), X_va[c].astype("string"), X_te[c].astype("string")],
                ignore_index=True,
            ).fillna("__MISSING__")
            cat = pd.Categorical(all_vals)
            n_tr = len(X_tr)
            n_va = len(X_va)
            X_tr[c] = pd.Categorical(all_vals[:n_tr], categories=cat.categories)
            X_va[c] = pd.Categorical(all_vals[n_tr:n_tr + n_va], categories=cat.categories)
            X_te[c] = pd.Categorical(all_vals[n_tr + n_va:], categories=cat.categories)
        else:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(0).astype(np.float32)
            X_va[c] = pd.to_numeric(X_va[c], errors="coerce").fillna(0).astype(np.float32)
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").fillna(0).astype(np.float32)
    return X_tr, X_va, X_te


def build_feature_tables(X: pd.DataFrame, T: pd.DataFrame, orig_X: pd.DataFrame, orig_y: np.ndarray):
    combined = pd.concat([reset(X), reset(T)], axis=0, ignore_index=True)
    high_cardinality = [col for col in combined.columns if combined[col].nunique(dropna=False) > HIGH_CARD_THRESHOLD]

    te1 = reset(combined[high_cardinality].astype(str).add_suffix("_te"))
    ce1 = reset(create_count_features(combined, high_cardinality))
    te1_original = reset(create_groupby_original_features(combined[high_cardinality], orig_X[high_cardinality], orig_y, high_cardinality))

    count_ratios = {}
    for c in ce1.columns:
        base = c.replace("count(", "").replace(")", "")
        size_col = f"orig_size__{base}"
        if size_col in te1_original.columns:
            denom = te1_original[size_col].replace(0, np.nan).astype(np.float32)
            count_ratios[f"{c}_ratio"] = (ce1[c].astype(np.float32) / denom).fillna(0.0)
    count_ratios = reset(pd.DataFrame(count_ratios))

    num_tokens = reset(make_num_token_blocks(combined))
    digit_df, digit_cols = add_digit_features(combined)
    digit_df = reset(digit_df)
    digit_combo = reset(add_digit_combo_features(digit_df))

    combo_pairs = [
        ("Crop_Type", "Season"),
        ("Irrigation_Type", "Water_Source"),
        ("Crop_Type", "Irrigation_Type"),
        ("Season", "Crop_Growth_Stage"),
        ("Soil_Type", "Crop_Type"),
        ("Region", "Crop_Type"),
    ]
    combo_df = pd.DataFrame(index=combined.index)
    for a, b in combo_pairs:
        combo_df[f"{a}__{b}_te"] = combined[[a, b]].astype(str).agg("|".join, axis=1)
    combo_df = reset(combo_df)

    combined_fe = pd.concat(
        [reset(combined), te1, ce1, te1_original, count_ratios, num_tokens, digit_df, digit_combo, combo_df],
        axis=1,
    )
    combined_fe = combined_fe.loc[:, ~combined_fe.columns.duplicated()]
    n_train = len(X)
    X_fe = combined_fe.iloc[:n_train].copy()
    T_fe = combined_fe.iloc[n_train:].copy()
    te_cols = [c for c in X_fe.columns if c.endswith("_te")]
    cat_cols = [c for c in CAT_COLS if c in X_fe.columns]
    return X_fe, T_fe, te_cols, cat_cols


def main():
    train = read_train()
    test = read_test()
    orig = read_original()

    y = train[TARGET].map(TARGET_MAP).astype(np.int32).values
    X = train.drop(columns=[TARGET, ID]).copy()
    T = test.drop(columns=[ID]).copy()
    orig_X = orig.drop(columns=[TARGET]).copy()
    orig_y = orig[TARGET].map(TARGET_MAP).astype(np.int32).values

    X, T, orig_X = label_encode_across_all(X, T, orig_X, CAT_COLS)
    X_fe, T_fe, te_cols, cat_cols = build_feature_tables(X, T, orig_X, orig_y)
    log(f"Final shapes: {X_fe.shape} / {T_fe.shape}")
    log(f"Target-encoding columns: {len(te_cols)}")

    fold_ids = folds(y)
    oof = np.zeros((len(X_fe), 3), dtype=np.float32)
    test_preds = np.zeros((len(T_fe), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        log(f"fold {k}")

        X_tr = X_fe.iloc[tr_idx].copy()
        y_tr = y[tr_idx].copy()
        X_va = X_fe.iloc[va_idx].copy()
        y_va = y[va_idx].copy()
        X_te = T_fe.copy()

        if te_cols:
            X_tr, X_va, X_te = nested_multiclass_te(X_tr, y_tr, X_va, X_te, te_cols)
        X_tr, X_va, X_te = finalize_for_lgbm(X_tr, X_va, X_te, cat_cols)

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.log_evaluation(1000)],
            categorical_feature=[c for c in cat_cols if c in X_tr.columns],
        )

        p_va = model.predict_proba(X_va)
        p_te = model.predict_proba(X_te)
        oof[va_idx] = p_va.astype(np.float32)
        test_preds += p_te.astype(np.float32) / N_FOLDS
        log(f"fold {k} BAC = {balanced_accuracy_score(y_va, p_va.argmax(axis=1)):.6f}")

        del X_tr, X_va, X_te, model, p_va, p_te
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, test_preds)


if __name__ == "__main__":
    main()
