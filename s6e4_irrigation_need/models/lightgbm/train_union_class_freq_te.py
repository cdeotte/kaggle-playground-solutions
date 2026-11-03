#!/usr/bin/env python
"""Class-weighted LightGBM on a union of token, count, prior, digit, drift, distribution and n-gram blocks with
per-fold class-frequency encodings (pair lgbm_union_dist_ngrams_class_freq_te_a).

Label-free features (train + test rows; original rows for priors, counts, ranks and drift; 447 columns):
  - the 19 columns (categoricals as pandas categories, native in LightGBM)
  - every column with > 7 distinct values (the 11 numerics): its value as a string key, its count over train + test,
    its add-one smoothed class probabilities in the original rows, count / count over original + train + test
  - rainfall, previous irrigation, soil moisture, humidity, temperature as string keys: round(x) // 2, int(x),
    <= 500 quantile bins
  - up to 4 integer and 4 decimal digits of every numeric; the 6 pairs of the 4 previous-irrigation decimal digits
    as keys
  - frequency (train + test + original) of rainfall, previous irrigation, soil moisture; percentile rank of 5
    numerics; log1p / sqrt of 3 numerics; rainfall + previous irrigation - 2 x moisture (signed, absolute),
    rainfall / (previous irrigation + 1), (rainfall + previous irrigation) / (area + 1); <= 200 equal-width bins of
    3 numerics as keys; add-one smoothed original class probabilities of the 8 categoricals
  - for 6 numerics (rainfall, previous irrigation, soil moisture, humidity, temperature, sunlight): 5-quantile bins
    (edges from the train rows) and value codes (factorized over train + test + original); roundings to 10 / 1 and
    digits at 100 / 10 / 1 / 0.1 of rainfall, previous irrigation, soil moisture; with the codes of the other 13
    columns: log1p(train frequency / original frequency) and original count and add-one smoothed original class
    probabilities (of the codes and digits); train means of the 6 numerics per code of Soil_Type, Soil_pH,
    Organic_Carbon, Electrical_Conductivity
  - previous irrigation against the original rows: percentile rank within the Low / Medium / High rows and overall,
    |z High| - |z Low|, distances to the q25 / q50 / q75 of the Low and High rows
  - the 3 numerics as string keys, the 15 pairs of Crop_Type, Season, Irrigation_Type, Water_Source, Region,
    Soil_Type and the 4 triples of the first 4, as keys
Per fold: each of the 57 string keys is replaced by its add-one smoothed class frequencies ((count_k + 1) /
(count + 3)), fitted on the fold's training rows and applied to the training, validation and test rows.
Model: LightGBM multiclass, balanced class sample weights, lr 0.01, max_depth 4, subsample 0.5, colsample_bytree 0.2,
lambda_l2 15, lambda_l1 10, force_col_wise, seed 42, 32 threads, up to 20,000 rounds with early stopping (500) on the
validation fold. Test = mean of the 5 fold models.
Usage: python train_union_class_freq_te.py"""
import gc
import itertools
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

PAIR = "lgbm_union_dist_ngrams_class_freq_te_a"
SEED = 42
N_FOLDS = 5
HIGH_CARD_THRESHOLD = 7
Q_LIST = [5]
DRIFT_EPS = 1e-6
CLASS_NAMES = ["Low", "Medium", "High"]
CLASS_IDS = list(range(len(CLASS_NAMES)))

TOP_CATS_FOR_NGRAM = [
    "Crop_Type", "Season", "Irrigation_Type",
    "Water_Source", "Region", "Soil_Type",
]
A_CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
A_NUMS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
NUMS_CAND = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C", "Sunlight_Hours"]
ROUND_SPECS = {
    "Rainfall_mm": [-1, 0],
    "Previous_Irrigation_mm": [-1, 0],
    "Soil_Moisture": [-1, 0],
}
DIGIT_SPECS = {
    "Rainfall_mm": list(range(-2, 2)),
    "Previous_Irrigation_mm": list(range(-2, 2)),
    "Soil_Moisture": list(range(-2, 2)),
}

LGB_PARAMS = dict(
    objective="multiclass",
    num_class=len(CLASS_IDS),
    metric="multi_logloss",
    n_estimators=20000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.2,
    lambda_l2=15.0,
    lambda_l1=10.0,
    random_state=SEED,
    verbosity=-1,
    num_threads=32,
    force_col_wise=True,
    device_type="cpu",
    n_jobs=32,
)

T0_GLOBAL = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0_GLOBAL:8.1f}s] {msg}", flush=True)


def sanitize_series(s: pd.Series, fill: str = "missing") -> pd.Series:
    return s.astype("string").fillna(fill).str.strip()


def _reset(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index(drop=True)


def create_categorical_combs(X: pd.DataFrame, list_of_col_tuples, delimiter="|") -> pd.DataFrame:
    combs = {}
    for cols in list_of_col_tuples:
        name = delimiter.join(cols)
        combs[name] = X[list(cols)].astype(str).agg(delimiter.join, axis=1)
    return pd.DataFrame(combs)


def create_count_features(X: pd.DataFrame, list_of_inputs) -> pd.DataFrame:
    list_of_inputs = [[col] if isinstance(col, str) else col for col in list_of_inputs]
    new_features = {}
    for cols in list_of_inputs:
        name = f"count({','.join(cols)})"
        new_features[name] = X.groupby(list(cols), dropna=False).transform("size")
    return pd.DataFrame(new_features)


def create_groupby_features_original_multiclass(X: pd.DataFrame, cat_cols, original_data: pd.DataFrame) -> pd.DataFrame:
    new_features = {}
    orig = original_data.copy()
    for col in cat_cols:
        grp = orig.groupby(col, dropna=False)[TARGET].value_counts().unstack(fill_value=0).reindex(columns=CLASS_IDS, fill_value=0)
        probs = (grp + 1.0).div(grp.sum(axis=1) + len(CLASS_IDS), axis=0)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            name = f"groupby({col})_proba_{cls_name}_original"
            global_p = float((orig[TARGET] == cls_idx).mean())
            new_features[name] = X[col].map(probs[cls_idx]).fillna(global_p).astype("float32")
    return pd.DataFrame(new_features)


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


def add_qcut_bins_3way(train, test, orig, num_cols, q_list):
    new_cols = []
    for c in num_cols:
        if c not in train.columns:
            continue
        for q in q_list:
            name = f"B_qcut_{c}_q{q}"
            try:
                b_tr, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = b_tr.astype("float32")
                test[name] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).astype("float32")
                orig[name] = pd.cut(orig[c], bins=bins, labels=False, include_lowest=True).astype("float32")
            except Exception:
                train[name] = 0.0
                test[name] = 0.0
                orig[name] = 0.0
            new_cols.append(name)
    return train, test, orig, new_cols


def add_round_features_3way(train, test, orig, round_specs):
    out = []
    for c, rr in round_specs.items():
        if c not in train.columns:
            continue
        for r in rr:
            n = f"B_round_{c}_r{r}"
            train[n] = pd.to_numeric(train[c], errors="coerce").round(r)
            test[n] = pd.to_numeric(test[c], errors="coerce").round(r)
            orig[n] = pd.to_numeric(orig[c], errors="coerce").round(r)
            out.append(n)
    return train, test, orig, out


def add_digit_features_3way(train, test, orig, digit_specs):
    out = []
    for c, ks in digit_specs.items():
        if c not in train.columns:
            continue
        for k in ks:
            n = f"B_digit_{c}_d{k}"
            for df in (train, test, orig):
                v = pd.to_numeric(df[c], errors="coerce")
                df[n] = (((v * (10 ** k)) % 10).fillna(-1)).astype("int8")
            out.append(n)
    return train, test, orig, out


def factorize_across_all_3way(train, test, orig, cols):
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[c] = codes[:len(train)]
        test[c] = codes[len(train):len(train) + len(test)]
        orig[c] = codes[len(train) + len(test):]
    return train, test, orig


def add_drift_ratio(train, test, orig, cols, eps=1e-6):
    new_cols = []
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        p_tr = train[c].value_counts(normalize=True)
        p_or = orig[c].value_counts(normalize=True)
        name = f"B_DRIFT_{c}"
        train[name] = np.log1p(train[c].map(p_tr).fillna(0) / (train[c].map(p_or).fillna(eps) + eps)).astype("float32")
        test[name] = np.log1p(test[c].map(p_tr).fillna(0) / (test[c].map(p_or).fillna(eps) + eps)).astype("float32")
        new_cols.append(name)
    return train, test, new_cols


def add_orig_te_ce_multiclass(train, test, orig, cols, y_orig):
    te_cols, ce_cols = [], []
    tmp = orig.copy()
    tmp["_y_"] = y_orig.astype("int32")
    priors = np.bincount(y_orig, minlength=len(CLASS_IDS)) / len(y_orig)
    for c in cols:
        if c not in tmp.columns:
            continue
        vc = tmp[c].value_counts(dropna=False)
        train[f"B_CEORIG_{c}"] = train[c].map(vc).fillna(0).astype("float32")
        test[f"B_CEORIG_{c}"] = test[c].map(vc).fillna(0).astype("float32")
        ce_cols.append(f"B_CEORIG_{c}")
        grp = tmp.groupby(c, dropna=False)["_y_"].value_counts().unstack(fill_value=0).reindex(columns=CLASS_IDS, fill_value=0)
        probs = (grp + 1.0).div(grp.sum(axis=1) + len(CLASS_IDS), axis=0)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            te_name = f"B_TEORIG_{c}_{cls_name}"
            train[te_name] = train[c].map(probs[cls_idx]).fillna(priors[cls_idx]).astype("float32")
            test[te_name] = test[c].map(probs[cls_idx]).fillna(priors[cls_idx]).astype("float32")
            te_cols.append(te_name)
    return train, test, te_cols, ce_cols


def add_group_means(train, test, group_cols, num_cols):
    new_cols = []
    for g in group_cols:
        if g not in train.columns:
            continue
        for n in num_cols:
            if n not in train.columns or g == n:
                continue
            name = f"B_MEAN_{n}_BY_{g}"
            m = train.groupby(g, dropna=False)[n].mean()
            train[name] = train[g].map(m).astype("float32")
            test[name] = test[g].map(m).astype("float32")
            fillv = float(np.nanmean(train[name].values))
            train[name] = train[name].fillna(fillv).astype("float32")
            test[name] = test[name].fillna(fillv).astype("float32")
            new_cols.append(name)
    return train, test, new_cols


def fit_multiclass_target_encoder(X, y, cols):
    encoders = {}
    priors = np.bincount(y, minlength=len(CLASS_IDS)) / len(y)
    for c in cols:
        x = X[c].astype(str)
        grp = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].value_counts().unstack(fill_value=0).reindex(columns=CLASS_IDS, fill_value=0)
        probs = (grp + 1.0).div(grp.sum(axis=1) + len(CLASS_IDS), axis=0)
        encoders[c] = (probs, priors)
    return encoders


def transform_multiclass_target_encoder(X, cols, encoders, prefix="TE"):
    out = pd.DataFrame(index=X.index)
    for c in cols:
        probs, priors = encoders[c]
        x = X[c].astype(str)
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            mapped = x.map(probs[cls_idx])
            out[f"{prefix}_{c}_{cls_name}"] = mapped.fillna(priors[cls_idx]).astype("float32")
    return out


def build_features():
    log("Loading data")
    train = read_train()
    test = read_test()
    orig = read_original()
    for df in (train, test, orig):
        if ID in df.columns:
            df.drop(columns=[ID], inplace=True)
    for df in (train, test, orig):
        for c in A_NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in A_CATS:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
    train[TARGET] = train[TARGET].map(TARGET_MAP).astype("int8")
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype("int8")
    y = train[TARGET].values
    X = train.drop(columns=[TARGET]).copy()

    log("Features: value keys, counts, original priors, digits")
    combined = pd.concat([_reset(X), _reset(test)], axis=0, ignore_index=True)
    N_combined = len(combined)
    N_train = len(X)
    N_test = len(test)
    high_cardinality = [col for col in combined.columns if combined[col].nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    te1 = _reset(combined[high_cardinality].astype(str).add_suffix("_te"))
    ce1 = _reset(create_count_features(combined, high_cardinality))
    te1_original = _reset(create_groupby_features_original_multiclass(combined, high_cardinality, orig))
    ce1_original = _reset(create_count_features(pd.concat([orig[high_cardinality], combined[high_cardinality]], axis=0, ignore_index=True), high_cardinality).iloc[len(orig):].reset_index(drop=True))
    count_ratios = {}
    for j, colname in enumerate(ce1.columns):
        denom = ce1_original.iloc[:, j].astype("float32")
        numer = ce1.iloc[:, j].astype("float32")
        count_ratios[f"{colname}_ratio"] = (numer / denom.replace(0, np.nan)).fillna(0.0)
    count_ratios = _reset(pd.DataFrame(count_ratios))

    num_bins = {}
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"] if c in combined.columns]:
        s = combined[col]
        num_bins[f"{col}_round_div2_te"] = (pd.to_numeric(s, errors="coerce").round().fillna(-1) // 2).astype("int64").astype(str)
        num_bins[f"{col}_int_te"] = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").astype(str)
        try:
            q = int(min(500, max(2, s.nunique(dropna=True))))
            num_bins[f"{col}_quantiled_te"] = pd.qcut(pd.to_numeric(s, errors="coerce"), q=q, duplicates="drop").astype(str)
        except Exception:
            num_bins[f"{col}_quantiled_te"] = pd.Series(["NA"] * N_combined)
    num_bins = _reset(pd.DataFrame(num_bins))

    digits = {}
    float_cols = [c for c in combined.columns if pd.api.types.is_float_dtype(combined[c])]
    for col in float_cols:
        s = combined[col]
        ss = pd.to_numeric(s, errors="coerce").fillna(0).astype("float64").map(lambda x: f"{x:.6f}")
        splitted = ss.str.split(".", expand=True)
        non_decimal_part = splitted[0]
        decimal_part = splitted[1] if splitted.shape[1] > 1 else pd.Series(["0"] * len(ss))
        max_len_non_decimal = int(non_decimal_part.str.len().max())
        max_len_decimal = int(decimal_part.str.len().max())
        non_decimal_part_padded = non_decimal_part.str.rjust(max_len_non_decimal, "0")
        decimal_part_padded = decimal_part.str.ljust(max_len_decimal, "0")
        keep_non = min(max_len_non_decimal, 4)
        keep_dec = min(max_len_decimal, 4)
        for i in range(keep_non):
            digits[f"{col}_digit{i}"] = non_decimal_part_padded.str[i].astype(int)
        for i in range(keep_dec):
            digits[f"{col}_decimal_digit{i}"] = decimal_part_padded.str[i].astype(int)
    digits = _reset(pd.DataFrame(digits))
    digits_combs = pd.DataFrame(index=pd.RangeIndex(N_combined))
    candidate = [c for c in digits.columns if c.startswith("Previous_Irrigation_mm_")] or [c for c in digits.columns if c.startswith("Rainfall_mm_")]
    candidate = [c for c in candidate if "decimal_digit" in c][:6]
    if len(candidate) >= 2:
        dig_pairs = list(itertools.combinations(candidate, 2))
        digits_combs = create_categorical_combs(digits[candidate].astype(str), dig_pairs).add_suffix("_te")
    digits_combs = _reset(digits_combs)

    log("Features: frequencies, ranks, transforms, bins")
    trA = _reset(train.drop(columns=[TARGET]))
    teA = _reset(test.copy())
    orA = orig.copy()
    A_blocks = {}
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"] if c in trA.columns]:
        freq = pd.concat([trA[col], teA[col], orA[col]], axis=0).value_counts(normalize=True)
        tt = pd.concat([trA[col], teA[col]], axis=0, ignore_index=True)
        A_blocks[f"A_FREQ_{col}"] = tt.map(freq).fillna(0).astype("float32")
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"] if c in trA.columns]:
        r = pd.concat([trA[col], teA[col], orA[col]], axis=0, ignore_index=True).rank(method="average", pct=True)
        A_blocks[f"A_RANK_{col}"] = r.iloc[:N_train + N_test].astype("float32").values
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"] if c in trA.columns]:
        v = pd.to_numeric(pd.concat([trA[col], teA[col]], axis=0, ignore_index=True), errors="coerce").fillna(0).astype("float32")
        A_blocks[f"A_LOG1P_{col}"] = np.log1p(v.clip(lower=0)).astype("float32")
        A_blocks[f"A_SQRT_{col}"] = np.sqrt(v.clip(lower=0)).astype("float32")
    if all(c in trA.columns for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Field_Area_hectare"]):
        rain = pd.concat([trA["Rainfall_mm"], teA["Rainfall_mm"]], axis=0, ignore_index=True).astype("float32")
        prev = pd.concat([trA["Previous_Irrigation_mm"], teA["Previous_Irrigation_mm"]], axis=0, ignore_index=True).astype("float32")
        moist = pd.concat([trA["Soil_Moisture"], teA["Soil_Moisture"]], axis=0, ignore_index=True).astype("float32")
        area = pd.concat([trA["Field_Area_hectare"], teA["Field_Area_hectare"]], axis=0, ignore_index=True).astype("float32")
        dev = (rain + prev - 2 * moist).astype("float32")
        A_blocks["A_water_deviation"] = dev
        A_blocks["A_abs_water_dev"] = np.abs(dev).astype("float32")
        A_blocks["A_rain_to_prev_ratio"] = (rain / (prev + 1)).astype("float32")
        A_blocks["A_avg_water_per_area"] = ((rain + prev) / (area + 1)).astype("float32")
    A_num_df = _reset(pd.DataFrame(A_blocks))

    A_te_extra = {}
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"] if c in combined.columns]:
        s = pd.to_numeric(combined[col], errors="coerce")
        A_te_extra[f"A_{col}_bin_te"] = pd.cut(s, bins=min(200, max(10, s.nunique(dropna=True))), duplicates="drop").astype(str)
    A_te_df = _reset(pd.DataFrame(A_te_extra))
    A_orig_df = _reset(create_groupby_features_original_multiclass(combined, [c for c in A_CATS if c in combined.columns], orig))

    log("Features: bins, roundings, digits, drift, original priors, group means")
    trB = train.copy()
    teB = test.copy()
    orB = orig.copy()
    y_orig = orB[TARGET].values.astype(np.int32)
    B_NUMS = [c for c in NUMS_CAND if c in trB.columns and c in teB.columns and c in orB.columns]
    B_CATS = [c for c in trB.columns if c not in [TARGET] + B_NUMS]
    for df in (trB, teB, orB):
        for c in B_CATS:
            df[c] = sanitize_series(df[c])
    trB, teB, orB, B_QCUT = add_qcut_bins_3way(trB, teB, orB, B_NUMS, Q_LIST)
    trB, teB, orB, B_ROUND = add_round_features_3way(trB, teB, orB, ROUND_SPECS)
    trB, teB, orB, B_DIGITS = add_digit_features_3way(trB, teB, orB, DIGIT_SPECS)
    B_NUM_CAT = []
    for c in B_NUMS:
        n = f"B_NUMCAT_{c}"
        for df in (trB, teB, orB):
            df[n] = pd.to_numeric(df[c], errors="coerce")
        B_NUM_CAT.append(n)
    B_FACT_COLS = B_CATS + B_NUM_CAT
    trB, teB, orB = factorize_across_all_3way(trB, teB, orB, B_FACT_COLS)
    trB, teB, B_DRIFT = add_drift_ratio(trB, teB, orB, cols=(B_CATS + B_NUM_CAT), eps=DRIFT_EPS)
    trB, teB, B_TEORIG, B_CEORIG = add_orig_te_ce_multiclass(trB, teB, orB, B_CATS + B_NUM_CAT + B_DIGITS, y_orig)
    trB, teB, B_GM = add_group_means(trB, teB, group_cols=B_CATS[:4], num_cols=B_NUMS)
    B_attach_cols = B_QCUT + B_ROUND + B_DIGITS + B_NUM_CAT + B_DRIFT + B_TEORIG + B_CEORIG + B_GM
    B_attach_df = _reset(pd.concat([trB[B_attach_cols].reset_index(drop=True), teB[B_attach_cols].reset_index(drop=True)], axis=0, ignore_index=True))

    log("Features: original class distributions, n-grams")
    orig_dist_all = pd.to_numeric(orig["Previous_Irrigation_mm"], errors="coerce").fillna(orig["Previous_Irrigation_mm"].median()).values
    orig_low = pd.to_numeric(orig.loc[orig[TARGET] == 0, "Previous_Irrigation_mm"], errors="coerce").dropna().values
    orig_med = pd.to_numeric(orig.loc[orig[TARGET] == 1, "Previous_Irrigation_mm"], errors="coerce").dropna().values
    orig_high = pd.to_numeric(orig.loc[orig[TARGET] == 2, "Previous_Irrigation_mm"], errors="coerce").dropna().values
    dist_base = pd.to_numeric(combined["Previous_Irrigation_mm"], errors="coerce").fillna(0).values
    dist = {
        "X_pctrank_low_dist": pctrank_against(dist_base, orig_low),
        "X_pctrank_med_dist": pctrank_against(dist_base, orig_med),
        "X_pctrank_high_dist": pctrank_against(dist_base, orig_high),
        "X_pctrank_orig_dist": pctrank_against(dist_base, orig_dist_all),
        "X_zscore_high_gap": (np.abs(zscore_against(dist_base, orig_high)) - np.abs(zscore_against(dist_base, orig_low))).astype("float32"),
    }
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        low_q = np.quantile(orig_low, q_val)
        high_q = np.quantile(orig_high, q_val)
        dist[f"X_dist_To_low_{q_label}"] = np.abs(dist_base - low_q).astype("float32")
        dist[f"X_dist_To_high_{q_label}"] = np.abs(dist_base - high_q).astype("float32")
    X_dist_df = _reset(pd.DataFrame(dist))

    ngram_te = {}
    for col in [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"] if c in combined.columns]:
        ngram_te[f"X_CAT_{col}_te"] = pd.to_numeric(combined[col], errors="coerce").fillna(-1).astype(str)
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        ngram_te[f"X_BG_{c1}_{c2}_te"] = (sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2])).astype(str)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        ngram_te[f"X_TG_{c1}_{c2}_{c3}_te"] = (sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2]) + "_" + sanitize_series(combined[c3])).astype(str)
    X_ngram_te_df = _reset(pd.DataFrame(ngram_te))

    dfs = [combined, te1, ce1, te1_original, num_bins, digits, digits_combs, count_ratios, A_num_df, A_te_df, A_orig_df, B_attach_df, X_dist_df, X_ngram_te_df]
    combined_fe = pd.concat(dfs, axis=1)
    combined_fe = combined_fe.loc[:, ~combined_fe.columns.duplicated()].copy()
    cat_cols = combined_fe.select_dtypes("object").columns.tolist() + combined_fe.select_dtypes(include=["string"]).columns.tolist()
    cat_cols = sorted(set(cat_cols))
    if len(cat_cols) > 0:
        combined_fe[cat_cols] = combined_fe[cat_cols].astype("category")
    X_fe = combined_fe.iloc[:N_train].copy()
    T_fe = combined_fe.iloc[N_train:].copy()
    log(f"X_fe: {X_fe.shape} | T_fe: {T_fe.shape}")
    return X_fe, T_fe, y


def main():
    X_fe, T_fe, y = build_features()

    fold_ids = folds(y)
    oof = np.zeros((len(X_fe), len(CLASS_IDS)), dtype=np.float64)
    pred_test_folds = np.zeros((len(T_fe), N_FOLDS, len(CLASS_IDS)), dtype=np.float64)
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        t_fold = time.perf_counter()
        log(f"---- Fold {k} ----")
        X_tr = X_fe.iloc[tr_idx].copy()
        y_tr = y[tr_idx].copy()
        X_va = X_fe.iloc[va_idx].copy()
        y_va = y[va_idx].copy()
        X_te = T_fe.copy()

        to_target_encode = [c for c in X_tr.columns if c.endswith("_te")]
        log(f"Target-encoding columns: {len(to_target_encode)}")
        if len(to_target_encode) > 0:
            enc = fit_multiclass_target_encoder(X_tr, y_tr, to_target_encode)
            X_tr_te = transform_multiclass_target_encoder(X_tr, to_target_encode, enc, prefix="MTE")
            X_va_te = transform_multiclass_target_encoder(X_va, to_target_encode, enc, prefix="MTE")
            X_te_te = transform_multiclass_target_encoder(X_te, to_target_encode, enc, prefix="MTE")
            X_tr = pd.concat([X_tr.drop(columns=to_target_encode), X_tr_te], axis=1)
            X_va = pd.concat([X_va.drop(columns=to_target_encode), X_va_te], axis=1)
            X_te = pd.concat([X_te.drop(columns=to_target_encode), X_te_te], axis=1)

        class_counts = np.bincount(y_tr, minlength=len(CLASS_IDS)).astype(np.float32)
        class_weights = len(y_tr) / (len(CLASS_IDS) * np.maximum(class_counts, 1.0))
        sample_weight = class_weights[y_tr]

        model = lgbm.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_va, y_va)],
            callbacks=[lgbm.early_stopping(500, verbose=False), lgbm.log_evaluation(1000)],
        )
        p_va = model.predict_proba(X_va)
        oof[va_idx] = p_va
        pred_test_folds[:, k, :] = model.predict_proba(X_te)
        fold_bac = balanced_accuracy_score(y_va, p_va.argmax(axis=1))
        log(f"Fold {k} BAC: {fold_bac:.6f} | elapsed {time.perf_counter() - t_fold:.1f}s | X_tr shape {X_tr.shape}")
        del X_tr, X_va, X_te, model
        gc.collect()

    test_pred = pred_test_folds.mean(axis=1)
    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, oof.argmax(axis=1)):.6f}")
    save_pair(HERE, PAIR, oof.astype(np.float32), test_pred.astype(np.float32))


if __name__ == "__main__":
    main()
