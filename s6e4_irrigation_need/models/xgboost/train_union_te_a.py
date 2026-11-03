#!/usr/bin/env python
"""GPU XGBoost on a wide union of label-free feature blocks plus nested 3-class target encodings of every `*_te` key
(pairs xgb_union_snap_te_a, xgb_union_snap_te_cs080_a, xgb_union_snap_te_cs060_a, xgb_union_snap_te_cs040_a).

Label-free features, built once on train + test + original rows (479 columns):
- snap: every train / test value of Rainfall_mm and Previous_Irrigation_mm that does not occur in the original data is
  replaced by the nearest original value (Rainfall_snap, Previous_Irrigation_snap), plus the difference (*_snap_diff);
  decimal-part strings and digits of Rainfall_mm, Previous_Irrigation_mm and Soil_Moisture;
- frequency and percentile rank of the 11 numerics; log1p / sqrt / 1 / (1 + x) of the numerics and snapped values;
  water-balance relations (rain + previous irrigation, their difference, gaps / ratios to soil moisture and field area,
  field area x snapped values); management counts and yes / no / other flags (mulching, irrigation type, water source,
  season);
- 12-quantile bins of rain / irrigation / moisture and the snapped values, 6 category crosses and 1 triple; counts and
  rare flags (<= 50) of categories, decimal parts and bins; log drift ratio (train frequency / original frequency) of the
  categoricals; original-data class frequencies of the categoricals, the bins and the snapped values; rounding (0 and 1
  decimals) and digit-position features;
- keys in the style of a public LightGBM notebook (L_*): every column with > 12 levels as a string key, its count,
  original High-rate and size, count ratio; round / int / quantile / equal-width bins of the rain / irrigation / moisture
  columns; string digits of rain / irrigation and pairs of their decimal digits;
- radix keys: round(100 x snapped value) and Soil_Moisture combined with the code of each categorical; 7 pair keys
  capped at the 20,000 most frequent levels.
Per fold: each of the 59 `*_te` keys gets 3 class target encodings (sklearn TargetEncoder, smooth 'auto', one binary encoder
per class), out-of-fold over an inner StratifiedKFold(5, shuffle, 42) of the training rows and fitted on all training rows
for the validation / test rows; the other string columns are integer-coded (training + validation rows jointly; X_tr is
already numeric when the test frame is coded, so the test frame gets its own codes, as in the run that produced the
stacked predictions). 597 model columns. Balanced class weights.
Model: XGBoost hist on GPU, multi:softprob, lr 0.03, depth 6, subsample 0.8, max_bin 4096, early stopping 500 (mlogloss)
on the validation fold, random_state 42. The presets differ in colsample_bytree / min_child_weight / gamma / alpha /
lambda. Test = mean of the 5 fold models. Competition rows only (the original rows only feed the statistics).
The feature build takes 30-40 minutes on CPU (mostly the snap maps), each fold about 15 minutes on one GPU.
Usage: python train_union_te_a.py <pair> [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
N_THREADS = 32
RADIX = 100_000
PAIR_MAX_UNIQUE = 20_000
CAT_RARE_THRESH = 50
HIGH_CARD_THRESHOLD = 12

LABELS = list(CLASSES)

BASE_CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
LOW_CARD_COLS = BASE_CATS.copy()
BASE_NUMS = [
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

CROSS_PAIRS = [
    ("Crop_Type", "Season"),
    ("Irrigation_Type", "Water_Source"),
    ("Crop_Type", "Irrigation_Type"),
    ("Crop_Growth_Stage", "Season"),
    ("Crop_Type", "Region"),
    ("Soil_Type", "Crop_Type"),
]
TRIPLE = [("Crop_Type", "Season", "Irrigation_Type")]
PAIR_FEATURES = [
    ("Crop_Type", "Season"),
    ("Irrigation_Type", "Water_Source"),
    ("Crop_Type", "Irrigation_Type"),
    ("Rainfall_bin", "Crop_Type"),
    ("Previous_Irrigation_bin", "Irrigation_Type"),
    ("Rainfall_snap_bin", "Season"),
    ("Previous_Irrigation_snap_bin", "Water_Source"),
]

ROUND_SPECS = {
    "Soil_Moisture": [0, 1],
    "Rainfall_mm": [0, 1],
    "Previous_Irrigation_mm": [0, 1],
    "Rainfall_snap": [0, 1],
    "Previous_Irrigation_snap": [0, 1],
}
DIGIT_BASE_COLS = [
    "Soil_Moisture",
    "Rainfall_mm",
    "Previous_Irrigation_mm",
    "Rainfall_snap",
    "Previous_Irrigation_snap",
]

XGB_PARAMS = dict(
    n_estimators=1_000_000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.80,
    colsample_bytree=0.45,
    min_child_weight=6,
    gamma=0.10,
    reg_alpha=0.50,
    reg_lambda=4.00,
    max_bin=4096,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    enable_categorical=False,
    device="cuda",
    random_state=SEED,
    n_jobs=N_THREADS,
    early_stopping_rounds=500,
    verbosity=0,
)

# pair -> overrides of XGB_PARAMS (the three siblings: lighter regularization, different column sampling)
PRESETS = {
    "xgb_union_snap_te_a": {},
    "xgb_union_snap_te_cs080_a": dict(learning_rate=0.03, max_depth=6, subsample=0.80, colsample_bytree=0.80,
                                      min_child_weight=5, gamma=0.05, reg_alpha=0.10, reg_lambda=1.00),
    "xgb_union_snap_te_cs060_a": dict(learning_rate=0.03, max_depth=6, subsample=0.80, colsample_bytree=0.60,
                                      min_child_weight=5, gamma=0.05, reg_alpha=0.10, reg_lambda=1.00),
    "xgb_union_snap_te_cs040_a": dict(learning_rate=0.03, max_depth=6, subsample=0.80, colsample_bytree=0.40,
                                      min_child_weight=5, gamma=0.05, reg_alpha=0.10, reg_lambda=1.00),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def to_target_codes(s: pd.Series) -> np.ndarray:
    return pd.Series(s).map(TARGET_MAP).astype("int8").values


def safe_string_df(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing").str.strip()


def ensure_numeric_columns(df, cols):
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").astype("float32")
            med = np.nanmedian(s.values)
            if np.isnan(med):
                med = 0.0
            df[c] = s.fillna(med).astype("float32")


def add_decimal_parts_like_eda(df):
    mapping = [
        ("Rainfall_mm", "rain"),
        ("Previous_Irrigation_mm", "prev"),
        ("Soil_Moisture", "moist"),
    ]
    for src, prefix in mapping:
        if src in df.columns:
            ss = df[src].astype("string")
            frac = ss.str.split(".").str[1]
            df[prefix] = frac
            df[f"{prefix}0"] = frac.str[0]
            df[f"{prefix}1"] = "-1"
            df[f"{prefix}_len"] = frac.str.len().fillna(0).astype("int16")
            idx = df[f"{prefix}_len"] >= 2
            df.loc[idx, f"{prefix}1"] = df.loc[idx, prefix].str[1]


def _build_snap_map(combo_s: pd.Series, orig_s: pd.Series, colname: str) -> dict:
    combo_unique = pd.Series(combo_s.astype("string").dropna().unique(), dtype="string")
    orig_unique = pd.Series(orig_s.astype("string").dropna().unique(), dtype="string")
    combo_only = combo_unique[~combo_unique.isin(orig_unique)]
    if len(combo_only) == 0:
        return {}
    rare_num = pd.to_numeric(combo_only, errors="coerce").dropna().values.astype("float64")
    rare_str = combo_only[pd.to_numeric(combo_only, errors="coerce").notna()].astype("string").values
    orig_unique_num = pd.to_numeric(orig_unique, errors="coerce").dropna()
    orig_unique = orig_unique[pd.to_numeric(orig_unique, errors="coerce").notna()].reset_index(drop=True)
    orig_unique_num = orig_unique_num.reset_index(drop=True).astype("float64")
    if len(orig_unique_num) == 0 or len(rare_num) == 0:
        return {}
    order = np.argsort(orig_unique_num.values)
    orig_vals = orig_unique_num.values[order]
    orig_txt = orig_unique.iloc[order].values
    pos = np.searchsorted(orig_vals, rare_num)
    snap_map = {}
    for s_val, v, p in zip(rare_str, rare_num, pos):
        if p <= 0:
            snap = orig_txt[0]
        elif p >= len(orig_vals):
            snap = orig_txt[-1]
        else:
            left, right = orig_vals[p - 1], orig_vals[p]
            snap = orig_txt[p - 1] if abs(v - left) <= abs(v - right) else orig_txt[p]
        snap_map[str(s_val)] = str(snap)
    log(f"{colname}: will snap {len(snap_map):,} values not in orig -> closest orig value")
    return snap_map


def apply_snap_features(combo_df, orig_df, col):
    snap_col = f"{col}_snap"
    diff_col = f"{col}_snap_diff"
    combo_df[snap_col] = combo_df[col].astype("string")
    mp = _build_snap_map(combo_df[col], orig_df[col], col)
    if mp:
        combo_df[snap_col] = combo_df[snap_col].replace(mp)
    orig_num = pd.to_numeric(combo_df[col].astype("string"), errors="coerce")
    snap_num = pd.to_numeric(combo_df[snap_col].astype("string"), errors="coerce")
    combo_df[diff_col] = (snap_num - orig_num).astype("float32")


def add_freq_rank_features(combined, num_cols):
    out = {}
    for c in num_cols:
        s = combined[c].astype("float32")
        vc = s.value_counts(normalize=True, dropna=False)
        out[f"FREQ_{c}"] = s.map(vc).fillna(0).astype("float32")
        out[f"RANK_{c}"] = s.rank(method="average", pct=True).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_num_transforms(combined, num_cols):
    out = {}
    for c in num_cols:
        v = combined[c].astype("float32")
        out[f"LOG1P_{c}"] = np.log1p(v.clip(lower=0)).astype("float32")
        out[f"SQRT_{c}"] = np.sqrt(v.clip(lower=0)).astype("float32")
        out[f"INV1P_{c}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")
        out[f"MISS_{c}"] = v.isna().astype("int8")
    return pd.DataFrame(out, index=combined.index)


def add_water_relationships(combined):
    moist = combined["Soil_Moisture"].astype("float32")
    rain = combined["Rainfall_mm"].astype("float32")
    prev = combined["Previous_Irrigation_mm"].astype("float32")
    area = combined["Field_Area_hectare"].astype("float32")
    out = {}
    out["water_input_total"] = (rain + prev).astype("float32")
    out["water_input_diff"] = (rain - prev).astype("float32")
    out["moisture_gap"] = (moist - out["water_input_total"]).astype("float32")
    out["abs_moisture_gap"] = np.abs(out["moisture_gap"]).astype("float32")
    out["moisture_to_water_ratio"] = (moist / (out["water_input_total"] + 1.0)).astype("float32")
    out["water_to_area_ratio"] = (out["water_input_total"] / (area + 1.0)).astype("float32")
    if "Rainfall_snap" in combined.columns:
        rains = combined["Rainfall_snap"].astype("float32")
        out["area_x_rainsnap"] = (area * rains).astype("float32")
    if "Previous_Irrigation_snap" in combined.columns:
        prevs = combined["Previous_Irrigation_snap"].astype("float32")
        out["area_x_prevsnap"] = (area * prevs).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_management_counts(combined):
    safe_string_df(combined, ["Irrigation_Type", "Water_Source", "Mulching_Used"])
    out = {}
    out["service_yes_count"] = (
        (combined["Mulching_Used"] == "Yes").astype("float32")
        + combined["Irrigation_Type"].isin(["Drip", "Sprinkler"]).astype("float32")
        + (combined["Water_Source"] != "Rainwater").astype("float32")
    )
    out["service_no_count"] = (
        (combined["Mulching_Used"] == "No").astype("float32")
        + (combined["Irrigation_Type"] == "Rainfed").astype("float32")
        + (combined["Water_Source"] == "Rainwater").astype("float32")
    )
    out["service_other_count"] = (
        (combined["Irrigation_Type"] == "Canal").astype("float32")
        + combined["Water_Source"].isin(["Reservoir", "River"]).astype("float32")
    )
    out["service_count"] = out["service_yes_count"].astype("float32")
    out["has_precision_irrigation"] = combined["Irrigation_Type"].isin(["Drip", "Sprinkler"]).astype("float32")
    out["has_surface_water"] = combined["Water_Source"].isin(["Reservoir", "River"]).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_yes_no_other_flags(combined):
    flag_cols = {
        "Mulching_Used": ["Yes", "No"],
        "Irrigation_Type": ["Drip", "Sprinkler", "Rainfed"],
        "Water_Source": ["Rainwater", "Groundwater"],
        "Season": ["Kharif", "Rabi"],
    }
    safe_string_df(combined, list(flag_cols))
    out = {}
    for c, vals in flag_cols.items():
        s = combined[c].astype("string")
        for v in vals:
            out[f"IS_{v}_{c}"] = (s == v).astype("float32")
        out[f"ISOTHER_{c}"] = (~s.isin(vals)).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_bins_and_cross_cats(combined):
    out = pd.DataFrame(index=combined.index)
    for col, q, name in [
        ("Rainfall_mm", 12, "Rainfall_bin"),
        ("Previous_Irrigation_mm", 12, "Previous_Irrigation_bin"),
        ("Soil_Moisture", 12, "Soil_Moisture_bin"),
        ("Rainfall_snap", 12, "Rainfall_snap_bin"),
        ("Previous_Irrigation_snap", 12, "Previous_Irrigation_snap_bin"),
    ]:
        if col not in combined.columns:
            continue
        s = pd.to_numeric(combined[col], errors="coerce")
        try:
            bins = pd.qcut(s, q=min(q, max(2, s.nunique(dropna=True))), duplicates="drop", retbins=True)[1]
            out[name] = pd.cut(s, bins=bins, include_lowest=True).astype("string")
        except Exception:
            out[name] = pd.Series(["NA"] * len(combined), index=combined.index, dtype="string")

    for a, b in CROSS_PAIRS:
        if a in combined.columns and b in combined.columns:
            out[f"{a}__{b}"] = combined[a].astype("string") + "|" + combined[b].astype("string")
    for a, b, c in TRIPLE:
        if a in combined.columns and b in combined.columns and c in combined.columns:
            out[f"{a}__{b}__{c}"] = (
                combined[a].astype("string") + "|" + combined[b].astype("string") + "|" + combined[c].astype("string")
            )
    return out


def add_cat_counts_and_rare(combined, cat_cols, rare_thresh=50):
    out = {}
    for c in cat_cols:
        vc = combined[c].astype("string").value_counts(dropna=False)
        cnt = combined[c].astype("string").map(vc).fillna(0).astype("float32")
        out[f"CAT_CNT_{c}"] = cnt
        out[f"CAT_RARE_{c}"] = (cnt <= rare_thresh).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_drift_ratio_from_orig(combined, n_train, n_test, cols, eps=1e-6):
    train_df = combined.iloc[:n_train].copy()
    orig_df = combined.iloc[n_train + n_test :].copy()
    out = {}
    for c in cols:
        p_tr = train_df[c].value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].value_counts(normalize=True, dropna=False)
        num = combined[c].map(p_tr).fillna(0)
        den = combined[c].map(p_or).fillna(eps) + eps
        out[f"DRIFT_{c}"] = np.log1p(num / den).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_orig_proba_features(combined, n_train, n_test, y_orig, cols):
    orig_df = combined.iloc[n_train + n_test :].copy()
    out = {}
    priors = np.bincount(y_orig, minlength=len(LABELS)).astype("float32") / len(y_orig)
    for cls in range(len(LABELS)):
        orig_df[f"_y_{cls}_"] = (y_orig == cls).astype("float32")
    for c in cols:
        if c not in orig_df.columns:
            continue
        g = orig_df.groupby(c, dropna=False)[[f"_y_{cls}_" for cls in range(len(LABELS))]].mean()
        for cls in range(len(LABELS)):
            out[f"ORIG_proba_{c}_c{cls}"] = combined[c].map(g[f"_y_{cls}_"]).fillna(priors[cls]).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_round_features(combined, round_specs):
    out = {}
    for c, rr in round_specs.items():
        if c not in combined.columns:
            continue
        s = pd.to_numeric(combined[c], errors="coerce")
        for r in rr:
            out[f"B_round_{c}_r{r}"] = s.round(r).astype("float32")
    return pd.DataFrame(out, index=combined.index)


def add_digit_position_features(combined, base_cols, max_dec_digits=2):
    out = {}
    for col in base_cols:
        if col not in combined.columns:
            continue
        vals = pd.to_numeric(combined[col], errors="coerce").fillna(0)
        abs_vals = vals.abs().round(max_dec_digits)
        max_val = int(abs_vals.max()) if len(abs_vals) else 0
        max_int_digits = 1 if max_val == 0 else int(np.log10(max_val)) + 1
        for i in range(max_int_digits):
            out[f"{col}_digit_pos_{i+1}"] = ((abs_vals // (10 ** i)).astype(np.int64) % 10).astype("int16")
        if not np.allclose(vals, vals.round(), equal_nan=True):
            for i in range(1, max_dec_digits + 1):
                out[f"{col}_digit_dec_{i}"] = ((abs_vals * (10 ** i)).round().astype(np.int64) % 10).astype("int16")
    return pd.DataFrame(out, index=combined.index)


def add_string_digit_features(combined, cols):
    out = {}
    for col in cols:
        if col not in combined.columns:
            continue
        s = combined[col].astype("string").fillna("NA")
        parts = s.str.split(".", n=1, expand=True)
        if parts.shape[1] == 1:
            left = parts[0].fillna("")
            right = pd.Series([""] * len(s), index=s.index)
        else:
            left, right = parts[0].fillna(""), parts[1].fillna("")
        left = left.str.replace("-", "", regex=False)
        right = right.str.pad(2, side="right", fillchar="0")
        for i in range(4):
            out[f"{col}_digit{i}"] = left.str.zfill(4).str[-4 + i].fillna("0")
        for i in range(2):
            out[f"{col}_decimal_digit{i}"] = right.str[i].fillna("0")
    return pd.DataFrame(out, index=combined.index)


def create_count_features(X: pd.DataFrame, cols_list):
    out = {}
    for cols in cols_list:
        name = f"count({','.join(cols)})"
        out[name] = X.groupby(list(cols), dropna=False).transform("size").astype("float32")
    return pd.DataFrame(out, index=X.index)


def create_groupby_features_original(X_keys: pd.DataFrame, inputs, orig_data: pd.DataFrame, target_col: str):
    out = {}
    for cols, agg in inputs:
        key_cols = list(cols)
        xk = X_keys[key_cols].copy()
        od = orig_data[key_cols + [target_col]].copy()
        for c in key_cols:
            xk[c] = xk[c].astype("string")
            od[c] = od[c].astype("string")
        if agg == "mean":
            mapping = od.groupby(key_cols, dropna=False)[target_col].mean().rename("v").reset_index()
        elif agg == "size":
            mapping = od.groupby(key_cols, dropna=False)[target_col].size().rename("v").reset_index()
        else:
            raise ValueError(agg)
        name = f"groupby({','.join(cols)})_{agg}_{target_col}_orig"
        out[name] = xk.merge(mapping, on=key_cols, how="left")["v"].astype("float32")
    return pd.DataFrame(out, index=X_keys.index)


def add_lgbm_port_features(combined, n_train, n_test, y_orig):
    out_blocks = []
    orig_df = combined.iloc[n_train + n_test :].copy()
    orig_df["_y_orig_"] = (y_orig == 2).astype("float32")

    base_cols = [c for c in combined.columns if c != TARGET]
    high_card = [c for c in base_cols if combined[c].astype("string").nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    if high_card:
        te1 = combined[high_card].astype("string").add_prefix("L_").add_suffix("_te")
        out_blocks.append(te1)
        ce1 = create_count_features(combined[high_card].astype("string"), [(c,) for c in high_card]).add_prefix("L_")
        out_blocks.append(ce1)

        mean_inputs = [((c,), "mean") for c in high_card]
        size_inputs = [((c,), "size") for c in high_card]
        gmean = create_groupby_features_original(
            combined[high_card].astype("string"), mean_inputs, orig_df[high_card + ["_y_orig_"]], "_y_orig_"
        ).add_prefix("L_")
        gsize = create_groupby_features_original(
            combined[high_card].astype("string"), size_inputs, orig_df[high_card + ["_y_orig_"]], "_y_orig_"
        ).add_prefix("L_")
        out_blocks.extend([gmean, gsize])

        for c in high_card:
            vc_comb = combined[c].astype("string").value_counts(dropna=False)
            vc_orig = orig_df[c].astype("string").value_counts(dropna=False)
            ratio = combined[c].astype("string").map(vc_comb).fillna(0) / (
                combined[c].astype("string").map(vc_orig).fillna(0) + 1
            )
            out_blocks.append(pd.DataFrame({f"L_count_ratio_{c}": ratio.astype("float32")}, index=combined.index))

    num_bins = {}
    num_cols = [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Rainfall_snap", "Previous_Irrigation_snap"] if c in combined.columns]
    for col in num_cols:
        s = pd.to_numeric(combined[col], errors="coerce")
        num_bins[f"L_{col}_round_div2_te"] = (s.round().fillna(-1).astype("int64") // 2).astype("string")
        num_bins[f"L_{col}_int_te"] = s.fillna(-1).astype("int64").astype("string")
        try:
            q = int(min(10, max(2, s.nunique(dropna=True))))
            num_bins[f"L_{col}_quantiled_te"] = pd.qcut(s, q=q, duplicates="drop").astype("string")
        except Exception:
            num_bins[f"L_{col}_quantiled_te"] = pd.Series(["NA"] * len(combined), index=combined.index, dtype="string")
        try:
            bins = np.unique(np.nanquantile(s.dropna(), np.linspace(0, 1, 11)))
            num_bins[f"L_{col}_binned_te"] = (
                pd.cut(s, bins=bins, duplicates="drop").astype("string")
                if len(bins) >= 2 else pd.Series(["NA"] * len(combined), index=combined.index, dtype="string")
            )
        except Exception:
            num_bins[f"L_{col}_binned_te"] = pd.Series(["NA"] * len(combined), index=combined.index, dtype="string")
    if num_bins:
        out_blocks.append(pd.DataFrame(num_bins, index=combined.index))

    digit_src_cols = [c for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Rainfall_snap", "Previous_Irrigation_snap"] if c in combined.columns]
    if digit_src_cols:
        digit_df = add_string_digit_features(combined, digit_src_cols)
        out_blocks.append(digit_df)
        digit_candidates = [c for c in digit_df.columns if ("Rainfall" in c or "Previous_Irrigation" in c) and ("decimal_digit" in c)][:6]
        if len(digit_candidates) >= 2:
            combs = {}
            for a, b in itertools.combinations(digit_candidates, 2):
                combs[f"L_digits_{a}__{b}_te"] = digit_df[a].astype("string") + "|" + digit_df[b].astype("string")
            out_blocks.append(pd.DataFrame(combs, index=combined.index))

    if not out_blocks:
        return pd.DataFrame(index=combined.index)
    return pd.concat(out_blocks, axis=1)


def add_radix_features(combined):
    out = {}
    for c in LOW_CARD_COLS:
        if c in combined.columns:
            codes, _ = pd.factorize(combined[c].astype("string"), sort=False)
            out[f"code_{c}"] = pd.Series(codes, index=combined.index).astype("int32")
    code_df = pd.DataFrame(out, index=combined.index)
    out2 = {}
    if "Rainfall_snap" in combined.columns:
        rain_i = np.round(combined["Rainfall_snap"].fillna(0).astype("float32") * 100.0).astype("int32")
        for c in LOW_CARD_COLS:
            if f"code_{c}" in code_df.columns:
                out2[f"rain_radix_{c}"] = (rain_i + code_df[f"code_{c}"] * RADIX).astype("int32")
    if "Previous_Irrigation_snap" in combined.columns:
        prev_i = np.round(combined["Previous_Irrigation_snap"].fillna(0).astype("float32") * 100.0).astype("int32")
        for c in LOW_CARD_COLS:
            if f"code_{c}" in code_df.columns:
                out2[f"prev_radix_{c}"] = (prev_i + code_df[f"code_{c}"] * RADIX).astype("int32")
    moist_i = combined["Soil_Moisture"].fillna(0).astype("float32").round().clip(lower=0, upper=RADIX - 1).astype("int32")
    for c in LOW_CARD_COLS:
        if f"code_{c}" in code_df.columns:
            out2[f"moisture_radix_{c}"] = (moist_i + code_df[f"code_{c}"] * RADIX).astype("int32")
    return pd.concat([code_df, pd.DataFrame(out2, index=combined.index)], axis=1)


def make_pair_feature_from_combined(combined, cols, pair_max_unique=20000):
    name = "B_pair_" + "__".join(cols)
    tmp = combined[list(cols)].astype("string").fillna("NA")
    joined = tmp.iloc[:, 0]
    for c in cols[1:]:
        joined = joined + "|" + tmp[c]
    vc = joined.value_counts(dropna=False)
    if len(vc) > pair_max_unique:
        keep = set(vc.nlargest(pair_max_unique).index)
        joined = joined.where(joined.isin(keep), other="__OTHER__")
    return pd.Series(joined, index=combined.index, name=name, dtype="string")


def nested_target_encode(train_raw, test_raw, y, te_cols):
    train_raw = train_raw.copy()
    test_raw = test_raw.copy()
    y = np.asarray(y)
    new_train = pd.DataFrame(index=train_raw.index)
    new_test = pd.DataFrame(index=test_raw.index)
    if len(te_cols) == 0:
        return new_train, new_test

    def _transform_te(enc, X_in, col_name):
        z = enc.transform(X_in)
        if isinstance(z, pd.DataFrame):
            arr = z[col_name].to_numpy(dtype=np.float32, copy=False)
        elif isinstance(z, pd.Series):
            arr = z.to_numpy(dtype=np.float32, copy=False)
        else:
            arr = np.asarray(z, dtype=np.float32)
            arr = arr[:, 0] if arr.ndim == 2 else arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for c in te_cols:
        oof = np.zeros((len(train_raw), len(LABELS)), dtype=np.float32)
        s_train = train_raw[c].astype("string").fillna("NA")
        s_test = test_raw[c].astype("string").fillna("NA")
        for itr, iva in inner_cv.split(s_train, y):
            X_itr = s_train.iloc[itr].to_frame(c)
            X_iva = s_train.iloc[iva].to_frame(c)
            for cls in range(len(LABELS)):
                enc = TargetEncoder(cv=2, smooth="auto", target_type="binary")
                y_bin = (y[itr] == cls).astype(np.int8)
                enc.fit(X_itr, y_bin)
                oof[iva, cls] = _transform_te(enc, X_iva, c)
        for cls in range(len(LABELS)):
            enc = TargetEncoder(cv=2, smooth="auto", target_type="binary")
            y_bin = (y == cls).astype(np.int8)
            enc.fit(s_train.to_frame(c), y_bin)
            xt = _transform_te(enc, s_test.to_frame(c), c)
            new_name = c.replace("_te", f"_TE_c{cls}") if c.endswith("_te") else f"{c}_TE_c{cls}"
            new_train[new_name] = oof[:, cls]
            new_test[new_name] = xt
    return new_train, new_test


def final_factorize(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = [c for c in train_df.columns if str(train_df[c].dtype).startswith(("object", "string", "category"))]
    for c in cat_cols:
        comb = pd.concat([train_df[c].astype("string"), test_df[c].astype("string")], axis=0, ignore_index=True)
        codes, _ = pd.factorize(comb, sort=False)
        codes = codes.astype("int32") + 1
        codes[codes < 1] = 0
        train_df[c] = codes[: len(train_df)]
        test_df[c] = codes[len(train_df) :]
    for df in (train_df, test_df):
        for c in df.columns:
            if str(df[c].dtype).startswith(("object", "string", "category")):
                df[c] = pd.factorize(df[c].astype("string"), sort=False)[0].astype("int32") + 1
            elif str(df[c].dtype) == "bool":
                df[c] = df[c].astype("int8")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return train_df, test_df


def build_features():
    log("Loading train/test/orig...")
    train = read_train()
    test = read_test()
    orig = read_original()

    train = train.drop(columns=[ID], errors="ignore")
    test = test.drop(columns=[ID], errors="ignore")

    add_decimal_parts_like_eda(orig)
    ensure_numeric_columns(orig, BASE_NUMS)
    for c in ["Rainfall_mm", "Previous_Irrigation_mm"]:
        orig[c.replace("_mm", "") + "_snap" if c.endswith("_mm") else f"{c}_snap"] = pd.to_numeric(orig[c], errors="coerce").astype("float32")

    combo = pd.concat([train, test], axis=0, ignore_index=True)
    add_decimal_parts_like_eda(combo)
    ensure_numeric_columns(combo, BASE_NUMS)
    apply_snap_features(combo, orig, "Rainfall_mm")
    apply_snap_features(combo, orig, "Previous_Irrigation_mm")
    combo = combo.rename(columns={"Rainfall_mm_snap": "Rainfall_snap", "Previous_Irrigation_mm_snap": "Previous_Irrigation_snap"})
    combo = combo.rename(columns={"Rainfall_mm_snap_diff": "Rainfall_snap_diff", "Previous_Irrigation_mm_snap_diff": "Previous_Irrigation_snap_diff"})

    n_train = len(train)
    n_test = len(test)
    train = combo.iloc[:n_train].copy().reset_index(drop=True)
    test = combo.iloc[n_train : n_train + n_test].copy().reset_index(drop=True)
    orig = orig.reset_index(drop=True)

    train[TARGET] = to_target_codes(train[TARGET]).astype("int8")
    y = train[TARGET].values
    y_orig = to_target_codes(orig[TARGET])

    safe_string_df(train, BASE_CATS + ["rain", "rain0", "rain1", "prev", "prev0", "prev1", "moist", "moist0", "moist1"])
    safe_string_df(test, BASE_CATS + ["rain", "rain0", "rain1", "prev", "prev0", "prev1", "moist", "moist0", "moist1"])
    safe_string_df(orig, BASE_CATS + ["rain", "rain0", "rain1", "prev", "prev0", "prev1", "moist", "moist0", "moist1"])
    ensure_numeric_columns(train, BASE_NUMS + ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])
    ensure_numeric_columns(test, BASE_NUMS + ["Rainfall_snap", "Previous_Irrigation_snap", "Rainfall_snap_diff", "Previous_Irrigation_snap_diff"])
    ensure_numeric_columns(orig, BASE_NUMS + ["Rainfall_snap", "Previous_Irrigation_snap"])

    combined = pd.concat(
        [train.drop(columns=[TARGET]), test.copy(), orig.drop(columns=[TARGET])],
        axis=0,
        ignore_index=True,
    )
    log(f"combined shape start = {combined.shape}")
    blocks = []

    log("Adding freq/rank features...")
    blocks.append(add_freq_rank_features(combined, [c for c in BASE_NUMS if c in combined.columns]))

    log("Adding numeric transforms...")
    blocks.append(add_num_transforms(combined, [c for c in BASE_NUMS + ["Rainfall_snap", "Previous_Irrigation_snap"] if c in combined.columns]))

    log("Adding relationship features...")
    blocks.append(add_water_relationships(combined))

    log("Adding management count features...")
    blocks.append(add_management_counts(combined))

    log("Adding yes/no/other flags...")
    blocks.append(add_yes_no_other_flags(combined))

    log("Adding bins + cross categorical features...")
    block_bins = add_bins_and_cross_cats(combined)
    blocks.append(block_bins)

    log("Adding category counts + rare flags...")
    cat_for_counts = dedup_keep_order(BASE_CATS + ["rain", "rain0", "rain1", "prev", "prev0", "prev1", "Rainfall_bin", "Previous_Irrigation_bin"])
    cat_for_counts = [c for c in cat_for_counts if c in pd.concat([combined, block_bins], axis=1).columns]
    combined_tmp = pd.concat([combined, block_bins], axis=1)
    blocks.append(add_cat_counts_and_rare(combined_tmp, cat_for_counts, rare_thresh=CAT_RARE_THRESH))

    log("Adding drift ratios vs orig...")
    drift_cols = [c for c in BASE_CATS + ["Crop_Type", "Season", "Irrigation_Type"] if c in combined.columns]
    blocks.append(add_drift_ratio_from_orig(pd.concat([combined, block_bins], axis=1), n_train, n_test, drift_cols))

    log("Adding orig target priors...")
    orig_proba_cols = dedup_keep_order(BASE_CATS + ["Rainfall_bin", "Previous_Irrigation_bin", "Rainfall_snap", "Previous_Irrigation_snap"])
    orig_proba_cols = [c for c in orig_proba_cols if c in pd.concat([combined, block_bins], axis=1).columns]
    blocks.append(add_orig_proba_features(pd.concat([combined, block_bins], axis=1), n_train, n_test, y_orig, orig_proba_cols))

    log("Adding round features...")
    blocks.append(add_round_features(combined, ROUND_SPECS))

    log("Adding digit position features...")
    blocks.append(add_digit_position_features(combined, DIGIT_BASE_COLS, max_dec_digits=2))

    log("Adding LightGBM-style key block...")
    combined_for_lgbm_port = pd.concat([combined, block_bins], axis=1)
    blocks.append(add_lgbm_port_features(combined_for_lgbm_port, n_train, n_test, y_orig))

    log("Adding snap/radix features...")
    blocks.append(add_radix_features(combined))

    log("Adding capped pair categorical features...")
    pair_block = {}
    combined_pairs = pd.concat([combined, block_bins], axis=1)
    for cols in PAIR_FEATURES:
        if all(c in combined_pairs.columns for c in cols):
            s = make_pair_feature_from_combined(combined_pairs, cols, pair_max_unique=PAIR_MAX_UNIQUE)
            pair_block[s.name] = s
    if pair_block:
        blocks.append(pd.DataFrame(pair_block, index=combined.index))

    X_all = pd.concat([combined] + blocks, axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()].copy()
    log(f"X_all shape after FE = {X_all.shape}")

    X_train_raw = X_all.iloc[:n_train].reset_index(drop=True)
    X_test_raw = X_all.iloc[n_train : n_train + n_test].reset_index(drop=True)
    return X_train_raw, X_test_raw, y


def run_fold(X_train_raw, X_test_raw, y, tr_idx, va_idx, xgb_params):
    class_counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    class_weights = class_counts.sum() / (len(LABELS) * class_counts)

    X_tr = X_train_raw.iloc[tr_idx].reset_index(drop=True).copy()
    X_va = X_train_raw.iloc[va_idx].reset_index(drop=True).copy()
    X_te = X_test_raw.reset_index(drop=True).copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    te_cols = [c for c in X_tr.columns if c.endswith("_te")]
    tr_te, va_te = nested_target_encode(X_tr[te_cols], X_va[te_cols], y_tr, te_cols) if te_cols else (
        pd.DataFrame(index=X_tr.index),
        pd.DataFrame(index=X_va.index),
    )
    _, te_te = nested_target_encode(X_tr[te_cols], X_te[te_cols], y_tr, te_cols) if te_cols else (
        pd.DataFrame(index=X_tr.index),
        pd.DataFrame(index=X_te.index),
    )

    if te_cols:
        X_tr = pd.concat([X_tr.drop(columns=te_cols).reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
        X_va = pd.concat([X_va.drop(columns=te_cols).reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
        X_te = pd.concat([X_te.drop(columns=te_cols).reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

    X_tr, X_va = final_factorize(X_tr, X_va)
    # X_tr is numeric by now: this call codes the string columns of the test frame on the test rows alone.
    _, X_te = final_factorize(X_tr, X_te)
    X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()].copy()
    X_va = X_va[X_tr.columns].copy()
    X_te = X_te[X_tr.columns].copy()

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=class_weights[y_tr],
        eval_set=[(X_va, y_va)],
        verbose=500,
    )

    p_va = model.predict_proba(X_va).astype(np.float32)
    p_te = model.predict_proba(X_te).astype(np.float32)
    best_it = getattr(model, "best_iteration", None)
    n_features = X_tr.shape[1]
    del X_tr, X_va, X_te, model, tr_te, va_te, te_te
    gc.collect()
    return p_va, p_te, best_it, n_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(PRESETS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    pair = args.pair
    xgb_params = {**XGB_PARAMS, **PRESETS[pair]}
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, pair, k)]
    if todo:
        X_train_raw, X_test_raw, y = build_features()
        fold_id = folds(y)
        log(f"TRAINING {pair} | train={X_train_raw.shape} | test={X_test_raw.shape}")
        for k in todo:
            t_fold = time.perf_counter()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            p_va, p_te, best_it, n_features = run_fold(X_train_raw, X_test_raw, y, tr_idx, va_idx, xgb_params)
            save_part(HERE, pair, k, p_va, p_te)
            bac = balanced_accuracy_score(y[va_idx], p_va.argmax(axis=1))
            log(f"{pair} fold {k}: BAC={bac:.6f} | best_iteration={best_it} | "
                f"time={time.perf_counter()-t_fold:.1f}s | n_features={n_features}")
    y = to_target_codes(read_train()[TARGET])
    assemble(HERE, pair, folds(y), y)


if __name__ == "__main__":
    main()
