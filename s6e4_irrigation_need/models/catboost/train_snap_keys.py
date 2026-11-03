#!/usr/bin/env python
"""CatBoost on snapped / binned values, category crosses and nested target encodings
(pair catboost_snap_bins_keys_nested_te_a).

Label-free features (numerics median-filled; bin edges and counts over the train + test + original rows):
  - the 11 numerics and 13 relations: rainfall + / - previous irrigation, soil moisture - that sum, soil moisture /
    (sum + 1), sum / (area + 1), temperature x (100 - humidity) / 100, sunlight x temperature, conductivity x pH,
    rainfall x moisture, previous irrigation x moisture, a management count (mulching yes + irrigation type in
    Drip / Sprinkler / Canal + water source in River / Reservoir / Groundwater), surface-water and sugarcane / cotton
    flags
  - quantile bins (as interval strings) of rainfall (40), previous irrigation (40), soil moisture (40), rainfall +
    previous irrigation (40) and humidity (30)
  - rainfall, previous irrigation, soil moisture, humidity and temperature snapped to the nearest original value
    (3-decimal string) and the first decimal digit of the snapped value
  - one-hot indicators of the Mulching_Used, Irrigation_Type, Water_Source and Season values
  - 12 pair keys (6 category pairs, 6 bin x category pairs), 3 triple keys, and the rainfall / previous-irrigation
    snaps crossed with each of the 8 categoricals
  - for the 8 categoricals, bins, snaps, digits, one-hot indicators and keys: original class frequencies, count over
    train + test + original, rare flag (count <= 50), original count, log1p(train + test frequency / original
    frequency)
All string columns (54) are native CatBoost categories.
Per fold: class frequencies of 11 keys (4 categoricals, 2 snaps, 2 bins, 3 pairs), out-of-fold over an inner
StratifiedKFold(5, seed 42 + fold number 1-5) of the training rows and from all training rows for the validation /
test rows.
Model: CatBoost GPU, MultiClass, depth 7, lr 0.03, l2 8, Bernoulli subsample 0.8, min_data_in_leaf 24,
random_strength 0.8, Cosine score, 10 Newton leaf iterations, up to 20,000 iterations with early stopping (500) on the
validation fold, seed 42. Test = mean of the 5 fold models.
Usage: python train_snap_keys.py"""
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "catboost_snap_bins_keys_nested_te_a"
SEED = 42
N_FOLDS = 5

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
LOW_CARD_FOR_PAIRS = BASE_CATS[:]
PAIR_COLS = [
    ("Crop_Type", "Season"),
    ("Irrigation_Type", "Water_Source"),
    ("Crop_Type", "Irrigation_Type"),
    ("Season", "Crop_Growth_Stage"),
    ("Soil_Type", "Crop_Type"),
    ("Region", "Crop_Type"),
    ("Rainfall_bin", "Crop_Type"),
    ("Rainfall_bin", "Irrigation_Type"),
    ("Previous_Irrigation_bin", "Crop_Type"),
    ("Previous_Irrigation_bin", "Irrigation_Type"),
    ("Soil_Moisture_bin", "Crop_Growth_Stage"),
    ("Humidity_bin", "Season"),
]
TRIPLE_COLS = [
    ("Crop_Type", "Season", "Irrigation_Type"),
    ("Soil_Type", "Crop_Type", "Region"),
    ("Water_Source", "Irrigation_Type", "Region"),
]
RADIX_BASE_COLS = LOW_CARD_FOR_PAIRS[:]

CAT_RARE_THRESH = 50
DRIFT_EPS = 1e-6

NESTED_TE_COLS = [
    "Crop_Type",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Rainfall_snap",
    "Previous_Irrigation_snap",
    "Rainfall_bin",
    "Previous_Irrigation_bin",
    "Crop_Type__Season",
    "Crop_Type__Irrigation_Type",
    "Irrigation_Type__Water_Source",
]

CAT_PARAMS = dict(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    random_seed=SEED,
    iterations=20000,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=8.0,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    grow_policy="SymmetricTree",
    min_data_in_leaf=24,
    random_strength=0.8,
    score_function="Cosine",
    leaf_estimation_iterations=10,
    leaf_estimation_method="Newton",
    od_type="Iter",
    od_wait=500,
    verbose=250,
    allow_writing_files=False,
    task_type="GPU",
)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_target_codes(s):
    return pd.Series(s).map(TARGET_MAP).astype(np.int32).values


def safe_str_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing").str.strip().astype(str)


def coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            med = df[c].median()
            if pd.isna(med):
                med = 0.0
            df[c] = df[c].fillna(np.float32(med))
    return df


def make_pair_key(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return df[c1].astype("string").fillna("<NA>") + "|" + df[c2].astype("string").fillna("<NA>")


def make_triple_key(df: pd.DataFrame, c1: str, c2: str, c3: str) -> pd.Series:
    return (
        df[c1].astype("string").fillna("<NA>")
        + "|"
        + df[c2].astype("string").fillna("<NA>")
        + "|"
        + df[c3].astype("string").fillna("<NA>")
    )


def first_decimal_digit(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").astype("float32").to_numpy(copy=False)
    out = np.full(len(arr), -1, dtype=np.int16)
    for i, v in enumerate(arr):
        if np.isnan(v):
            out[i] = -1
        else:
            out[i] = int(np.floor(np.abs(v) * 10.0 + 1e-6)) % 10
    return pd.Series(out, index=s.index, dtype="int16")


def nearest_orig_snap(train_s: pd.Series, test_s: pd.Series, orig_s: pd.Series):
    """Each value replaced by the nearest value present in the original data."""
    vals = np.sort(pd.to_numeric(orig_s, errors="coerce").dropna().astype("float32").unique())
    if len(vals) == 0:
        vals = np.array([0.0], dtype=np.float32)

    def snap_one(s):
        x = pd.to_numeric(s, errors="coerce").astype("float32").to_numpy(copy=False)
        pos = np.searchsorted(vals, x)
        pos = np.clip(pos, 0, len(vals) - 1)
        left = np.clip(pos - 1, 0, len(vals) - 1)
        right = pos
        left_vals = vals[left]
        right_vals = vals[right]
        choose_right = np.abs(x - right_vals) <= np.abs(x - left_vals)
        snapped = np.where(choose_right, right_vals, left_vals).astype("float32")
        snapped[np.isnan(x)] = np.nan
        return snapped

    return snap_one(train_s), snap_one(test_s), snap_one(orig_s)


def add_orig_priors(train_df, test_df, orig_df, cols):
    y_orig = orig_df[TARGET].values.astype(np.int32)
    for c in cols:
        grp = pd.crosstab(orig_df[c], y_orig, normalize="index").reindex(columns=[0, 1, 2], fill_value=0.0)
        for cls in range(3):
            name = f"ORIG_proba_{c}_c{cls}"
            mp = grp[cls]
            train_df[name] = train_df[c].map(mp).fillna(1 / 3).astype("float32")
            test_df[name] = test_df[c].map(mp).fillna(1 / 3).astype("float32")
    return train_df, test_df


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)
    for c in cat_cols:
        vc_all = all_cat[c].astype("string").value_counts(dropna=False)
        vc_orig = orig_df[c].astype("string").value_counts(dropna=False)
        for df in (train_df, test_df, orig_df):
            cnt = df[c].astype("string").map(vc_all)
            cnt_orig = df[c].astype("string").map(vc_orig)
            df[f"CAT_CNT_{c}"] = pd.to_numeric(cnt, errors="coerce").fillna(0).astype("float32")
            df[f"CAT_RARE_{c}"] = (df[f"CAT_CNT_{c}"] <= CAT_RARE_THRESH).astype("float32")
            df[f"CAT_ORIG_CNT_{c}"] = pd.to_numeric(cnt_orig, errors="coerce").fillna(0).astype("float32")
    return train_df, test_df, orig_df


def add_drift_ratio(train_df, test_df, orig_df, cols):
    drift_cols = []
    synth_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for c in cols:
        p_sy = synth_df[c].value_counts(normalize=True)
        p_or = orig_df[c].value_counts(normalize=True)
        name = f"DRIFT_{c}"
        train_df[name] = np.log1p(train_df[c].map(p_sy).fillna(0) / (train_df[c].map(p_or).fillna(DRIFT_EPS) + DRIFT_EPS)).astype("float32")
        test_df[name] = np.log1p(test_df[c].map(p_sy).fillna(0) / (test_df[c].map(p_or).fillna(DRIFT_EPS) + DRIFT_EPS)).astype("float32")
        drift_cols.append(name)
    return train_df, test_df, drift_cols


def nested_target_encode(X_tr, y_tr, X_va, col, n_splits=5, seed=42):
    """Class frequencies of `col`: out-of-fold on X_tr (inner split), from all of X_tr for X_va."""
    tr_col = X_tr[col].astype("string").fillna("<NA>")
    va_col = X_va[col].astype("string").fillna("<NA>")
    y_tr_series = pd.Series(y_tr, index=X_tr.index)
    global_means = np.bincount(y_tr, minlength=3) / len(y_tr)
    oof_parts = np.zeros((len(X_tr), 3), dtype=np.float32)

    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for itr_idx, iva_idx in inner.split(X_tr, y_tr):
        sub_tr = X_tr.iloc[itr_idx]
        sub_y = y_tr_series.iloc[itr_idx]
        grp = pd.crosstab(sub_tr[col].astype("string").fillna("<NA>"), sub_y, normalize="index").reindex(columns=[0, 1, 2], fill_value=0.0)
        mapped = X_tr.iloc[iva_idx][col].astype("string").fillna("<NA>")
        for cls in range(3):
            oof_parts[iva_idx, cls] = mapped.map(grp[cls]).fillna(global_means[cls]).astype("float32").values

    grp_full = pd.crosstab(tr_col, y_tr_series, normalize="index").reindex(columns=[0, 1, 2], fill_value=0.0)
    va_parts = np.zeros((len(X_va), 3), dtype=np.float32)
    for cls in range(3):
        va_parts[:, cls] = va_col.map(grp_full[cls]).fillna(global_means[cls]).astype("float32").values
    return oof_parts, va_parts


def preprocess():
    train = read_train()
    test = read_test()
    orig = read_original()

    train = train.drop(columns=[ID], errors="ignore")
    test = test.drop(columns=[ID], errors="ignore")
    train[TARGET] = to_target_codes(train[TARGET])
    orig[TARGET] = to_target_codes(orig[TARGET])

    for df in (train, test, orig):
        coerce_numeric(df, BASE_NUMS)
        safe_str_inplace(df, BASE_CATS)

    train_x = train.drop(columns=[TARGET]).copy()
    test_x = test.copy()
    orig_x = orig.drop(columns=[TARGET]).copy()

    for df in (train_x, test_x, orig_x):
        df["water_input_total"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"]).astype("float32")
        df["water_input_diff"] = (df["Rainfall_mm"] - df["Previous_Irrigation_mm"]).astype("float32")
        df["moisture_gap"] = (df["Soil_Moisture"] - df["water_input_total"]).astype("float32")
        df["moisture_to_water_ratio"] = (df["Soil_Moisture"] / (df["water_input_total"] + 1)).astype("float32")
        df["water_to_area_ratio"] = (df["water_input_total"] / (df["Field_Area_hectare"] + 1)).astype("float32")
        df["heat_stress_index"] = (df["Temperature_C"] * (100 - df["Humidity"]) / 100.0).astype("float32")
        df["sun_heat_interaction"] = (df["Sunlight_Hours"] * df["Temperature_C"]).astype("float32")
        df["salinity_ph_interaction"] = (df["Electrical_Conductivity"] * df["Soil_pH"]).astype("float32")
        df["rain_x_moisture"] = (df["Rainfall_mm"] * df["Soil_Moisture"]).astype("float32")
        df["prev_x_moisture"] = (df["Previous_Irrigation_mm"] * df["Soil_Moisture"]).astype("float32")
        df["management_yes_count"] = (
            (df["Mulching_Used"] == "Yes").astype("float32")
            + df["Irrigation_Type"].isin(["Drip", "Sprinkler", "Canal"]).astype("float32")
            + df["Water_Source"].isin(["River", "Reservoir", "Groundwater"]).astype("float32")
        ).astype("float32")
        df["water_source_surface"] = df["Water_Source"].isin(["River", "Reservoir"]).astype("float32")
        df["needs_precision_irrigation"] = df["Crop_Type"].isin(["Sugarcane", "Cotton"]).astype("float32")

    for col, bins in [("Rainfall_mm", 40), ("Previous_Irrigation_mm", 40), ("Soil_Moisture", 40), ("water_input_total", 40), ("Humidity", 30)]:
        all_vals = pd.concat([train_x[col], test_x[col], orig_x[col]], axis=0)
        edges = pd.qcut(all_vals, q=bins, retbins=True, duplicates="drop")[1]
        name = col.replace("_mm", "").replace("_hectare", "").replace(" ", "_") + "_bin"
        for df in (train_x, test_x, orig_x):
            df[name] = pd.cut(df[col], bins=edges, include_lowest=True).astype(str)

    for col in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]:
        tr_snap, te_snap, or_snap = nearest_orig_snap(train_x[col], test_x[col], orig_x[col])
        snap_name = col.replace("_mm", "").replace("_hectare", "") + "_snap"
        train_x[snap_name] = pd.Series(tr_snap, index=train_x.index).round(3).astype(str)
        test_x[snap_name] = pd.Series(te_snap, index=test_x.index).round(3).astype(str)
        orig_x[snap_name] = pd.Series(or_snap, index=orig_x.index).round(3).astype(str)
        train_x[snap_name.replace("_snap", "_d1")] = first_decimal_digit(pd.Series(tr_snap, index=train_x.index)).astype(str)
        test_x[snap_name.replace("_snap", "_d1")] = first_decimal_digit(pd.Series(te_snap, index=test_x.index)).astype(str)
        orig_x[snap_name.replace("_snap", "_d1")] = first_decimal_digit(pd.Series(or_snap, index=orig_x.index)).astype(str)

    for c in ["Mulching_Used", "Irrigation_Type", "Water_Source", "Season"]:
        for df in (train_x, test_x, orig_x):
            vals = df[c].astype(str)
            top = sorted(vals.unique().tolist())
            for v in top:
                df[f"IS_{c}__{v}"] = (vals == v).astype("float32")
            df[f"ISOTHER_{c}"] = (~vals.isin(top)).astype("float32")

    for a, b in PAIR_COLS:
        if a in train_x.columns and b in train_x.columns:
            name = f"{a}__{b}"
            for df in (train_x, test_x, orig_x):
                df[name] = make_pair_key(df, a, b)
    for a, b, c in TRIPLE_COLS:
        if a in train_x.columns and b in train_x.columns and c in train_x.columns:
            name = f"{a}__{b}__{c}"
            for df in (train_x, test_x, orig_x):
                df[name] = make_triple_key(df, a, b, c)

    for snap_col in ["Rainfall_snap", "Previous_Irrigation_snap"]:
        for base in RADIX_BASE_COLS:
            if snap_col in train_x.columns:
                name = f"{snap_col}__{base}__radix"
                for df in (train_x, test_x, orig_x):
                    df[name] = df[snap_col].astype("string").fillna("<NA>") + "|" + df[base].astype("string").fillna("<NA>")

    cat_like = BASE_CATS + [c for c in train_x.columns if c.endswith("_bin") or c.endswith("_snap") or c.endswith("_d1") or "__" in c]
    cat_like = list(dict.fromkeys([c for c in cat_like if c in train_x.columns]))

    train_x, test_x = add_orig_priors(train_x, test_x, pd.concat([orig_x, orig[[TARGET]]], axis=1), BASE_CATS + [c for c in cat_like if c in orig_x.columns])
    train_x, test_x, orig_x = add_cat_counts_and_rare(train_x, test_x, orig_x, cat_like)
    train_x, test_x, drift_cols = add_drift_ratio(train_x, test_x, orig_x, BASE_CATS + [c for c in cat_like if c in orig_x.columns])

    numeric_features = BASE_NUMS + [
        "water_input_total",
        "water_input_diff",
        "moisture_gap",
        "moisture_to_water_ratio",
        "water_to_area_ratio",
        "heat_stress_index",
        "sun_heat_interaction",
        "salinity_ph_interaction",
        "rain_x_moisture",
        "prev_x_moisture",
        "management_yes_count",
        "water_source_surface",
        "needs_precision_irrigation",
    ]
    numeric_features += [c for c in train_x.columns if c.startswith("ORIG_proba_") or c.startswith("CAT_CNT_") or c.startswith("CAT_RARE_") or c.startswith("CAT_ORIG_CNT_") or c.startswith("DRIFT_") or c.startswith("IS_") or c == "ISOTHER_Mulching_Used" or c == "ISOTHER_Irrigation_Type" or c == "ISOTHER_Water_Source" or c == "ISOTHER_Season"]
    numeric_features = list(dict.fromkeys([c for c in numeric_features if c in train_x.columns]))
    cat_features = [c for c in train_x.columns if c not in numeric_features]

    for df in (train_x, test_x):
        for c in cat_features:
            df[c] = df[c].astype(str)
        for c in numeric_features:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    return train_x, test_x, train[TARGET].values.astype(np.int32), numeric_features, cat_features


def main():
    train_x, test_x, y, numeric_features, cat_features = preprocess()
    log(f"train shape={train_x.shape} test shape={test_x.shape}")
    log(f"numeric features={len(numeric_features)} categorical features={len(cat_features)}")

    fold_ids = folds(y)
    oof = np.zeros((len(train_x), 3), dtype=np.float32)
    pred_test = np.zeros((len(test_x), 3), dtype=np.float32)
    for k in range(N_FOLDS):
        fold = k + 1
        tr_idx = np.where(fold_ids != k)[0]
        va_idx = np.where(fold_ids == k)[0]
        log(f"Fold {k}")

        X_tr = train_x.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = train_x.iloc[va_idx].copy()
        y_va = y[va_idx]
        X_te = test_x.copy()

        te_cols = [c for c in NESTED_TE_COLS if c in X_tr.columns]
        for col in te_cols:
            oof_te, va_te = nested_target_encode(X_tr, y_tr, X_va, col, n_splits=5, seed=SEED + fold)
            _, te_te = nested_target_encode(X_tr, y_tr, X_te, col, n_splits=5, seed=SEED + fold)
            for cls in range(3):
                name = f"NTE_{col}_c{cls}"
                X_tr[name] = oof_te[:, cls]
                X_va[name] = va_te[:, cls]
                X_te[name] = te_te[:, cls]

        # NTE_* are numeric by construction
        fold_cat_cols = [c for c in X_tr.columns if c not in numeric_features and not c.startswith("NTE_")]

        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=fold_cat_cols,
            use_best_model=True,
            verbose=CAT_PARAMS["verbose"],
        )

        p_va = model.predict_proba(X_va).astype("float32")
        p_te = model.predict_proba(X_te).astype("float32")
        oof[va_idx] = p_va
        pred_test += p_te / N_FOLDS

        bac = balanced_accuracy_score(y_va, np.argmax(p_va, axis=1))
        log(f"Fold {k} BAC = {bac:.6f} best_iteration = {model.get_best_iteration()}")

        del X_tr, X_va, X_te, model, p_va, p_te
        gc.collect()

    log(f"{PAIR}: OOF balanced accuracy {balanced_accuracy_score(y, np.argmax(oof, axis=1)):.6f}")
    save_pair(HERE, PAIR, oof, pred_test)


if __name__ == "__main__":
    main()
