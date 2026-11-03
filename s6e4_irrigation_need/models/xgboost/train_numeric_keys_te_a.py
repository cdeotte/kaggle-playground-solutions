#!/usr/bin/env python
"""GPU XGBoost on many label-free keys of every numeric, categorical n-grams and original-data statistics, with every
key target-encoded per fold (pair xgb_numeric_keys_ngram_te_a).

Label-free features (each frame - train, test, original - is processed on its own, including its quantile and
equal-width bins):
- for each of the 11 numerics: 10 keys (rounding to 0 / 1 / 2 decimals, floor, round(x / 2, 1), round(x / 5, 1), 10- and
  20-quantile bins, 100 equal-width bins, 50 bins of log1p), 5 digit characters and 3 digit pairs of the 3-decimal
  string; frequency of the 2-decimal value, percentile rank, log1p, sqrt, 1 / (|x| + 1);
- 6 category pairs and 4 triples; 9 arithmetic interactions (rain + / - / ratio previous irrigation, moisture x rain,
  humidity x temperature, sunlight / rain, field area x previous irrigation, pH x EC, carbon x moisture);
- for the categoricals and the 1-decimal / integer keys of the numerics: count over the frame + original rows, original
  count and their ratio, original class rates; for the numerics also the original mean class id and size per value and
  the z-score against each original class;
- pair keys (category pairs, rounded-numeric pairs) with <= 20,000 possible levels.
Per fold: each of the 211 keys gets class rates (crosstab), out-of-fold over an inner StratifiedKFold(5, shuffle, 42)
for the training rows and from all training rows for the validation / test rows (unseen levels: mean level rate); the
remaining columns (including id) are kept, strings coded by their order in the training rows.
Model: XGBoost hist on GPU, depth 6, lr 0.03, subsample 0.8, colsample_bytree 0.4, min_child_weight 5, alpha 0.1,
lambda 1, gamma 0.05, up to 1,000,000 rounds, early stopping 500 (mlogloss) on the validation fold, random_state 42.
Test = mean of the 5 fold models. Competition rows only (the original rows only feed the statistics).
About 1 hour per fold (mostly the nested encodings).
Usage: python train_numeric_keys_te_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)

PAIR = "xgb_numeric_keys_ngram_te_a"
ID_COL = ID
CLASS_NAMES = list(CLASSES)
CLASS_TO_ID = TARGET_MAP

N_FOLDS = 5
INNER_FOLDS = 5
SEED = 42
PAIR_MAX_UNIQUE = 20000
MAKE_PAIR_FEATURES = True

XGB_PARAMS = {
    "n_estimators": 1_000_000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.05,
    "early_stopping_rounds": 500,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
    "random_state": SEED,
}


def clean_df(df):
    out = df.copy()
    for c in out.columns:
        if c == TARGET:
            continue
        if out[c].dtype == "object":
            out[c] = out[c].astype(str)
    return out


def add_numeric_derived(df, num_cols, te_cols):
    out = df.copy()
    for c in num_cols:
        s = out[c].astype("float32")
        out[f"{c}_round0_te"] = np.round(s, 0).astype(str)
        out[f"{c}_round1_te"] = np.round(s, 1).astype(str)
        out[f"{c}_round2_te"] = np.round(s, 2).astype(str)
        out[f"{c}_int_te"] = np.floor(s).astype("int32").astype(str)
        out[f"{c}_div2_te"] = np.round(s / 2.0, 1).astype(str)
        out[f"{c}_div5_te"] = np.round(s / 5.0, 1).astype(str)
        out[f"{c}_q10_te"] = pd.qcut(s, q=10, duplicates="drop").astype(str)
        out[f"{c}_q20_te"] = pd.qcut(s, q=20, duplicates="drop").astype(str)
        out[f"{c}_bin100_te"] = pd.cut(s, bins=100, duplicates="drop").astype(str)
        out[f"{c}_logbin_te"] = pd.cut(np.log1p(np.maximum(s, 0)), bins=50, duplicates="drop").astype(str)
        te_cols += [
            f"{c}_round0_te", f"{c}_round1_te", f"{c}_round2_te", f"{c}_int_te",
            f"{c}_div2_te", f"{c}_div5_te", f"{c}_q10_te", f"{c}_q20_te",
            f"{c}_bin100_te", f"{c}_logbin_te"
        ]

        fmt = s.map(lambda x: f"{float(x):.3f}")
        digits = fmt.str.replace(".", "", regex=False).str.replace("-", "", regex=False).str.pad(8, fillchar="0")
        for i in range(5):
            col = f"{c}_d{i}_te"
            out[col] = digits.str[i]
            te_cols.append(col)
        out[f"{c}_d01_te"] = digits.str[0] + digits.str[1]
        out[f"{c}_d12_te"] = digits.str[1] + digits.str[2]
        out[f"{c}_d23_te"] = digits.str[2] + digits.str[3]
        te_cols += [f"{c}_d01_te", f"{c}_d12_te", f"{c}_d23_te"]

        vc = out[f"{c}_round2_te"].value_counts(dropna=False)
        out[f"{c}_freq"] = out[f"{c}_round2_te"].map(vc).astype("float32")
        out[f"{c}_rank_pct"] = s.rank(method="average", pct=True).astype("float32")
        out[f"{c}_log1p"] = np.log1p(np.maximum(s, 0)).astype("float32")
        out[f"{c}_sqrt"] = np.sqrt(np.maximum(s, 0)).astype("float32")
        out[f"{c}_inv"] = (1.0 / (np.abs(s) + 1.0)).astype("float32")
    return out, te_cols


def add_cat_ngrams(df, cat_cols, te_cols):
    out = df.copy()
    pair_cols = [
        ("Crop_Type", "Season"),
        ("Crop_Type", "Irrigation_Type"),
        ("Water_Source", "Irrigation_Type"),
        ("Soil_Type", "Crop_Type"),
        ("Region", "Crop_Type"),
        ("Crop_Growth_Stage", "Season"),
    ]
    tri_cols = [
        ("Crop_Type", "Season", "Irrigation_Type"),
        ("Soil_Type", "Crop_Type", "Region"),
        ("Water_Source", "Irrigation_Type", "Region"),
        ("Crop_Growth_Stage", "Season", "Crop_Type"),
    ]
    for cols in pair_cols:
        name = "__".join(cols) + "_te"
        out[name] = out[list(cols)].astype(str).agg("_".join, axis=1)
        te_cols.append(name)
    for cols in tri_cols:
        name = "__".join(cols) + "_te"
        out[name] = out[list(cols)].astype(str).agg("_".join, axis=1)
        te_cols.append(name)
    return out, te_cols


def add_interactions(df):
    out = df.copy()
    eps = 1e-3
    out["rain_plus_prev"] = (out["Rainfall_mm"] + out["Previous_Irrigation_mm"]).astype("float32")
    out["rain_minus_prev"] = (out["Rainfall_mm"] - out["Previous_Irrigation_mm"]).astype("float32")
    out["rain_prev_ratio"] = (out["Rainfall_mm"] / (out["Previous_Irrigation_mm"] + eps)).astype("float32")
    out["moisture_rain_prod"] = (out["Soil_Moisture"] * out["Rainfall_mm"]).astype("float32")
    out["humidity_temp_prod"] = (out["Humidity"] * out["Temperature_C"]).astype("float32")
    out["sun_rain_ratio"] = (out["Sunlight_Hours"] / (out["Rainfall_mm"] + eps)).astype("float32")
    out["area_prev_prod"] = (out["Field_Area_hectare"] * out["Previous_Irrigation_mm"]).astype("float32")
    out["ph_ec_prod"] = (out["Soil_pH"] * out["Electrical_Conductivity"]).astype("float32")
    out["carbon_moisture_prod"] = (out["Organic_Carbon"] * out["Soil_Moisture"]).astype("float32")
    return out


def add_orig_stats_features(df, orig, target_ids, group_cols, num_cols, te_cols):
    out = df.copy()
    orig2 = orig.copy()
    orig2["target_id"] = target_ids
    for c in group_cols:
        synth_cnt = pd.concat([df[c], orig[c]], axis=0).value_counts(dropna=False)
        orig_cnt = orig[c].value_counts(dropna=False)
        out[f"{c}_count"] = out[c].map(synth_cnt).fillna(0).astype("float32")
        out[f"{c}_orig_count"] = out[c].map(orig_cnt).fillna(0).astype("float32")
        out[f"{c}_count_ratio"] = (out[f"{c}_count"] / (out[f"{c}_orig_count"] + 1.0)).astype("float32")
        probs = pd.crosstab(orig2[c], orig2["target_id"], normalize="index")
        for k in range(3):
            mapper = probs.get(k, pd.Series(dtype="float32"))
            out[f"{c}_orig_p{k}"] = out[c].map(mapper).fillna(probs.get(k, pd.Series([1/3])).mean() if k in probs.columns else 1/3).astype("float32")
        if c in num_cols:
            means = orig2.groupby(c)["target_id"].mean()
            sizes = orig2.groupby(c).size()
            out[f"{c}_orig_mean_target"] = out[c].map(means).fillna(means.mean()).astype("float32")
            out[f"{c}_orig_size"] = out[c].map(sizes).fillna(0).astype("float32")
    for c in num_cols:
        for k in range(3):
            sub = orig2.loc[orig2["target_id"] == k, c].astype("float32")
            mu = float(sub.mean())
            sd = float(sub.std() + 1e-6)
            out[f"{c}_z_to_class{k}"] = ((out[c].astype("float32") - mu) / sd).astype("float32")
    return out, te_cols


def add_pair_features(df, pair_sources, te_cols):
    out = df.copy()
    if not MAKE_PAIR_FEATURES:
        return out, te_cols
    for a, b in pair_sources:
        if a not in out.columns or b not in out.columns:
            continue
        nunique = out[a].nunique(dropna=False) * out[b].nunique(dropna=False)
        if nunique > PAIR_MAX_UNIQUE:
            continue
        col = f"{a}__{b}_te"
        out[col] = out[a].astype(str) + "_" + out[b].astype(str)
        te_cols.append(col)
    return out, te_cols


def nested_multiclass_te(train_df, valid_df, test_df, te_cols, y):
    train_num = pd.DataFrame(index=train_df.index)
    valid_num = pd.DataFrame(index=valid_df.index)
    test_num = pd.DataFrame(index=test_df.index)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    for c in te_cols:
        oof_block = np.zeros((len(train_df), 3), dtype=np.float32)
        test_block = np.zeros((len(test_df), 3), dtype=np.float32)
        global_tab = pd.crosstab(train_df[c], y, normalize="index")
        global_means = np.array([(y == k).mean() for k in range(3)], dtype=np.float32)

        for itrn, ival in inner.split(train_df, y):
            trn_c = train_df.iloc[itrn][c]
            val_c = train_df.iloc[ival][c]
            y_trn = y[itrn]
            tab = pd.crosstab(trn_c, y_trn, normalize="index")
            for k in range(3):
                vals = val_c.map(tab[k]) if k in tab.columns else pd.Series(index=val_c.index, dtype="float32")
                oof_block[ival, k] = vals.fillna(global_tab[k].mean() if k in global_tab.columns else global_means[k]).to_numpy(dtype=np.float32)

        for k in range(3):
            vals = valid_df[c].map(global_tab[k]) if k in global_tab.columns else pd.Series(index=valid_df.index, dtype="float32")
            valid_num[f"{c}_te_p{k}"] = vals.fillna(global_tab[k].mean() if k in global_tab.columns else global_means[k]).astype("float32")
            test_vals = test_df[c].map(global_tab[k]) if k in global_tab.columns else pd.Series(index=test_df.index, dtype="float32")
            test_block[:, k] = test_vals.fillna(global_tab[k].mean() if k in global_tab.columns else global_means[k]).to_numpy(dtype=np.float32)
            train_num[f"{c}_te_p{k}"] = oof_block[:, k]
            test_num[f"{c}_te_p{k}"] = test_block[:, k]

    return train_num, valid_num, test_num


def encode_remaining(train_df, valid_df, test_df, te_cols):
    drop_cols = set(te_cols)
    base_cols = [c for c in train_df.columns if c not in drop_cols]
    out_tr = pd.DataFrame(index=train_df.index)
    out_va = pd.DataFrame(index=valid_df.index)
    out_te = pd.DataFrame(index=test_df.index)
    for c in base_cols:
        tr = train_df[c]
        va = valid_df[c]
        te = test_df[c]
        if tr.dtype == "object" or str(tr.dtype).startswith("category"):
            cats = pd.Index(pd.Series(tr.astype(str).unique()))
            mapper = {v: i for i, v in enumerate(cats)}
            out_tr[c] = tr.astype(str).map(mapper).fillna(-1).astype("int32")
            out_va[c] = va.astype(str).map(mapper).fillna(-1).astype("int32")
            out_te[c] = te.astype(str).map(mapper).fillna(-1).astype("int32")
        else:
            out_tr[c] = tr.astype("float32")
            out_va[c] = va.astype("float32")
            out_te[c] = te.astype("float32")
    return out_tr, out_va, out_te


def build_features():
    train = clean_df(read_train())
    test = clean_df(read_test())
    orig = clean_df(read_original())
    orig[TARGET] = orig[TARGET].astype(str)

    cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != TARGET]
    num_cols = [c for c in train.columns if c not in cat_cols + [ID_COL, TARGET]]
    orig_y = orig[TARGET].map(CLASS_TO_ID).astype(int).to_numpy()
    y = train[TARGET].map(CLASS_TO_ID).astype(int).to_numpy()

    all_train = train.drop(columns=[TARGET]).copy()
    all_test = test.copy()
    all_orig = orig.drop(columns=[TARGET]).copy()
    te_cols = []

    all_train, te_cols = add_numeric_derived(all_train, num_cols, te_cols)
    all_test, te_cols = add_numeric_derived(all_test, num_cols, te_cols)
    all_orig, te_cols = add_numeric_derived(all_orig, num_cols, te_cols)

    all_train, te_cols = add_cat_ngrams(all_train, cat_cols, te_cols)
    all_test, te_cols = add_cat_ngrams(all_test, cat_cols, te_cols)
    all_orig, te_cols = add_cat_ngrams(all_orig, cat_cols, te_cols)

    all_train = add_interactions(all_train)
    all_test = add_interactions(all_test)
    all_orig = add_interactions(all_orig)

    orig_group_cols = cat_cols + [f"{c}_round1_te" for c in num_cols] + [f"{c}_int_te" for c in num_cols]
    all_train, te_cols = add_orig_stats_features(all_train, all_orig, orig_y, orig_group_cols, num_cols, te_cols)
    all_test, te_cols = add_orig_stats_features(all_test, all_orig, orig_y, orig_group_cols, num_cols, te_cols)
    all_orig, te_cols = add_orig_stats_features(all_orig, all_orig, orig_y, orig_group_cols, num_cols, te_cols)

    pair_sources = [
        ("Crop_Type", "Season"),
        ("Crop_Type", "Irrigation_Type"),
        ("Water_Source", "Region"),
        ("Soil_Type", "Crop_Type"),
        ("Region", "Irrigation_Type"),
        ("Soil_Moisture_round1_te", "Rainfall_mm_round1_te"),
        ("Previous_Irrigation_mm_round1_te", "Rainfall_mm_round1_te"),
        ("Humidity_round1_te", "Temperature_C_round1_te"),
        ("Field_Area_hectare_round1_te", "Crop_Type"),
    ]
    all_train, te_cols = add_pair_features(all_train, pair_sources, te_cols)
    all_test, te_cols = add_pair_features(all_test, pair_sources, te_cols)
    all_orig, te_cols = add_pair_features(all_orig, pair_sources, te_cols)

    te_cols = list(dict.fromkeys([c for c in te_cols if c in all_train.columns]))

    print(f"train shape = {all_train.shape}")
    print(f"test shape  = {all_test.shape}")
    print(f"TE columns  = {len(te_cols)}")
    return all_train, all_test, y, te_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        all_train, all_test, y, te_cols = build_features()
        fold_id = folds(y)
        for k in todo:
            t0 = time.time()
            itr, iva = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            X_tr_raw = all_train.iloc[itr].reset_index(drop=True)
            X_va_raw = all_train.iloc[iva].reset_index(drop=True)
            X_te_raw = all_test.reset_index(drop=True)
            y_tr = y[itr]
            y_va = y[iva]

            X_tr_base, X_va_base, X_te_base = encode_remaining(X_tr_raw, X_va_raw, X_te_raw, te_cols)
            X_tr_te, X_va_te, X_te_te = nested_multiclass_te(X_tr_raw, X_va_raw, X_te_raw, te_cols, y_tr)

            X_tr = pd.concat([X_tr_base, X_tr_te], axis=1)
            X_va = pd.concat([X_va_base, X_va_te], axis=1)
            X_te = pd.concat([X_te_base, X_te_te], axis=1)

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)

            va_pred = model.predict_proba(X_va).astype(np.float32)
            te_pred = model.predict_proba(X_te).astype(np.float32)
            save_part(HERE, PAIR, k, va_pred, te_pred)

            bac = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
            print(f"{PAIR} fold {k}: balanced_acc = {bac:.8f}  time = {(time.time()-t0)/60:.1f} min", flush=True)

            del X_tr_raw, X_va_raw, X_te_raw, X_tr_base, X_va_base, X_te_base
            del X_tr_te, X_va_te, X_te_te, X_tr, X_va, X_te, model
            gc.collect()
    y = read_train()[TARGET].map(CLASS_TO_ID).astype(int).to_numpy()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
