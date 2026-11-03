#!/usr/bin/env python
"""GPU XGBoost on a union of label-free blocks around a Rainfall_mm anchor, plus fold-local counts and nested class
target rates (pair xgb_rainfall_anchor_union_te_a).

Label-free features (train / test / original rows):
- pipeline A: rain + / - previous irrigation, rain / previous irrigation, moisture / humidity, temperature x sunlight,
  field area x previous irrigation, EC x pH, carbon x moisture; 2 category triples;
- pipeline B: 5- and 10-quantile bins of rain, previous irrigation, moisture, humidity and temperature (edges from the
  train rows), their rounding to -1 / 0 / 1 decimals, characters of their 4-decimal strings, fractional part, mod 10
  and mod 100;
- original-data class rates of the 8 categoricals and of Rainfall_mm;
- anchor: ANCHOR_KEY = round(10 x Rainfall_mm), its 6 digits and 5 adjacent digit pairs, each categorical x anchor key,
  5 category pairs.
Per fold: anchor statistics of the training rows (mean / std / count / median / min / max of Rainfall_mm, number of crop
types, 9 quantiles per anchor key); counts on the training rows of the 78 key columns; class rates (crosstab) of the 78
keys and the first 30 count columns, out-of-fold over an inner StratifiedKFold(5, shuffle, 42) for the training rows
and from all training rows for the validation / test rows (unseen levels: class prior); plus the 62 numeric columns
(480 model columns).
Model: XGBoost hist on GPU, depth 6, lr 0.03, subsample / colsample_bytree 0.8, min_child_weight 5, alpha 0.1,
lambda 1, gamma 0.05, up to 50,000 rounds, early stopping 500 (mlogloss) on the validation fold, random_state 42.
Test = mean of the 5 fold models. Competition rows only (the original rows only feed the class rates).
About 13 minutes per fold on one GPU (mostly the nested encodings).
Usage: python train_rainfall_anchor_union_a.py [--folds 0,1,2,3,4] [--force]"""
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
from common.data import ID, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 500)

PAIR = "xgb_rainfall_anchor_union_te_a"
ID_COL = ID
CLASS_IDS = [0, 1, 2]

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5

ANCHOR_COL = "Rainfall_mm"
ANCHOR_SCALE = 10.0
ANCHOR_DIGITS_K = 6
TE2_MAX_COMBOS = 250
ANCHOR_QUANTILES = [5, 10, 20, 40, 50, 60, 80, 90, 95]
STATS_TE_ANCHOR = ["mean", "std", "count", "nunique", "median", "min", "max"]

CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
NUMS_CAND = ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]
ROUND_SPECS = {
    "Rainfall_mm": [-1, 0, 1],
    "Previous_Irrigation_mm": [-1, 0, 1],
    "Soil_Moisture": [0, 1],
    "Humidity": [0, 1],
    "Temperature_C": [0, 1],
}
DIGIT_SPECS = {
    "Rainfall_mm": list(range(-2, 3)),
    "Previous_Irrigation_mm": list(range(-2, 3)),
    "Soil_Moisture": list(range(-2, 2)),
    "Humidity": list(range(-2, 2)),
    "Temperature_C": list(range(-2, 2)),
}
Q_LIST = [5, 10]

XGB_PARAMS = {
    "n_estimators": 50000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 0.05,
    "early_stopping_rounds": 500,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "device": "cuda",
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": SEED,
}


def sanitize(df):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype(str).fillna("__NA__")
    for c in NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


def add_qcut_bins(train, test, orig, cols, q_list):
    added = []
    for c in cols:
        for q in q_list:
            name = f"B_qcut_{c}_q{q}_te"
            try:
                tr_bin, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = tr_bin.astype(str)
                test[name] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).astype(str)
                orig[name] = pd.cut(orig[c], bins=bins, labels=False, include_lowest=True).astype(str)
            except Exception:
                train[name] = "0"
                test[name] = "0"
                orig[name] = "0"
            added.append(name)
    return train, test, orig, added


def add_round_features(train, test, orig, round_specs):
    added = []
    for c, rr in round_specs.items():
        for r in rr:
            name = f"B_round_{c}_{r}_te"
            train[name] = np.round(train[c], r).astype(str)
            test[name] = np.round(test[c], r).astype(str)
            orig[name] = np.round(orig[c], r).astype(str)
            added.append(name)
    return train, test, orig, added


def add_digit_features(train, test, orig, digit_specs):
    added_cat = []
    added_num = []
    for c, positions in digit_specs.items():
        for df in [train, test, orig]:
            fmt = df[c].map(lambda x: f"{float(x):.4f}")
            stripped = fmt.str.replace(".", "", regex=False).str.replace("-", "", regex=False).str.pad(12, fillchar="0")
            for p in positions:
                idx = p if p >= 0 else len(stripped.iloc[0]) + p
                idx = max(0, min(idx, len(stripped.iloc[0]) - 1))
                name = f"B_digit_{c}_{p}_te"
                df[name] = stripped.str[idx]
            df[f"B_frac_{c}"] = (df[c] - np.floor(df[c])).astype("float32")
            df[f"B_mod10_{c}"] = (np.floor(np.abs(df[c])).astype(int) % 10).astype("float32")
            df[f"B_mod100_{c}"] = (np.floor(np.abs(df[c])).astype(int) % 100).astype("float32")
        added_cat += [f"B_digit_{c}_{p}_te" for p in positions]
        added_num += [f"B_frac_{c}", f"B_mod10_{c}", f"B_mod100_{c}"]
    return train, test, orig, added_cat, added_num


def add_pipeline_a(train, test, orig):
    added_num = []
    added_cat = []
    for df in [train, test, orig]:
        df["A_water_total"] = (df["Rainfall_mm"] + df["Previous_Irrigation_mm"]).astype("float32")
        df["A_water_gap"] = (df["Rainfall_mm"] - df["Previous_Irrigation_mm"]).astype("float32")
        df["A_water_ratio"] = (df["Rainfall_mm"] / (df["Previous_Irrigation_mm"] + 1e-3)).astype("float32")
        df["A_sm_hum_ratio"] = (df["Soil_Moisture"] / (df["Humidity"] + 1e-3)).astype("float32")
        df["A_temp_sun_prod"] = (df["Temperature_C"] * df["Sunlight_Hours"]).astype("float32")
        df["A_field_prev_prod"] = (df["Field_Area_hectare"] * df["Previous_Irrigation_mm"]).astype("float32")
        df["A_ec_ph_prod"] = (df["Electrical_Conductivity"] * df["Soil_pH"]).astype("float32")
        df["A_org_moist_prod"] = (df["Organic_Carbon"] * df["Soil_Moisture"]).astype("float32")
        df["A_crop_season_irr_te"] = df[["Crop_Type","Season","Irrigation_Type"]].astype(str).agg("_".join, axis=1)
        df["A_source_irr_region_te"] = df[["Water_Source","Irrigation_Type","Region"]].astype(str).agg("_".join, axis=1)
    added_num += [
        "A_water_total","A_water_gap","A_water_ratio","A_sm_hum_ratio",
        "A_temp_sun_prod","A_field_prev_prod","A_ec_ph_prod","A_org_moist_prod"
    ]
    added_cat += ["A_crop_season_irr_te","A_source_irr_region_te"]
    return train, test, orig, added_cat, added_num


def add_orig_maps(train, test, orig, orig_y):
    added_num = []
    orig2 = orig.copy()
    orig2["target_id"] = orig_y
    group_cols = CAT_COLS + [ANCHOR_COL]
    for c in group_cols:
        tab = pd.crosstab(orig2[c].astype(str), orig2["target_id"], normalize="index")
        for df in [train, test, orig]:
            key = df[c].astype(str)
            for k in CLASS_IDS:
                name = f"ORIG_{c}_p{k}"
                vals = key.map(tab[k]) if k in tab.columns else pd.Series(index=key.index, dtype="float32")
                df[name] = vals.fillna(1 / len(CLASS_IDS)).astype("float32")
                if name not in added_num:
                    added_num.append(name)
    return train, test, orig, added_num


def build_anchor_extras(df):
    out = df.copy()
    anchor_key = np.round(out[ANCHOR_COL] * ANCHOR_SCALE).astype(np.int64)
    out["ANCHOR_KEY"] = anchor_key
    s = anchor_key.astype(str).str.replace("-", "", regex=False).str.pad(ANCHOR_DIGITS_K, fillchar="0")
    cat_added = []
    num_added = ["ANCHOR_KEY"]
    for i in range(ANCHOR_DIGITS_K):
        col = f"ANCHOR_D{i+1}_te"
        out[col] = s.str[i]
        cat_added.append(col)
    for i in range(ANCHOR_DIGITS_K - 1):
        col = f"ANCHOR_DPAIR_{i+1}{i+2}_te"
        out[col] = s.str[i] + s.str[i + 1]
        cat_added.append(col)
    for c in CAT_COLS:
        col = f"ANCHORCAT_{c}_te"
        out[col] = out[c].astype(str) + "_" + anchor_key.astype(str)
        cat_added.append(col)
    pair_candidates = [("Crop_Type","Season"),("Crop_Type","Irrigation_Type"),("Season","Irrigation_Type"),("Water_Source","Region"),("Soil_Type","Crop_Type")]
    for c1, c2 in pair_candidates:
        col = f"PAIRID_{c1}_{c2}_te"
        out[col] = out[c1].astype(str) + "_" + out[c2].astype(str)
        cat_added.append(col)
    return out, cat_added, num_added


def add_anchor_stats(train_df, apply_df):
    feats = pd.DataFrame(index=apply_df.index)
    grp = train_df.groupby("ANCHOR_KEY", observed=False)
    for stat in STATS_TE_ANCHOR:
        if stat == "count":
            vals = grp[ANCHOR_COL].count()
        elif stat == "nunique":
            vals = grp["Crop_Type"].nunique()
        else:
            vals = getattr(grp[ANCHOR_COL], stat)()
        feats[f"ANCHORSTAT_{stat}"] = apply_df["ANCHOR_KEY"].map(vals).fillna(vals.mean()).astype("float32")
    for q in ANCHOR_QUANTILES:
        qq = grp[ANCHOR_COL].quantile(q / 100.0)
        feats[f"ANCHORQ_{q}"] = apply_df["ANCHOR_KEY"].map(qq).fillna(qq.mean()).astype("float32")
    return feats


def add_counts(train_df, apply_df, cat_cols):
    feats = pd.DataFrame(index=apply_df.index)
    for c in cat_cols:
        vc = train_df[c].astype(str).value_counts(dropna=False)
        feats[f"CE_{c}"] = apply_df[c].astype(str).map(vc).fillna(0).astype("float32")
    return feats


def nested_multiclass_te(train_df, valid_df, test_df, te_cols, y):
    tr_out = pd.DataFrame(index=train_df.index)
    va_out = pd.DataFrame(index=valid_df.index)
    te_out = pd.DataFrame(index=test_df.index)
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    priors = np.bincount(y, minlength=len(CLASS_IDS)) / len(y)

    te_cols2 = te_cols[:TE2_MAX_COMBOS]
    for c in te_cols2:
        oof_block = np.zeros((len(train_df), len(CLASS_IDS)), dtype=np.float32)
        global_tab = pd.crosstab(train_df[c].astype(str), y, normalize="index")
        for itrn, ival in inner.split(train_df, y):
            xtr = train_df.iloc[itrn][c].astype(str)
            xva = train_df.iloc[ival][c].astype(str)
            ytr = y[itrn]
            tab = pd.crosstab(xtr, ytr, normalize="index")
            for k in CLASS_IDS:
                vals = xva.map(tab[k]) if k in tab.columns else pd.Series(index=xva.index, dtype="float32")
                oof_block[ival, k] = vals.fillna(priors[k]).to_numpy(dtype=np.float32)
        for k in CLASS_IDS:
            tr_out[f"TE_{c}_p{k}"] = oof_block[:, k]
            vv = valid_df[c].astype(str)
            tt = test_df[c].astype(str)
            va_vals = vv.map(global_tab[k]) if k in global_tab.columns else pd.Series(index=vv.index, dtype="float32")
            te_vals = tt.map(global_tab[k]) if k in global_tab.columns else pd.Series(index=tt.index, dtype="float32")
            va_out[f"TE_{c}_p{k}"] = va_vals.fillna(priors[k]).astype("float32")
            te_out[f"TE_{c}_p{k}"] = te_vals.fillna(priors[k]).astype("float32")
    return tr_out, va_out, te_out


def factorize_non_numeric(train_df, valid_df, test_df, skip_cols):
    use_cols = [c for c in train_df.columns if c not in skip_cols]
    tr_out = pd.DataFrame(index=train_df.index)
    va_out = pd.DataFrame(index=valid_df.index)
    te_out = pd.DataFrame(index=test_df.index)
    for c in use_cols:
        if pd.api.types.is_numeric_dtype(train_df[c]):
            tr_out[c] = train_df[c].astype("float32")
            va_out[c] = valid_df[c].astype("float32")
            te_out[c] = test_df[c].astype("float32")
        else:
            tr = train_df[c].astype(str)
            cats = pd.Index(tr.unique())
            mapper = {v: i for i, v in enumerate(cats)}
            tr_out[c] = tr.map(mapper).fillna(-1).astype("int32")
            va_out[c] = valid_df[c].astype(str).map(mapper).fillna(-1).astype("int32")
            te_out[c] = test_df[c].astype(str).map(mapper).fillna(-1).astype("int32")
    return tr_out, va_out, te_out


def build_features():
    print("Loading data...")
    train = sanitize(read_train())
    test = sanitize(read_test())
    orig = sanitize(read_original())

    train[TARGET] = train[TARGET].map(TARGET_MAP).astype(int)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP).astype(int)
    y = train[TARGET].values
    orig_y = orig[TARGET].values

    train_base = train.drop(columns=[TARGET]).copy()
    test_base = test.copy()
    orig_base = orig.drop(columns=[TARGET]).copy()

    train_a, test_a, orig_a, a_cat, a_num = add_pipeline_a(train_base.copy(), test_base.copy(), orig_base.copy())
    train_b, test_b, orig_b, b_q = add_qcut_bins(train_a, test_a, orig_a, NUMS_CAND, Q_LIST)
    train_b, test_b, orig_b, b_round = add_round_features(train_b, test_b, orig_b, ROUND_SPECS)
    train_b, test_b, orig_b, b_digit_cat, b_digit_num = add_digit_features(train_b, test_b, orig_b, DIGIT_SPECS)
    train_b, test_b, orig_b, orig_num = add_orig_maps(train_b, test_b, orig_b, orig_y)
    train_b, anchor_cat, anchor_num = build_anchor_extras(train_b)
    test_b, _, _ = build_anchor_extras(test_b)
    orig_b, _, _ = build_anchor_extras(orig_b)

    all_cat_te_cols = CAT_COLS + a_cat + b_q + b_round + b_digit_cat + anchor_cat
    all_num_cols = NUM_COLS + a_num + b_digit_num + orig_num + anchor_num
    all_cat_te_cols = list(dict.fromkeys(all_cat_te_cols))
    all_num_cols = list(dict.fromkeys([c for c in all_num_cols if c in train_b.columns]))

    print(f"Base numeric cols: {len(all_num_cols)}")
    print(f"TE/categorical cols: {len(all_cat_te_cols)}")
    return train_b, test_b, y, all_cat_te_cols, all_num_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train_b, test_b, y, all_cat_te_cols, all_num_cols = build_features()
        fold_id = folds(y)
        for k in todo:
            t0 = time.time()
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            print("\n" + "=" * 70)
            print(f"Fold {k}", flush=True)
            X_tr = train_b.iloc[tr_idx].reset_index(drop=True).copy()
            X_va = train_b.iloc[va_idx].reset_index(drop=True).copy()
            X_te = test_b.reset_index(drop=True).copy()
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            tr_anchor = add_anchor_stats(X_tr, X_tr)
            va_anchor = add_anchor_stats(X_tr, X_va)
            te_anchor = add_anchor_stats(X_tr, X_te)

            tr_counts = add_counts(X_tr, X_tr, all_cat_te_cols)
            va_counts = add_counts(X_tr, X_va, all_cat_te_cols)
            te_counts = add_counts(X_tr, X_te, all_cat_te_cols)

            te_cols_for_nested = all_cat_te_cols + [c for c in tr_counts.columns[:30]]
            X_tr_te, X_va_te, X_te_te = nested_multiclass_te(
                pd.concat([X_tr[all_cat_te_cols], tr_counts.iloc[:, :30]], axis=1),
                pd.concat([X_va[all_cat_te_cols], va_counts.iloc[:, :30]], axis=1),
                pd.concat([X_te[all_cat_te_cols], te_counts.iloc[:, :30]], axis=1),
                te_cols_for_nested,
                y_tr,
            )

            skip_cols = list(set(all_cat_te_cols))
            X_tr_num, X_va_num, X_te_num = factorize_non_numeric(X_tr, X_va, X_te, skip_cols=skip_cols)

            X_tr_final = pd.concat([X_tr_num[all_num_cols], tr_anchor, tr_counts, X_tr_te], axis=1)
            X_va_final = pd.concat([X_va_num[all_num_cols], va_anchor, va_counts, X_va_te], axis=1)
            X_te_final = pd.concat([X_te_num[all_num_cols], te_anchor, te_counts, X_te_te], axis=1)

            for df in [X_tr_final, X_va_final, X_te_final]:
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

            model = XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr_final, y_tr, eval_set=[(X_va_final, y_va)], verbose=500)

            va_pred = model.predict_proba(X_va_final).astype(np.float32)
            te_pred = model.predict_proba(X_te_final).astype(np.float32)
            save_part(HERE, PAIR, k, va_pred, te_pred)

            fold_ba = balanced_accuracy_score(y_va, va_pred.argmax(axis=1))
            print(f"{PAIR} fold {k}: BAC = {fold_ba:.8f} | time = {(time.time()-t0)/60:.1f} min", flush=True)

            del X_tr, X_va, X_te, tr_anchor, va_anchor, te_anchor, tr_counts, va_counts, te_counts
            del X_tr_te, X_va_te, X_te_te, X_tr_num, X_va_num, X_te_num, X_tr_final, X_va_final, X_te_final, model
            gc.collect()
    y = read_train()[TARGET].map(TARGET_MAP).astype(int).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
