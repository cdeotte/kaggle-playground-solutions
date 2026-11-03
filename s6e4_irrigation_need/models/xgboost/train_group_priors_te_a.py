#!/usr/bin/env python
"""GPU XGBoost on original-data group priors and class profiles, fold-local interaction target encodings and engineered
numerics (pair xgb_group_priors_interaction_te_a).

Per fold (statistics from the fold's training rows unless noted):
- the 11 numerics, 8 engineered numerics (rain + previous irrigation, moisture - humidity, water per field area,
  sunlight x temperature, pH x EC, carbon x moisture, rain / moisture, previous irrigation / moisture) and rounding to
  1 / 2 decimals of 5 numerics;
- counts of the 8 categoricals, the Crop_Type x Irrigation_Type group and 6 category crosses; percentile ranks of the
  11 numerics;
- from the original rows: Laplace-smoothed class rates of the categoricals, the group and the crosses; group class rates
  and a class-profile block (z-distance of 6 numerics to each original class mean, exp(-distance), normalized
  posterior, margin, entropy). As run, the original labels stay strings while these lookups are indexed by the integer
  class ids, so the class rates come out as 1/3 for levels seen in the original data (0 otherwise) and the distance
  block is all missing;
- smoothed (20) 3-class target encodings of the 6 crosses, fitted on the training rows (also applied to them) and
  their row mean / std / min / max.
138 columns. Balanced class weights on the training and validation rows.
Model: XGBoost hist on GPU, depth 7, lr 0.01, subsample 0.9, colsample_bytree 0.4, min_child_weight 6, alpha 0.05,
lambda 5, max_bin 1024, up to 20,000 rounds, early stopping 300 (weighted mlogloss) on the validation fold,
random_state 42. Test = mean of the 5 fold models.
About 5 minutes per fold on one GPU.
Usage: python train_group_priors_te_a.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, log_loss

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import CLASSES, TARGET, TARGET_MAP, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "xgb_group_priors_interaction_te_a"
SEED = 42
N_FOLDS = 5
CLASS_NAMES = list(CLASSES)
CLASS_IDS = list(range(len(CLASS_NAMES)))
CLASS_TO_INT = TARGET_MAP

CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
FRAILTY_NUMS = [
    "Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
    "Sunlight_Hours", "Previous_Irrigation_mm",
]
INTERACTION_COLS = [
    "Crop_Type__Season",
    "Irrigation_Type__Water_Source",
    "Soil_Type__Region",
    "Crop_Growth_Stage__Season",
    "Crop_Type__Season__Irrigation_Type",
    "Soil_Type__Crop_Type__Region",
]

xgb_params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "mlogloss",
    "n_estimators": 20000,
    "early_stopping_rounds": 300,
    "max_depth": 7,
    "learning_rate": 0.01,
    "subsample": 0.9,
    "colsample_bytree": 0.4,
    "min_child_weight": 6,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
    "max_bin": 1024,
    "random_state": SEED,
    "n_jobs": 32,
    "verbosity": 0,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def add_base_features(df):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df["Crop_Type__Season"] = df["Crop_Type"] + "_" + df["Season"]
    df["Irrigation_Type__Water_Source"] = df["Irrigation_Type"] + "_" + df["Water_Source"]
    df["Soil_Type__Region"] = df["Soil_Type"] + "_" + df["Region"]
    df["Crop_Growth_Stage__Season"] = df["Crop_Growth_Stage"] + "_" + df["Season"]
    df["Crop_Type__Season__Irrigation_Type"] = df["Crop_Type"] + "_" + df["Season"] + "_" + df["Irrigation_Type"]
    df["Soil_Type__Crop_Type__Region"] = df["Soil_Type"] + "_" + df["Crop_Type"] + "_" + df["Region"]
    df["Frailty_Group"] = df["Crop_Type"] + "_" + df["Irrigation_Type"]

    df["rain_plus_prev_irrig"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["moisture_gap"] = df["Soil_Moisture"] - df["Humidity"]
    df["water_intensity"] = df["rain_plus_prev_irrig"] / (df["Field_Area_hectare"] + 1.0)
    df["sun_temp_interaction"] = df["Sunlight_Hours"] * df["Temperature_C"]
    df["ph_ec_interaction"] = df["Soil_pH"] * df["Electrical_Conductivity"]
    df["organic_moisture"] = df["Organic_Carbon"] * df["Soil_Moisture"]
    df["rain_to_moisture"] = df["Rainfall_mm"] / (df["Soil_Moisture"] + 1.0)
    df["prev_to_moisture"] = df["Previous_Irrigation_mm"] / (df["Soil_Moisture"] + 1.0)

    for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]:
        df[f"{c}_round1"] = np.round(df[c], 1)
        df[f"{c}_round2"] = np.round(df[c], 2)
    return df

def add_freq_features(train_ref, apply_df, cols):
    out = pd.DataFrame(index=apply_df.index)
    for c in cols:
        vc = train_ref[c].astype(str).value_counts(dropna=False)
        out[f"freq_{c}"] = apply_df[c].astype(str).map(vc).fillna(0).astype(np.float32)
    return out

def add_rank_features(train_ref, apply_df, cols):
    out = pd.DataFrame(index=apply_df.index)
    for c in cols:
        ref = pd.to_numeric(train_ref[c], errors="coerce").fillna(train_ref[c].median()).values.astype(np.float32)
        vals = pd.to_numeric(apply_df[c], errors="coerce").fillna(train_ref[c].median()).values.astype(np.float32)
        sorter = np.sort(ref)
        left = np.searchsorted(sorter, vals, side="left")
        right = np.searchsorted(sorter, vals, side="right")
        out[f"rank_{c}"] = ((left + right) / 2.0 / len(sorter)).astype(np.float32)
    return out

def add_orig_prob_features(orig_df, apply_df, cat_cols, target_col):
    out = pd.DataFrame(index=apply_df.index)
    global_probs = orig_df[target_col].value_counts(normalize=True).reindex(CLASS_IDS).fillna(0.0).values.astype(np.float32)
    for c in cat_cols:
        grp = orig_df.groupby(c)[target_col].value_counts().unstack(fill_value=0).reindex(columns=CLASS_IDS, fill_value=0)
        probs = (grp + 1.0).div(grp.sum(axis=1) + len(CLASS_NAMES), axis=0)
        prob_map = probs.to_dict(orient="index")
        mapped = apply_df[c].astype(str).map(prob_map)
        arr = np.array([
            [row.get(i, global_probs[i]) for i in CLASS_IDS] if isinstance(row, dict) else global_probs
            for row in mapped
        ], dtype=np.float32)
        for i, cls in enumerate(CLASS_NAMES):
            out[f"ORIG_{c}_{cls}"] = arr[:, i].astype(np.float32)
    return out

def fit_orig_frailty(orig_df):
    frailty_group = orig_df["Frailty_Group"].astype(str)
    grp_counts = orig_df.groupby(["Frailty_Group", TARGET]).size().unstack(fill_value=0).reindex(columns=CLASS_IDS, fill_value=0)
    grp_probs = (grp_counts + 1.0).div(grp_counts.sum(axis=1) + len(CLASS_NAMES), axis=0)
    means = orig_df.groupby(TARGET)[FRAILTY_NUMS].mean().reindex(CLASS_IDS)
    stds = orig_df.groupby(TARGET)[FRAILTY_NUMS].std().reindex(CLASS_IDS).fillna(1.0).replace(0.0, 1.0)
    priors = orig_df[TARGET].value_counts(normalize=True).reindex(CLASS_IDS).fillna(0.0)
    return grp_probs, means, stds, priors

def add_frailty_features(apply_df, frailty_fit):
    grp_probs, means, stds, priors = frailty_fit
    out = pd.DataFrame(index=apply_df.index)
    mapped = apply_df["Frailty_Group"].astype(str).map(grp_probs.to_dict(orient="index"))
    grp_arr = np.array([
        [row.get(i, float(priors[i])) for i in CLASS_IDS] if isinstance(row, dict) else [float(priors[i]) for i in CLASS_IDS]
        for row in mapped
    ], dtype=np.float32)
    for i, cls in enumerate(CLASS_NAMES):
        out[f"frailty_grp_{cls}"] = grp_arr[:, i]

    dist_mat = []
    vals = apply_df[FRAILTY_NUMS].copy()
    for c in FRAILTY_NUMS:
        vals[c] = pd.to_numeric(vals[c], errors="coerce").fillna(means[c].mean())
    vals = vals.values.astype(np.float32)
    for cls_idx, cls in enumerate(CLASS_NAMES):
        mu = means.loc[cls_idx].values.astype(np.float32)
        sd = stds.loc[cls_idx].values.astype(np.float32)
        z = (vals - mu) / sd
        dist = np.sqrt((z ** 2).sum(axis=1)).astype(np.float32)
        dist_mat.append(dist)
        out[f"frailty_dist_{cls}"] = dist
        out[f"frailty_score_{cls}"] = np.exp(-dist).astype(np.float32)
    dist_mat = np.vstack(dist_mat).T
    score_mat = np.exp(-dist_mat)
    score_mat = score_mat / score_mat.sum(axis=1, keepdims=True)
    for i, cls in enumerate(CLASS_NAMES):
        out[f"frailty_post_{cls}"] = score_mat[:, i].astype(np.float32)
    sorted_scores = np.sort(score_mat, axis=1)
    out["frailty_margin"] = (sorted_scores[:, -1] - sorted_scores[:, -2]).astype(np.float32)
    out["frailty_entropy"] = (-(score_mat * np.log(score_mat + 1e-12)).sum(axis=1)).astype(np.float32)
    out["frailty_group_max"] = grp_arr.max(axis=1).astype(np.float32)
    out["frailty_group_margin"] = (np.sort(grp_arr, axis=1)[:, -1] - np.sort(grp_arr, axis=1)[:, -2]).astype(np.float32)
    return out

def add_multiclass_te(train_df, apply_df, cols, y, min_samples_leaf=10, smoothing=20.0):
    out = pd.DataFrame(index=apply_df.index)
    y_series = pd.Series(y, index=train_df.index)
    global_probs = y_series.value_counts(normalize=True).reindex(range(len(CLASS_NAMES))).fillna(0.0)
    for c in cols:
        tmp = train_df[[c]].copy()
        tmp["y"] = y
        counts = tmp.groupby(c).size()
        for cls_idx, cls in enumerate(CLASS_NAMES):
            means = tmp.assign(target=(tmp["y"] == cls_idx).astype(np.float32)).groupby(c)["target"].mean()
            smooth = (counts * means + smoothing * global_probs[cls_idx]) / (counts + smoothing)
            out[f"te_{c}_{cls}"] = apply_df[c].map(smooth).fillna(global_probs[cls_idx]).astype(np.float32)
    return out

def assemble_features(df_tr_raw, df_va_raw, df_te_raw, orig_df):
    df_tr = add_base_features(df_tr_raw)
    df_va = add_base_features(df_va_raw)
    df_te = add_base_features(df_te_raw)
    orig_fe = add_base_features(orig_df)

    y_tr = df_tr_raw[TARGET].map(CLASS_TO_INT).astype(np.int64).values
    y_va = df_va_raw[TARGET].map(CLASS_TO_INT).astype(np.int64).values

    base_num_cols = NUM_COLS + [
        "rain_plus_prev_irrig", "moisture_gap", "water_intensity", "sun_temp_interaction",
        "ph_ec_interaction", "organic_moisture", "rain_to_moisture", "prev_to_moisture",
    ] + [f"{c}_round1" for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]] \
      + [f"{c}_round2" for c in ["Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture", "Humidity", "Temperature_C"]]

    X_tr = df_tr[base_num_cols].copy()
    X_va = df_va[base_num_cols].copy()
    X_te = df_te[base_num_cols].copy()

    for c in base_num_cols:
        med = pd.to_numeric(X_tr[c], errors="coerce").median()
        X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").fillna(med).astype(np.float32)
        X_va[c] = pd.to_numeric(X_va[c], errors="coerce").fillna(med).astype(np.float32)
        X_te[c] = pd.to_numeric(X_te[c], errors="coerce").fillna(med).astype(np.float32)

    freq_cols = CAT_COLS + ["Frailty_Group"] + INTERACTION_COLS
    X_tr = pd.concat([X_tr, add_freq_features(df_tr, df_tr, freq_cols)], axis=1)
    X_va = pd.concat([X_va, add_freq_features(df_tr, df_va, freq_cols)], axis=1)
    X_te = pd.concat([X_te, add_freq_features(df_tr, df_te, freq_cols)], axis=1)

    X_tr = pd.concat([X_tr, add_rank_features(df_tr, df_tr, NUM_COLS)], axis=1)
    X_va = pd.concat([X_va, add_rank_features(df_tr, df_va, NUM_COLS)], axis=1)
    X_te = pd.concat([X_te, add_rank_features(df_tr, df_te, NUM_COLS)], axis=1)

    orig_prob_cols = CAT_COLS + ["Frailty_Group"] + INTERACTION_COLS
    X_tr = pd.concat([X_tr, add_orig_prob_features(orig_fe, df_tr, orig_prob_cols, TARGET)], axis=1)
    X_va = pd.concat([X_va, add_orig_prob_features(orig_fe, df_va, orig_prob_cols, TARGET)], axis=1)
    X_te = pd.concat([X_te, add_orig_prob_features(orig_fe, df_te, orig_prob_cols, TARGET)], axis=1)

    frailty_fit = fit_orig_frailty(orig_fe)
    X_tr = pd.concat([X_tr, add_frailty_features(df_tr, frailty_fit)], axis=1)
    X_va = pd.concat([X_va, add_frailty_features(df_va, frailty_fit)], axis=1)
    X_te = pd.concat([X_te, add_frailty_features(df_te, frailty_fit)], axis=1)

    te_tr = add_multiclass_te(df_tr, df_tr, INTERACTION_COLS, y_tr)
    te_va = add_multiclass_te(df_tr, df_va, INTERACTION_COLS, y_tr)
    te_te = add_multiclass_te(df_tr, df_te, INTERACTION_COLS, y_tr)
    for frame in [te_tr, te_va, te_te]:
        te_cols = frame.columns.tolist()
        frame["te_mean"] = frame[te_cols].mean(axis=1).astype(np.float32)
        frame["te_std"] = frame[te_cols].std(axis=1).fillna(0).astype(np.float32)
        frame["te_min"] = frame[te_cols].min(axis=1).astype(np.float32)
        frame["te_max"] = frame[te_cols].max(axis=1).astype(np.float32)
    X_tr = pd.concat([X_tr, te_tr], axis=1)
    X_va = pd.concat([X_va, te_va], axis=1)
    X_te = pd.concat([X_te, te_te], axis=1)

    return X_tr, X_va, X_te, y_tr, y_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        log("Loading data...")
        train_df = read_train()
        test_df = read_test()
        orig_df = read_original()

        y_all = train_df[TARGET].map(CLASS_TO_INT).astype(np.int64).values
        class_counts = np.bincount(y_all, minlength=3)
        class_weights = len(y_all) / (3.0 * np.maximum(class_counts, 1))
        sample_weight_all = class_weights[y_all].astype(np.float32)

        fold_id = folds(y_all)
        for k in todo:
            fold = k + 1
            tr_idx, va_idx = np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)
            log("=" * 70)
            log(f"Fold {fold}")
            df_tr_raw = train_df.iloc[tr_idx].reset_index(drop=True).copy()
            df_va_raw = train_df.iloc[va_idx].reset_index(drop=True).copy()
            df_te_raw = test_df.reset_index(drop=True).copy()

            X_tr, X_va, X_te, y_tr, y_va = assemble_features(df_tr_raw, df_va_raw, df_te_raw, orig_df)
            w_tr = sample_weight_all[tr_idx]
            w_va = sample_weight_all[va_idx]

            if fold == 1:
                log(f"Total features: {X_tr.shape[1]}")

            model = xgb.XGBClassifier(**xgb_params)
            model.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                sample_weight_eval_set=[w_va],
                verbose=False,
            )

            val_pred = model.predict_proba(X_va).astype(np.float32)
            te_pred = model.predict_proba(X_te).astype(np.float32)
            save_part(HERE, PAIR, k, val_pred, te_pred)

            fold_bac = balanced_accuracy_score(y_va, val_pred.argmax(axis=1))
            fold_ll = log_loss(y_va, val_pred, labels=[0, 1, 2])
            log(f"{PAIR} fold {k}: BAC={fold_bac:.8f}, LogLoss={fold_ll:.8f}")

            del df_tr_raw, df_va_raw, X_tr, X_va, X_te, y_tr, y_va, model, val_pred, te_pred
            gc.collect()
    y = read_train()[TARGET].map(CLASS_TO_INT).astype(np.int64).values
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
