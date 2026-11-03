#!/usr/bin/env python
"""GPU CatBoost (depth 6, Bernoulli) on the 150 features that a small fold-local CatBoost ranks highest.

Column groups: object columns and numerics with < 100 distinct train + test values are categorical, the other numerics
continuous; every numeric column is numeric-like. Features, built once on train + test + original (169 columns):
- per numeric-like column: value, log1p |x|, count over train + test + original, integer mod 100 / mod 10, the 3
  fractional digits; native categoricals of the exact value (5 decimals), floor, floor to even numbers and 2-decimal
  rounding;
- race-state ratios (tyre / lap, lap / stint, tyre / pit stops, degradation / tyre, delta / degradation, lap time /
  tyre), lap time and degradation x progress, position risk, a 3-level |degradation per tyre lap| flag (also as a
  category);
- native categoricals of the raw categorical columns and their count bins, the exact (lap time, delta, degradation)
  triple and 7 key pairs;
- original-label mean and original count of the 10 categorical keys (3 string columns and 7 low-cardinality
  numerics), from all original rows.
Per outer fold, all original rows are appended to the training side (weight 1); a 300-iteration depth-4 CatBoost
(lr 0.06, seed 42) fit on that training side ranks the features by importance and the top 150 are kept. CatBoost:
depth 6, lr 0.03, l2 3, random_strength 0.5, Bernoulli subsample 0.8, border_count 254, one_hot_max_size 4,
max_ctr_complexity 1, 5000 iterations, early stopping 300 on the validation fold (use_best_model=True), seed 42.
Test = mean of the 5 fold models (probabilities clipped to [1e-7, 1 - 1e-7]).

Usage: python train_feature_select.py [--folds 0,1,2,3,4] [--force]"""
import gc
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_part  # noqa: E402
from models.catboost.lib.runner import FOLDER, finish, fold_parser, todo_folds  # noqa: E402

PAIR = "catboost_top150_select_depth6"
SEED = 42
N_SELECT = 150

CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_strength=0.5,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    border_count=254,
    grow_policy="SymmetricTree",
    eval_metric="AUC",
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=300,
    use_best_model=True,
    one_hot_max_size=4,
    max_ctr_complexity=1,
)

SEL_PARAMS = dict(
    iterations=300,
    learning_rate=0.06,
    depth=4,
    loss_function="Logloss",
    task_type="GPU",
    devices="0",
    random_seed=SEED,
    verbose=0,
)


def safe_name(col):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "feature"


def sanitize_series(s):
    return s.astype("string").fillna("__NA__").str.strip()


def infer_groups(train, test):
    cols = [c for c in train.columns if c not in [TARGET, ID]]
    cats, nums, numeric_like = [], [], []
    for c in cols:
        if train[c].dtype == "object" or str(train[c].dtype).startswith("string"):
            cats.append(c)
        else:
            nun = pd.concat([train[c], test[c]], ignore_index=True).nunique(dropna=False)
            if nun < 100:
                cats.append(c)
            else:
                nums.append(c)
            numeric_like.append(c)
    return cats, nums, numeric_like


def prepare_base(train, test, orig, cats, numeric_like):
    train = train.copy(); test = test.copy(); orig = orig.copy()
    for df in (train, test, orig):
        for c in cats:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
        for c in numeric_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return train, test, orig


def add_features(train, test, orig, cats, nums, numeric_like):
    combined = pd.concat([
        train.drop(columns=[TARGET], errors="ignore"),
        test.drop(columns=[TARGET], errors="ignore"),
        orig.drop(columns=[TARGET], errors="ignore"),
    ], axis=0, ignore_index=True)
    n_train, n_test = len(train), len(test)
    y_train = train[TARGET].values.astype("int8")
    y_orig = orig[TARGET].values.astype("int8")
    feat = pd.DataFrame(index=combined.index)
    cat_native = pd.DataFrame(index=combined.index)

    # value transforms, counts and digit artifacts of every numeric-like column
    for c in numeric_like:
        if c not in combined.columns:
            continue
        p = safe_name(c)
        vals = pd.to_numeric(combined[c], errors="coerce").fillna(0).astype("float64")
        feat[p] = vals.astype("float32")
        feat[f"{p}_log1p"] = np.log1p(np.abs(vals)).astype("float32")
        feat[f"{p}_freq"] = combined[c].map(combined[c].value_counts(dropna=False)).fillna(0).astype("float32")
        int_part = np.floor(np.abs(vals)).astype("int64")
        feat[f"{p}_mod100"] = (int_part % 100).astype("int16")
        feat[f"{p}_mod10"] = (int_part % 10).astype("int16")
        frac3 = np.floor((np.abs(vals) * 1000.0) % 1000.0).astype("int32")
        feat[f"{p}_frac3"] = frac3.astype("float32")
        cat_native[f"NUMCAT_{p}_exact"] = vals.round(5).astype(str)
        cat_native[f"NUMCAT_{p}_int"] = int_part.astype(str)
        cat_native[f"NUMCAT_{p}_div2"] = (np.floor(vals / 2) * 2).astype("int64").astype(str)
        cat_native[f"NUMCAT_{p}_round2"] = vals.round(2).astype(str)

    lap = pd.to_numeric(combined.get("LapNumber", 0), errors="coerce").fillna(0).astype("float32")
    tyre = pd.to_numeric(combined.get("TyreLife", 0), errors="coerce").fillna(0).astype("float32")
    stint = pd.to_numeric(combined.get("Stint", 0), errors="coerce").fillna(0).astype("float32")
    pit = pd.to_numeric(combined.get("PitStop", 0), errors="coerce").fillna(0).astype("float32")
    pos = pd.to_numeric(combined.get("Position", 0), errors="coerce").fillna(0).astype("float32")
    pchg = pd.to_numeric(combined.get("Position_Change", 0), errors="coerce").fillna(0).astype("float32")
    lapt = pd.to_numeric(combined.get("LapTime (s)", 0), errors="coerce").fillna(0).astype("float32")
    delta = pd.to_numeric(combined.get("LapTime_Delta", 0), errors="coerce").fillna(0).astype("float32")
    degr = pd.to_numeric(combined.get("Cumulative_Degradation", 0), errors="coerce").fillna(0).astype("float32")
    prog = pd.to_numeric(combined.get("RaceProgress", 0), errors="coerce").fillna(0).astype("float32")
    feat["tyre_lap_ratio"] = (tyre / (lap + 1)).astype("float32")
    feat["lap_per_stint"] = (lap / (stint + 1)).astype("float32")
    feat["tyre_after_pit"] = (tyre / (pit + 1)).astype("float32")
    feat["degr_per_tyre"] = (degr / (tyre + 1)).astype("float32")
    feat["delta_per_degr"] = (delta / (np.abs(degr) + 1)).astype("float32")
    feat["laptime_per_tyre"] = (lapt / (tyre + 1)).astype("float32")
    feat["laptime_x_progress"] = (lapt * prog).astype("float32")
    feat["degr_x_progress"] = (degr * prog).astype("float32")
    feat["position_risk"] = ((21 - pos).clip(lower=0) + np.abs(pchg)).astype("float32")
    feat["math_broken_flag"] = pd.cut(feat["degr_per_tyre"].abs(), bins=[-np.inf, 0.1, 1.0, np.inf], labels=[0, 1, 2], right=False).astype("int8")
    cat_native["math_broken_flag_str"] = feat["math_broken_flag"].astype(str)

    # native categorical block
    for c in cats:
        if c in combined.columns:
            p = safe_name(c)
            cat_native[f"RAW_{p}"] = sanitize_series(combined[c])
            cat_native[f"{p}_freq_bin"] = pd.cut(combined[c].map(combined[c].value_counts(dropna=False)).fillna(0), bins=[-1, 5, 20, 100, 1000, np.inf], labels=False).astype(str)

    # exact high-cardinality triple and selected categorical crosses
    exact_cols = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation"] if c in combined.columns]
    if exact_cols:
        cat_native["CROSS_exact_highcard"] = combined[exact_cols].round(5).astype(str).agg("_".join, axis=1)
    top_pairs = [("Driver", "Race"), ("Driver", "Compound"), ("Race", "Compound"), ("Year", "Race"), ("Stint", "Compound"), ("PitStop", "Stint"), ("Position", "Position_Change")]
    for a, b in top_pairs:
        if a in combined.columns and b in combined.columns:
            cat_native[f"BI_{safe_name(a)}_{safe_name(b)}"] = sanitize_series(combined[a]) + "_" + sanitize_series(combined[b])

    # original-label priors (mean, count) from all original rows
    orig_part = combined.iloc[n_train + n_test:].reset_index(drop=True)
    prior_keys = [c for c in cats + ["LapNumber", "TyreLife", "PitStop", "Stint", "Position", "Position_Change"] if c in combined.columns and c in orig.columns]
    global_mean = float(y_orig.mean())
    for c in prior_keys:
        p = safe_name(c)
        key_orig = orig_part[c]
        m = pd.DataFrame({"k": key_orig, "y": y_orig}).groupby("k")["y"].mean()
        vc = key_orig.value_counts(dropna=False)
        feat[f"ORIG_MEAN_{p}"] = combined[c].map(m).fillna(global_mean).astype("float32")
        feat[f"ORIG_COUNT_{p}"] = combined[c].map(vc).fillna(0).astype("float32")

    train_feat = feat.iloc[:n_train].reset_index(drop=True)
    test_feat = feat.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_feat = feat.iloc[n_train + n_test:].reset_index(drop=True)
    train_cat = cat_native.iloc[:n_train].reset_index(drop=True)
    test_cat = cat_native.iloc[n_train:n_train + n_test].reset_index(drop=True)
    orig_cat = cat_native.iloc[n_train + n_test:].reset_index(drop=True)
    train_all = pd.concat([train_feat, train_cat], axis=1)
    test_all = pd.concat([test_feat, test_cat], axis=1)
    orig_all = pd.concat([orig_feat, orig_cat], axis=1)
    all_features = train_all.columns.tolist()
    cat_features = train_cat.columns.tolist()
    for c in cat_features:
        train_all[c] = train_all[c].astype(str)
        test_all[c] = test_all[c].astype(str)
        orig_all[c] = orig_all[c].astype(str)
    print(f"features: {len(all_features)} ({len(cat_features)} categorical)", flush=True)
    return train_all, test_all, orig_all, all_features, cat_features, y_train, y_orig


def main():
    from catboost import CatBoostClassifier, Pool

    args = fold_parser().parse_args()
    todo = todo_folds(PAIR, args.folds, args.force)
    if todo:
        train, test, orig = read_train(), read_test(), read_original()
        train[TARGET] = train[TARGET].astype("int8")
        orig[TARGET] = orig[TARGET].astype("int8")
        cats, nums, numeric_like = infer_groups(train, test)
        train, test, orig = prepare_base(train, test, orig, cats, numeric_like)
        train_all, test_all, orig_all, all_features, cat_features, y_train, y_orig = add_features(
            train, test, orig, cats, nums, numeric_like)
        fold_id = folds(y_train)

        for k in todo:
            tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            X_tr_base = train_all.iloc[tr_idx][all_features].reset_index(drop=True)
            y_tr_base = y_train[tr_idx]
            # every original row is appended to the training side
            X_tr = pd.concat([X_tr_base, orig_all[all_features]], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_tr_base, y_orig])
            X_val = train_all.iloc[va_idx][all_features].reset_index(drop=True)
            y_val = y_train[va_idx]
            X_te = test_all[all_features].reset_index(drop=True)

            # fold-local feature selection
            cat_idx_full = [all_features.index(c) for c in cat_features if c in all_features]
            sel_pool = Pool(X_tr, y_tr, cat_features=cat_idx_full, feature_names=all_features)
            sel_model = CatBoostClassifier(**SEL_PARAMS)
            sel_model.fit(sel_pool)
            imp = pd.DataFrame({"feature": all_features, "importance": sel_model.get_feature_importance()})
            top_feats = imp.sort_values("importance", ascending=False).head(min(N_SELECT, len(all_features)))["feature"].tolist()
            top_cats = [c for c in cat_features if c in top_feats]
            print(f"{PAIR} fold {k}: selected {len(top_feats)} features, {len(top_cats)} categorical", flush=True)
            del sel_pool, sel_model, imp
            gc.collect()

            cat_idx = [top_feats.index(c) for c in top_cats]
            train_pool = Pool(X_tr[top_feats], y_tr, cat_features=cat_idx, feature_names=top_feats)
            val_pool = Pool(X_val[top_feats], y_val, cat_features=cat_idx, feature_names=top_feats)
            test_pool = Pool(X_te[top_feats], cat_features=cat_idx, feature_names=top_feats)
            model = CatBoostClassifier(**CB_PARAMS)
            model.fit(train_pool, eval_set=val_pool)
            p_val = model.predict_proba(val_pool)[:, 1].astype("float32")
            p_te = model.predict_proba(test_pool)[:, 1].astype("float32")
            save_part(FOLDER, PAIR, k, p_val, p_te)
            print(f"{PAIR} fold {k}: AUC {roc_auc_score(y_val, p_val):.6f} best_iter {model.get_best_iteration()}", flush=True)
            del X_tr_base, X_tr, X_val, X_te, train_pool, val_pool, test_pool, model
            gc.collect()
    finish(PAIR, clip=1e-7)


if __name__ == "__main__":
    main()
