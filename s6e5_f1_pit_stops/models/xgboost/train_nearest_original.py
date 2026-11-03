#!/usr/bin/env python
"""GPU XGBoost on nearest-original-row features and original-data priors (pair xgb_nearest_original_knn).

Features (90 columns; the competition labels are used only by the model):
  - the 14 raw columns (Driver / Compound / Race and the crosses ordinal-encoded over train + test + original);
  - 10 race-state ratios (tyre / lap, lap / stint, tyre / pit stops, degradation / tyre, delta / degradation, ...);
  - nearest original rows: StandardScaler fitted on the original rows over 20 numerics (10 raw + the 10 ratios), a cKDTree
    on the scaled original rows; for k in 1, 3, 5, 10 the mean and std of the original targets of the k nearest original
    rows and the mean distance;
  - original priors: mean target and count in the original data of every categorical-like column (strings and the
    numerics with < 100 levels) and of 6 crosses (Driver x Race, Driver x Compound, Race x Compound, Year x Race,
    Stint x Compound, PitStop x Stint);
  - for the 4 numerics with >= 100 levels: value frequency over train + test + original, percentile rank among the
    original values, value rounded to 2 decimals, 3 digits after the decimal point.
Model: depth 6, lr 0.03, 500 rounds (no early stopping), subsample 0.8, colsample_bytree 0.8, random_state 43..47 for
folds 0..4. Test = mean of the 5 fold models; predictions clipped to [1e-7, 1 - 1e-7].
Usage: python train_nearest_original.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = "xgb_nearest_original_knn"
SEED = 42

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=6,
    learning_rate=0.03,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    tree_method="hist",
    device="cuda",
    n_jobs=8,
    verbosity=0,
)


def safe_name(col):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "feature"


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int8")
    return s.astype("string").str.strip().str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(0).astype("int8")


def sanitize_series(s):
    return s.astype("string").fillna("missing").str.strip()


def infer_groups(train, test):
    features = [c for c in train.columns if c not in [ID, TARGET]]
    cats, nums, numeric_like = [], [], []
    for c in features:
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


def pctrank_against(values, reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    ref = np.sort(reference)
    return (np.searchsorted(ref, values) / len(ref)).astype("float32")


def add_state_features(df):
    df = df.copy()
    lap = pd.to_numeric(df.get("LapNumber", 0), errors="coerce").fillna(0).astype("float32")
    tyre = pd.to_numeric(df.get("TyreLife", 0), errors="coerce").fillna(0).astype("float32")
    stint = pd.to_numeric(df.get("Stint", 0), errors="coerce").fillna(0).astype("float32")
    pit = pd.to_numeric(df.get("PitStop", 0), errors="coerce").fillna(0).astype("float32")
    pos = pd.to_numeric(df.get("Position", 0), errors="coerce").fillna(0).astype("float32")
    pchg = pd.to_numeric(df.get("Position_Change", 0), errors="coerce").fillna(0).astype("float32")
    lapt = pd.to_numeric(df.get("LapTime (s)", 0), errors="coerce").fillna(0).astype("float32")
    delta = pd.to_numeric(df.get("LapTime_Delta", 0), errors="coerce").fillna(0).astype("float32")
    degr = pd.to_numeric(df.get("Cumulative_Degradation", 0), errors="coerce").fillna(0).astype("float32")
    prog = pd.to_numeric(df.get("RaceProgress", 0), errors="coerce").fillna(0).astype("float32")
    df["state_tyre_lap_ratio"] = (tyre / (lap + 1)).astype("float32")
    df["state_lap_per_stint"] = (lap / (stint + 1)).astype("float32")
    df["state_tyre_after_pit"] = (tyre / (pit + 1)).astype("float32")
    df["state_degr_per_tyre"] = (degr / (tyre + 1)).astype("float32")
    df["state_delta_per_degr"] = (delta / (np.abs(degr) + 1)).astype("float32")
    df["state_laptime_per_tyre"] = (lapt / (tyre + 1)).astype("float32")
    df["state_laptime_x_progress"] = (lapt * prog).astype("float32")
    df["state_degr_x_progress"] = (degr * prog).astype("float32")
    df["state_position_risk"] = ((21 - pos).clip(lower=0) + np.abs(pchg)).astype("float32")
    df["state_late_stint"] = ((prog > 0.55).astype("int8") * (tyre > 12).astype("int8")).astype("int8")
    return df


def add_original_group_features(train, test, orig, cols, target):
    new_cols = []
    global_mean = float(orig[target].mean())
    for c in cols:
        if c not in orig.columns or c not in train.columns or c not in test.columns:
            continue
        p = safe_name(c)
        mean_name = f"orig_mean_{p}"
        cnt_name = f"orig_count_{p}"
        mean_map = orig.groupby(c, observed=False)[target].mean()
        cnt_map = orig[c].value_counts(dropna=False)
        for df in (train, test):
            df[mean_name] = df[c].map(mean_map).fillna(global_mean).astype("float32")
            df[cnt_name] = df[c].map(cnt_map).fillna(0).astype("float32")
        new_cols += [mean_name, cnt_name]
    return new_cols


def add_frequency_rank_features(train, test, orig, numeric_cols):
    new_cols = []
    for c in numeric_cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        p = safe_name(c)
        freq = pd.concat([train[c], test[c], orig[c]], ignore_index=True).value_counts(normalize=True, dropna=False)
        for df in (train, test):
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
            df[f"freq_{p}"] = df[c].map(freq).fillna(0).astype("float32")
            df[f"rank_orig_{p}"] = pctrank_against(v, orig[c])
            df[f"round2_{p}"] = v.round(2).astype("float32")
            df[f"frac3_{p}"] = np.floor((np.abs(v.astype("float64")) * 1000) % 1000).astype("float32")
        new_cols += [f"freq_{p}", f"rank_orig_{p}", f"round2_{p}", f"frac3_{p}"]
    return new_cols


def add_nn_original_features(train, test, orig, numeric_cols, target):
    from scipy.spatial import cKDTree
    from sklearn.preprocessing import StandardScaler

    train = train.copy()
    test = test.copy()
    orig = orig.copy()
    use_cols = [c for c in numeric_cols if c in train.columns and c in test.columns and c in orig.columns]
    print(f"Nearest-original numeric columns ({len(use_cols)}): {use_cols}", flush=True)
    tr_num = train[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    te_num = test[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    or_num = orig[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    scaler = StandardScaler()
    orig_scaled = scaler.fit_transform(or_num)
    train_scaled = scaler.transform(tr_num)
    test_scaled = scaler.transform(te_num)
    tree = cKDTree(orig_scaled)
    y_orig = orig[target].values.astype("float32")
    for k in [1, 3, 5, 10]:
        dist_tr, idx_tr = tree.query(train_scaled, k=k)
        dist_te, idx_te = tree.query(test_scaled, k=k)
        if k == 1:
            idx_tr = idx_tr.reshape(-1, 1)
            idx_te = idx_te.reshape(-1, 1)
            dist_tr = dist_tr.reshape(-1, 1)
            dist_te = dist_te.reshape(-1, 1)
        train[f"NN{k}_orig_target_mean"] = y_orig[idx_tr].mean(axis=1).astype("float32")
        test[f"NN{k}_orig_target_mean"] = y_orig[idx_te].mean(axis=1).astype("float32")
        train[f"NN{k}_orig_target_std"] = y_orig[idx_tr].std(axis=1).astype("float32")
        test[f"NN{k}_orig_target_std"] = y_orig[idx_te].std(axis=1).astype("float32")
        train[f"NN{k}_distance_mean"] = np.asarray(dist_tr).mean(axis=1).astype("float32")
        test[f"NN{k}_distance_mean"] = np.asarray(dist_te).mean(axis=1).astype("float32")
    return train, test


def ordinal_encode(train, test, orig, cat_cols):
    from sklearn.preprocessing import OrdinalEncoder

    train = train.copy()
    test = test.copy()
    orig = orig.copy()
    use = [c for c in cat_cols if c in train.columns and c in test.columns]
    for df in (train, test, orig):
        for c in use:
            df[c] = sanitize_series(df[c])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    enc.fit(pd.concat([train[use], test[use], orig[[c for c in use if c in orig.columns]]], ignore_index=True).astype(str))
    train[use] = enc.transform(train[use].astype(str)).astype("float32")
    test[use] = enc.transform(test[use].astype(str)).astype("float32")
    return train, test, use


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])
    y = train[TARGET].values.astype("int8")

    cats, nums, numeric_like = infer_groups(train, test)
    print(f"CATS={cats}\nNUMS={nums}", flush=True)
    for df in (train, test, orig):
        for c in cats:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
        for c in numeric_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    train = add_state_features(train)
    test = add_state_features(test)
    orig = add_state_features(orig)
    state_cols = [c for c in train.columns if c.startswith("state_")]

    # Nearest original rows over the race-state numerics.
    knn_cols = [c for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "TyreLife", "LapNumber", "Stint", "PitStop", "Position", "Position_Change"] if c in train.columns]
    knn_cols += [c for c in state_cols if c in train.columns]
    train, test = add_nn_original_features(train, test, orig, knn_cols, TARGET)

    orig_group_cols = [c for c in cats + ["LapNumber", "TyreLife", "PitStop", "Stint", "Position", "Position_Change"] if c in train.columns]
    add_original_group_features(train, test, orig, orig_group_cols, TARGET)
    add_frequency_rank_features(train, test, orig, nums)

    crosses = [("Driver", "Race"), ("Driver", "Compound"), ("Race", "Compound"), ("Year", "Race"), ("Stint", "Compound"), ("PitStop", "Stint")]
    cross_cols = []
    for a, b in crosses:
        if a in train.columns and b in train.columns:
            n = f"cross_{safe_name(a)}__{safe_name(b)}"
            for df in (train, test, orig):
                df[n] = sanitize_series(df[a]) + "|" + sanitize_series(df[b])
            cross_cols.append(n)
    add_original_group_features(train, test, orig, cross_cols, TARGET)

    feature_cols = [c for c in train.columns if c not in [ID, TARGET]]
    cat_cols = [c for c in feature_cols if train[c].dtype == "object" or str(train[c].dtype).startswith("string")]
    train_enc, test_enc, _ = ordinal_encode(train[feature_cols].copy(), test[feature_cols].copy(),
                                            orig.drop(columns=[TARGET, ID], errors="ignore").copy(), cat_cols)

    X = train_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_enc.reindex(columns=X.columns, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0).astype("float32")
    print(f"Training matrix: {X.shape}", flush=True)
    return X, X_test, y


def main():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        X, X_test, y = build_features()
        f = folds(y)
        for k in todo:
            tr_idx, va_idx = np.flatnonzero(f != k), np.flatnonzero(f == k)
            model = xgb.XGBClassifier(**{**XGB_PARAMS, "random_state": SEED + k + 1})
            model.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=200)
            p_va = model.predict_proba(X.iloc[va_idx])[:, 1].astype("float32")
            p_te = model.predict_proba(X_test)[:, 1].astype("float32")
            save_part(HERE, PAIR, k, np.clip(p_va, 1e-7, 1 - 1e-7), np.clip(p_te, 1e-7, 1 - 1e-7))
            print(f"{PAIR} fold {k}: AUC {roc_auc_score(y[va_idx], p_va):.6f}", flush=True)
    y = read_train()[TARGET].values.astype("int8")
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
