#!/usr/bin/env python
"""LightGBM with nearest-original-row target features (pair lgbm_original_knn_target).

Features of the train / test rows (the original rows are only a reference, not training rows):
  - the 14 columns; Driver, Compound, Race and the crosses ordinal-encoded over train + test + original
  - 10 race-state ratios / products (state_*)
  - nearest original rows: the 10 numerics LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, TyreLife,
    LapNumber, Stint, PitStop, Position, Position_Change and the state features, standardized on the original rows; a
    cKDTree on the original rows; for k = 1, 3, 5, 10 the mean and std of the k nearest original targets and their mean
    distance
  - original target mean and row count of every category and of LapNumber, TyreLife, PitStop, Stint, Position,
    Position_Change
  - for LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress: frequency over train + test + original, percentile
    rank among the original values, the value rounded to 2 decimals, the first 3 decimals as an integer
  - 6 crosses (Driver x Race, Driver x Compound, Race x Compound, Year x Race, Stint x Compound, PitStop x Stint) and their
    original target mean and row count
Model: LightGBM, 600 rounds (no early stopping), lr 0.03, 31 leaves, depth 6, min_child_samples 50, subsample 0.8 every
round, colsample_bytree 0.8, alpha 0.1, lambda 1, seed 42, 32 threads. Test = mean of the 5 fold models.
Usage: python train_original_knn.py"""
import gc
import re
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_original_knn_target"
SEED = 42
N_FOLDS = 5
N_THREADS = 32

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    max_depth=6,
    num_leaves=31,
    learning_rate=0.03,
    n_estimators=600,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=50,
    random_state=SEED,
    verbosity=-1,
    n_jobs=N_THREADS,
    num_threads=N_THREADS,
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
    """Categoricals: string columns and numerics with < 100 values in train + test; numerics: the rest."""
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
    """Target mean / std and mean distance of the k nearest original rows (standardized on the original rows)."""
    train = train.copy()
    test = test.copy()
    orig = orig.copy()
    use_cols = [c for c in numeric_cols if c in train.columns and c in test.columns and c in orig.columns]
    print(f"nearest-original columns ({len(use_cols)}): {use_cols}", flush=True)
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
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])
    y = train[TARGET].values.astype("int8")

    cats, nums, numeric_like = infer_groups(train, test)
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
    train_enc, test_enc, _ = ordinal_encode(train[feature_cols].copy(), test[feature_cols].copy(), orig.drop(columns=[TARGET, ID], errors="ignore").copy(), cat_cols)

    X = train_enc.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_enc.reindex(columns=X.columns, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(0).astype("float32")
    print(f"training matrix {X.shape}", flush=True)
    return X, X_test, y


def main():
    X, X_test, y = build_features()
    oof = np.zeros(len(X), dtype="float32")
    test_pred = np.zeros(len(X_test), dtype="float32")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx].copy()
        X_va, y_va = X.iloc[va_idx].copy(), y[va_idx].copy()
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.log_evaluation(100)])
        p_va = model.predict_proba(X_va)[:, 1].astype("float32")
        p_te = model.predict_proba(X_test)[:, 1].astype("float32")
        oof[va_idx] = p_va
        test_pred += p_te / N_FOLDS
        print(f"fold {fold - 1}: AUC {roc_auc_score(y_va, p_va):.6f}", flush=True)
        del X_tr, X_va, y_tr, y_va, model, p_va, p_te
        gc.collect()

    oof = np.clip(oof, 1e-7, 1 - 1e-7)
    test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == "__main__":
    main()
