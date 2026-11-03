#!/usr/bin/env python
"""XGBoost on KNN target features and logistic-regression meta features (pair xgb_knn_stack_lr_meta).

Base matrix (64 columns, built once on train + test + original): the 11 numeric columns with log1p |x|, 2-decimal
rounding and the 3 digits after the decimal point, 10 state ratios (tyre / lap, lap / stint, degradation / tyre, ...),
and integer codes of the 10 categoricals (strings + numerics with < 100 values).
Static features: original-data priors (target mean and count of every categorical in the original data) and original
KNN priors: cKDTree neighbours of every train / test row among the original rows (standardized on the original rows),
mean original target of the k = 1, 3, 5, 10, 20 nearest neighbours plus mean / min distance, under 5 column-weight
schemes (uniform, artifact, state, cat, numeric).
Per outer fold:
- train KNN features: neighbours among the outer-training rows (standardized on them), k = 5, 20, 40, schemes uniform /
  artifact / cat; the outer-training rows get inner-out-of-fold values (5 inner folds, seed 42 + fold);
- logistic-regression meta features (C 0.2, on the logits of the KNN features, fitted on the outer-training rows):
  one on the original-KNN priors, one on the train-KNN features, their average, and the mean of all train-KNN features;
- XGBoost hist on GPU, lr 0.018, depth 4, min_child_weight 8, subsample 0.82, colsample_bytree 0.32, alpha 1,
  lambda 8, up to 6,000 rounds with early stopping 250 on the fold's validation rows (AUC), seed 42; 138 columns.
Probabilities clipped to [1e-7, 1 - 1e-7]. Test = mean of the 5 fold models. The original rows are never training rows.
Usage: python train_knn_stack_xgb.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import gc
import os
import re
import sys
from pathlib import Path

for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[var] = '8'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = 'xgb_knn_stack_lr_meta'
FOLDER = 'classic'
SEED = 42
INNER_SPLITS = 5
EPS = 1e-6

ORIG_KS = [1, 3, 5, 10, 20]
TRAIN_KS = [5, 20, 40]
QUERY_CHUNK = 50000

LR_PARAMS = dict(C=0.20, solver="lbfgs", max_iter=1000, random_state=SEED)
XGB_PARAMS = dict(
    n_estimators=6000,
    learning_rate=0.018,
    max_depth=4,
    min_child_weight=8,
    subsample=0.82,
    colsample_bytree=0.32,
    reg_alpha=1.0,
    reg_lambda=8.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    device="cuda",
    enable_categorical=False,
    random_state=SEED,
    early_stopping_rounds=250,
    verbosity=0,
)


def safe_name(col):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "feature"


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


def safe_logit(x, eps=EPS):
    x = np.clip(np.asarray(x, dtype="float32"), eps, 1 - eps)
    return np.log(x / (1 - x)).astype("float32")


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
    df["state_late_old_tyre"] = ((prog > 0.50).astype("float32") * np.log1p(tyre.clip(lower=0))).astype("float32")
    return df


def fit_category_maps(train, test, orig, cats):
    maps = {}
    for c in cats:
        vals = [sanitize_series(df[c]) for df in (train, test, orig) if c in df.columns]
        if vals:
            uniques = pd.Index(pd.concat(vals, ignore_index=True).unique())
            maps[c] = {v: i for i, v in enumerate(uniques)}
    return maps


def apply_category_maps(df, cats, maps):
    out = pd.DataFrame(index=df.index)
    for c in cats:
        if c in df.columns:
            out[f"cat_{safe_name(c)}"] = sanitize_series(df[c]).map(maps[c]).fillna(-1).astype("int32")
    return out


def build_numeric_frame(df, numeric_like, cats, cat_maps):
    num = pd.DataFrame(index=df.index)
    for c in numeric_like:
        if c in df.columns:
            p = safe_name(c)
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
            num[p] = v
            num[f"log1p_{p}"] = np.log1p(np.abs(v)).astype("float32")
            num[f"round2_{p}"] = v.round(2).astype("float32")
            num[f"frac3_{p}"] = np.floor((np.abs(v.astype("float64")) * 1000) % 1000).astype("float32")
    for c in [c for c in df.columns if c.startswith("state_")]:
        num[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    parts = [num, apply_category_maps(df, cats, cat_maps).astype("float32")]
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")


def get_weight_schemes(cols):
    """Per-column weights of the KNN distance (column values are scaled by sqrt(weight))."""
    cols = list(cols)
    schemes = {"uniform": np.ones(len(cols), dtype="float32")}
    artifact_tokens = ["LapTime", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress", "frac3", "round2"]
    state_tokens = ["state_", "TyreLife", "LapNumber", "Stint", "PitStop", "Position"]
    cat_tokens = ["cat_"]
    for name, tokens, boost in [
        ("artifact", artifact_tokens, 2.5),
        ("state", state_tokens, 2.0),
        ("cat", cat_tokens, 2.2),
        ("numeric", [], 1.0),
    ]:
        w = np.ones(len(cols), dtype="float32")
        if name == "numeric":
            for i, c in enumerate(cols):
                if c.startswith("cat_"):
                    w[i] = 0.35
                elif "frac3" in c or "round2" in c:
                    w[i] = 1.3
                else:
                    w[i] = 1.6
        else:
            for i, c in enumerate(cols):
                if any(tok in c for tok in tokens):
                    w[i] *= boost
                elif c.startswith("cat_") and name != "cat":
                    w[i] *= 0.55
        schemes[name] = w / np.mean(w)
    return schemes


def query_knn_features(ref_mat, ref_y, query_mat, schemes, ks, prefix):
    out = {}
    ref_y = np.asarray(ref_y, dtype="float32")
    max_k = max(ks)
    for scheme_name, weights in schemes.items():
        scale = np.sqrt(weights).astype("float32")
        ref_scaled = np.asarray(ref_mat, dtype="float32") * scale[None, :]
        query_scaled = np.asarray(query_mat, dtype="float32") * scale[None, :]
        tree = cKDTree(ref_scaled)
        means = {k: np.empty(len(query_scaled), dtype="float32") for k in ks}
        dist_mean = np.empty(len(query_scaled), dtype="float32")
        dist_min = np.empty(len(query_scaled), dtype="float32")
        for lo in range(0, len(query_scaled), QUERY_CHUNK):
            hi = min(lo + QUERY_CHUNK, len(query_scaled))
            dist, idx = tree.query(query_scaled[lo:hi], k=max_k, workers=-1)
            if max_k == 1:
                dist = dist.reshape(-1, 1)
                idx = idx.reshape(-1, 1)
            labels = ref_y[idx]
            for k in ks:
                means[k][lo:hi] = labels[:, :k].mean(axis=1).astype("float32")
            dist_mean[lo:hi] = np.asarray(dist).mean(axis=1).astype("float32")
            dist_min[lo:hi] = np.asarray(dist)[:, 0].astype("float32")
        for k in ks:
            out[f"{prefix}_{scheme_name}_k{k}_mean"] = means[k]
        out[f"{prefix}_{scheme_name}_dist_mean"] = dist_mean
        out[f"{prefix}_{scheme_name}_dist_min"] = dist_min
        del tree, ref_scaled, query_scaled
        gc.collect()
    return pd.DataFrame(out)


def build_lr_feature(X_tr, y_tr, X_va, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_tr_s, y_tr)
    return lr.predict_proba(X_va_s)[:, 1].astype("float32"), lr.predict_proba(X_te_s)[:, 1].astype("float32")


def add_original_group_priors(train, test, orig, cats, target):
    tr = pd.DataFrame(index=train.index)
    te = pd.DataFrame(index=test.index)
    global_mean = float(orig[target].mean())
    keys = [c for c in cats + ["LapNumber", "TyreLife", "PitStop", "Stint", "Position", "Position_Change"]
            if c in train.columns and c in orig.columns]
    for c in keys:
        p = safe_name(c)
        mean_map = orig.groupby(c, observed=False)[target].mean()
        cnt_map = orig[c].value_counts(dropna=False)
        tr[f"orig_mean_{p}"] = train[c].map(mean_map).fillna(global_mean).astype("float32")
        te[f"orig_mean_{p}"] = test[c].map(mean_map).fillna(global_mean).astype("float32")
        tr[f"orig_count_{p}"] = train[c].map(cnt_map).fillna(0).astype("float32")
        te[f"orig_count_{p}"] = test[c].map(cnt_map).fillna(0).astype("float32")
    return tr, te


def build_static():
    """Base matrix, original priors and original-KNN priors of every train / test row (no train labels used)."""
    train, test, orig = read_train(), read_test(), read_original()
    train[TARGET] = train[TARGET].astype("int8")
    orig[TARGET] = orig[TARGET].astype("int8")
    y = train[TARGET].values.astype("int8")
    y_orig = orig[TARGET].values.astype("int8")

    cats, nums, numeric_like = infer_groups(train, test)
    print(f"categoricals {cats}\nnumerics {nums}", flush=True)
    for df in (train, test, orig):
        for c in cats:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
        for c in numeric_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    train = add_state_features(train)
    test = add_state_features(test)
    orig = add_state_features(orig)
    cat_maps = fit_category_maps(train, test, orig, cats)
    X_base_train = build_numeric_frame(train, numeric_like, cats, cat_maps)
    X_base_test = build_numeric_frame(test, numeric_like, cats, cat_maps)
    X_base_orig = build_numeric_frame(orig, numeric_like, cats, cat_maps)
    all_cols = sorted(set(X_base_train.columns) | set(X_base_test.columns) | set(X_base_orig.columns))
    X_base_train = X_base_train.reindex(columns=all_cols, fill_value=0).astype("float32")
    X_base_test = X_base_test.reindex(columns=all_cols, fill_value=0).astype("float32")
    X_base_orig = X_base_orig.reindex(columns=all_cols, fill_value=0).astype("float32")
    print(f"base matrix: {len(all_cols)} columns", flush=True)

    orig_prior_train, orig_prior_test = add_original_group_priors(train, test, orig, cats, TARGET)

    # original-reference KNN, standardized on the original rows (labels come only from the original data)
    scaler_orig = StandardScaler()
    orig_ref = scaler_orig.fit_transform(X_base_orig.values).astype("float32")
    train_query_orig = scaler_orig.transform(X_base_train.values).astype("float32")
    test_query_orig = scaler_orig.transform(X_base_test.values).astype("float32")
    schemes = get_weight_schemes(all_cols)
    orig_knn_train = query_knn_features(orig_ref, y_orig, train_query_orig, schemes, ORIG_KS, "origknn")
    orig_knn_test = query_knn_features(orig_ref, y_orig, test_query_orig, schemes, ORIG_KS, "origknn")
    orig_logit_cols = [f"logit_{c}" for c in orig_knn_train.columns]
    orig_knn_train_logit = pd.DataFrame(safe_logit(orig_knn_train.values), columns=orig_logit_cols)
    orig_knn_test_logit = pd.DataFrame(safe_logit(orig_knn_test.values), columns=orig_logit_cols)

    base_train = pd.concat([X_base_train.reset_index(drop=True), orig_prior_train.reset_index(drop=True),
                            orig_knn_train.reset_index(drop=True)], axis=1)
    base_test = pd.concat([X_base_test.reset_index(drop=True), orig_prior_test.reset_index(drop=True),
                           orig_knn_test.reset_index(drop=True)], axis=1)
    return dict(y=y, X_base_train=X_base_train, X_base_test=X_base_test, schemes=schemes, base_train=base_train,
                base_test=base_test, orig_knn_train_logit=orig_knn_train_logit, orig_knn_test_logit=orig_knn_test_logit)


def run_fold(s, k, tr_idx, va_idx):
    fold = k + 1
    y = s["y"]
    X_base_train, X_base_test, schemes = s["X_base_train"], s["X_base_test"], s["schemes"]
    orig_knn_train_logit, orig_knn_test_logit = s["orig_knn_train_logit"], s["orig_knn_test_logit"]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # original-KNN LR meta feature: fitted on the outer-training labels only
    p_va_orig_lr, p_te_orig_lr = build_lr_feature(orig_knn_train_logit.iloc[tr_idx].values, y_tr,
                                                  orig_knn_train_logit.iloc[va_idx].values, orig_knn_test_logit.values)

    # train-KNN features of the validation / test rows: the outer-training rows are the reference
    scaler_fold = StandardScaler()
    ref_mat = scaler_fold.fit_transform(X_base_train.iloc[tr_idx].values).astype("float32")
    va_mat = scaler_fold.transform(X_base_train.iloc[va_idx].values).astype("float32")
    te_mat = scaler_fold.transform(X_base_test.values).astype("float32")
    train_schemes = {name: schemes[name] for name in ["uniform", "artifact", "cat"] if name in schemes}
    trknn_va = query_knn_features(ref_mat, y_tr, va_mat, train_schemes, TRAIN_KS, "trknn")
    trknn_te = query_knn_features(ref_mat, y_tr, te_mat, train_schemes, TRAIN_KS, "trknn")

    # inner-out-of-fold train-KNN features of the outer-training rows
    trknn_oof = pd.DataFrame(0.0, index=np.arange(len(tr_idx)), columns=trknn_va.columns, dtype="float32")
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + fold)
    X_outer = X_base_train.iloc[tr_idx].reset_index(drop=True)
    for inner_fold, (itr, iva) in enumerate(inner.split(X_outer, y_tr), start=1):
        print(f"  fold {k}: inner KNN fold {inner_fold}/{INNER_SPLITS}", flush=True)
        sc = StandardScaler()
        ref_inner = sc.fit_transform(X_outer.iloc[itr].values).astype("float32")
        val_inner = sc.transform(X_outer.iloc[iva].values).astype("float32")
        feats_inner = query_knn_features(ref_inner, y_tr[itr], val_inner, train_schemes, TRAIN_KS, "trknn")
        trknn_oof.iloc[iva] = feats_inner.values.astype("float32")
        del sc, ref_inner, val_inner, feats_inner
        gc.collect()

    trknn_oof_logit = safe_logit(trknn_oof.values)
    trknn_va_logit = safe_logit(trknn_va.values)
    trknn_te_logit = safe_logit(trknn_te.values)
    p_va_tr_lr, p_te_tr_lr = build_lr_feature(trknn_oof_logit, y_tr, trknn_va_logit, trknn_te_logit)

    # LR meta features of the outer-training rows (fitted and applied on the same rows; KNN inputs are inner-OOF)
    p_tr_tr_lr, _ = build_lr_feature(trknn_oof_logit, y_tr, trknn_oof_logit, trknn_te_logit)
    p_tr_orig_lr, _ = build_lr_feature(orig_knn_train_logit.iloc[tr_idx].values, y_tr,
                                       orig_knn_train_logit.iloc[tr_idx].values, orig_knn_test_logit.values)

    def stack_block(knn, p_orig, p_tr):
        meta = pd.DataFrame({
            "meta_orig_lr": p_orig,
            "meta_trknn_lr": p_tr,
            "meta_full_lr": (0.5 * p_orig + 0.5 * p_tr).astype("float32"),
            "trknn_mean_all": knn.mean(axis=1).astype("float32").values,
        })
        return pd.concat([knn.reset_index(drop=True), meta], axis=1)

    train_stack = stack_block(trknn_oof, p_tr_orig_lr, p_tr_tr_lr)
    val_stack = stack_block(trknn_va, p_va_orig_lr, p_va_tr_lr)
    test_stack = stack_block(trknn_te, p_te_orig_lr, p_te_tr_lr)

    X_tr = pd.concat([s["base_train"].iloc[tr_idx].reset_index(drop=True), train_stack.reset_index(drop=True)], axis=1)
    X_va = pd.concat([s["base_train"].iloc[va_idx].reset_index(drop=True), val_stack.reset_index(drop=True)], axis=1)
    X_te = pd.concat([s["base_test"].reset_index(drop=True), test_stack.reset_index(drop=True)], axis=1)
    X_tr = X_tr.replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0).replace([np.inf, -np.inf], 0).fillna(0).astype("float32")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
    p_va = np.clip(model.predict_proba(X_va)[:, 1].astype("float32"), 1e-7, 1 - 1e-7)
    p_te = np.clip(model.predict_proba(X_te)[:, 1].astype("float32"), 1e-7, 1 - 1e-7)
    print(f"{PAIR} fold {k}: {X_tr.shape[1]} columns, AUC {roc_auc_score(y_va, p_va):.6f}", flush=True)
    return p_va, p_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    out = model_dir(FOLDER)
    s = build_static()
    fold_id = folds(s["y"])
    for k in fold_list(args.folds):
        if have_part(out, PAIR, k):
            print(f'{PAIR} fold {k}: done', flush=True)
            continue
        tr_idx, va_idx = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        p_va, p_te = run_fold(s, k, tr_idx, va_idx)
        save_part(out, PAIR, k, p_va, p_te)
        gc.collect()
    assemble(out, PAIR, fold_id, s["y"])


if __name__ == '__main__':
    main()
