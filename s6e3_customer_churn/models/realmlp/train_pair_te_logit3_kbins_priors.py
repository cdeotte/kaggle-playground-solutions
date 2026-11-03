#!/usr/bin/env python
"""RealMLP on value codes, quantile bins, original-data priors / distribution features and per-fold smoothed target
encodings of all 105 pairs of the 15 string categoricals as logit, logit^2, logit^3 (pair
realmlp_pair_te_logit3_kbins_priors).

Training rows = competition rows only; the original data (7,043 rows, blank TotalCharges -> MonthlyCharges x tenure)
feeds the priors. Features, fitted once on train + test + original:
- service count, has_internet, has_phone; MonthlyCharges / TotalCharges, TotalCharges / tenure,
  TotalCharges / MonthlyCharges, MonthlyCharges / (TotalCharges / tenure), tenure^2, TotalCharges - tenure x
  MonthlyCharges; four `_FREQ_*` columns as in the public notebook (whole-row DataFrame.value_counts mapped onto single
  columns, which gives 0 everywhere);
- categories: (TotalCharges / 1000) mod 10; KBinsDiscretizer quantile bins of MonthlyCharges (200), TotalCharges
  (4000) and tenure (25); the factorized values of SeniorCitizen, tenure, MonthlyCharges and of TotalCharges rounded
  to an integer;
- original churn rate of every raw value (0.5 if unseen); TotalCharges against the original data (percentile ranks
  among all / churned / retained customers and within the row's InternetService and Contract, z-scores, gaps, distances
  to the original quartiles of both classes); MonthlyCharges minus the original mean of its InternetService.
Per fold (the 5 folds of common/data.py): for every pair of the 15 string categoricals, the label mean of the pair
key smoothed toward the fold mean (m = 20), out-of-fold over StratifiedKFold(5, shuffle=True, random_state=42 + fold)
for the training rows and from all training rows for the validation / test rows; clipped logit z, z^2, z^3,
standardized on the training rows (315 columns).
Model: pytabkit RealMLP_TD_Classifier (4 x 352 mish, n_ens 8, lr 0.075, batch 256, up to 10 epochs with early
stopping on the validation fold's AUC, random_state 42). Test = mean of the 5 fold models.
Usage: python train_pair_te_logit3_kbins_priors.py"""
import gc
import os
import random
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, N_FOLDS, TARGET, folds, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

FOLDER = "realmlp"
PAIR = "realmlp_pair_te_logit3_kbins_priors"
SEED = 42
TE_INNER_SPLITS = 5
USE_LOGIT3 = True
TE_SMOOTH = 20.0

REALMLP_PARAMS = {
    "device": "cuda",
    "random_state": 42,
    "verbosity": 2,
    "n_epochs": 10,
    "batch_size": 256,
    "n_ens": 8,
    "val_metric_name": "1-auc_ovr",
    "use_early_stopping": True,
    "early_stopping_additive_patience": 20,
    "early_stopping_multiplicative_patience": 1,
    "act": "mish",
    "embedding_size": 6,
    "first_layer_lr_factor": 0.25,
    "hidden_sizes": "rectangular",
    "hidden_width": 352,
    "lr": 0.075,
    "ls_eps": 0.01,
    "ls_eps_sched": "coslog4",
    "max_one_hot_cat_size": 18,
    "n_hidden_layers": 4,
    "p_drop": 0.05,
    "p_drop_sched": "flat_cos",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "scale_lr_factor": 2.24,
    "sq_mom": 0.988,
    "wd": 0.0236,
}


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (
        np.zeros(len(values), dtype="float32")
        if sigma == 0
        else ((values - mu) / sigma).astype("float32")
    )


def make_pair_key(df, f1, f2):
    return df[f1].astype(str) + "___" + df[f2].astype(str)


def add_smoothed_mean_encoding(train_keys, train_y, apply_keys, smooth=20.0):
    global_mean = float(np.mean(train_y))
    stats = pd.DataFrame({
        "key": train_keys,
        "y": np.asarray(train_y, dtype=np.float32),
    }).groupby("key")["y"].agg(["mean", "count"])

    enc = (stats["mean"] * stats["count"] + global_mean * smooth) / (stats["count"] + smooth)
    return apply_keys.map(enc).fillna(global_mean).astype(np.float32).values


def build_pair_te_features(Xtr, ytr, Xva, Xte, pair_cols, inner_splits=5, seed=0, smooth=20.0, use_logit3=True):
    """Training rows: inner-fold out-of-fold encodings; validation / test rows: encodings fitted on all of Xtr."""
    n_tr, n_va, n_te, n_pair = len(Xtr), len(Xva), len(Xte), len(pair_cols)

    tr_pair = np.zeros((n_tr, n_pair), dtype=np.float32)
    va_pair = np.zeros((n_va, n_pair), dtype=np.float32)
    te_pair = np.zeros((n_te, n_pair), dtype=np.float32)

    inner_kf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    ytr_np = np.asarray(ytr)

    for j, (f1, f2) in enumerate(pair_cols):
        if j == 0 or (j + 1) % 20 == 0 or (j + 1) == n_pair:
            print(f"  pair {j+1:>4d}/{n_pair}: {f1} x {f2}", flush=True)

        tr_keys_full = make_pair_key(Xtr, f1, f2)
        va_keys = make_pair_key(Xva, f1, f2)
        te_keys = make_pair_key(Xte, f1, f2)

        tr_oof = np.zeros(n_tr, dtype=np.float32)
        for it, iv in inner_kf.split(Xtr, ytr_np):
            tr_oof[iv] = add_smoothed_mean_encoding(tr_keys_full.iloc[it], ytr_np[it], tr_keys_full.iloc[iv],
                                                    smooth=smooth)
        tr_pair[:, j] = tr_oof
        va_pair[:, j] = add_smoothed_mean_encoding(tr_keys_full, ytr_np, va_keys, smooth=smooth)
        te_pair[:, j] = add_smoothed_mean_encoding(tr_keys_full, ytr_np, te_keys, smooth=smooth)

    if use_logit3:
        eps = 1e-5

        def logit3(m):
            m = np.clip(m, eps, 1.0 - eps)
            z = np.log(m / (1.0 - m)).astype(np.float32)
            return np.hstack([z, z**2, z**3]).astype(np.float32)

        tr_pair = logit3(tr_pair)
        va_pair = logit3(va_pair)
        te_pair = logit3(te_pair)

    return tr_pair, va_pair, te_pair


def feature_engineering(df, orig, cat_cols, num_cols):
    """Fitted on (and applied to) all rows of df. Returns df, the new category columns (names ending in '_') and the
    new numeric columns (names starting with '_')."""
    df = df.copy()

    SERVICE_COLS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["_service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
    df["_has_internet"] = (df["InternetService"] != "No").astype("float32")
    df["_has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    for col in num_cols:
        freq = df.value_counts(normalize=True)
        df[f"_FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

    # arithmetic interactions
    df["_charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
    df["_MonthlyCharges_DIV_TotalCharges"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)).astype("float32")
    df["_TotalCharges_DIV_tenure"] = (df["TotalCharges"] / (df["tenure"] + 1e-6)).astype("float32")
    df["_TotalCharges_DIV_MonthlyCharges"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1e-6)).astype("float32")
    df["_Monthly_to_avg_ratio"] = (df["MonthlyCharges"] / (df["_TotalCharges_DIV_tenure"] + 1e-6)).astype("float32")
    df["_tenure_sq"] = (df["tenure"] ** 2).astype("float32")

    # digit extraction
    for col in ["TotalCharges"]:
        k = -3
        digit_name = f"{col}_d{k}_"
        df[digit_name] = ((df[col] * 10**k) % 10).astype("int8")
        df[digit_name] = df[digit_name].astype("category")

    # quantile bins
    bin_config = {"MonthlyCharges": 200, "TotalCharges": 4000, "tenure": 25}
    for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
        bin_name = f"{col}_bin_"
        kb = KBinsDiscretizer(n_bins=bin_config[col], encode="ordinal", strategy="quantile", subsample=None)
        df[bin_name] = kb.fit_transform(df[[col]]).ravel().astype("int32")
        df[bin_name] = df[bin_name].astype("category")

    # original churn rates
    for col in cat_cols + num_cols:
        tmp = orig.groupby(col)[TARGET].mean()
        _name = f"_ORIG_proba_{col}"
        df = df.merge(tmp.rename(_name), on=col, how="left")
        df[_name] = df[_name].fillna(0.5).astype("float32")

    # TotalCharges against the original distributions
    orig_churner_tc = orig.loc[orig[TARGET] == 1, "TotalCharges"].values
    orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, "TotalCharges"].values
    orig_tc = orig["TotalCharges"].values
    orig_is_mc_mean = orig.groupby("InternetService")["MonthlyCharges"].mean()

    tc = df["TotalCharges"].values
    df["_pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nonchurner_tc)
    df["_pctrank_churner_TC"] = pctrank_against(tc, orig_churner_tc)
    df["_pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
    df["_zscore_churn_gap_TC"] = (
        np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))
    ).astype("float32")
    df["_zscore_nonchurner_TC"] = zscore_against(tc, orig_nonchurner_tc)
    df["_pctrank_churn_gap_TC"] = (
        pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)
    ).astype("float32")
    df["_resid_IS_MC"] = (
        df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)
    ).astype("float32")

    vals = np.zeros(len(df), dtype="float32")
    for cat_val in orig["InternetService"].unique():
        mask = df["InternetService"] == cat_val
        ref = orig.loc[orig["InternetService"] == cat_val, "TotalCharges"].values
        if len(ref) > 0 and mask.sum() > 0:
            vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
    df["_cond_pctrank_IS_TC"] = vals

    vals = np.zeros(len(df), dtype="float32")
    for cat_val in orig["Contract"].unique():
        mask = df["Contract"] == cat_val
        ref = orig.loc[orig["Contract"] == cat_val, "TotalCharges"].values
        if len(ref) > 0 and mask.sum() > 0:
            vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
    df["_cond_pctrank_C_TC"] = vals

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        df[f"_dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
        df[f"_dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
        df[f"_qdist_gap_To_{q_label}"] = (df[f"_dist_To_nc_{q_label}"] - df[f"_dist_To_ch_{q_label}"]).astype("float32")

    # numerics as value codes (TotalCharges rounded to an integer first)
    for col in num_cols:
        cat_name = f"{col}_cat_"
        series = df[col].round(0) if col == "TotalCharges" else df[col]
        codes, _ = series.factorize()
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")

    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols


def main():
    seed_everything(SEED)

    train = read_train()
    test = read_test()
    orig = read_original()
    _tc = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    miss = _tc.isnull()
    _tc.loc[miss] = orig.loc[miss, "MonthlyCharges"] * orig.loc[miss, "tenure"]
    orig["TotalCharges"] = _tc
    print("train:", train.shape, "test:", test.shape, "orig:", orig.shape)

    train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
    orig[TARGET] = orig[TARGET].map({"Yes": 1, "No": 0})

    X = train.drop([ID, TARGET], axis=1)
    y = train[TARGET].copy()
    X_test = test.drop([ID], axis=1)
    X_orig = orig.drop([TARGET], axis=1)

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    raw_cat_cols = list(cat_cols)
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    df_all = pd.concat([X, X_test, X_orig], axis=0).reset_index(drop=True)
    df_tf, new_cat_cols, new_num_cols = feature_engineering(df_all, orig, cat_cols, num_cols)

    X = df_tf.iloc[:len(train)].copy()
    X_test = df_tf.iloc[len(train):len(train) + len(test)].copy()

    cat_cols = cat_cols + new_cat_cols
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
        X_test[c] = X_test[c].astype(str).astype("category")

    cat_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        X_test[c] = X_test[c].astype("string").fillna("__MISSING__")
    for c in num_cols:
        X[c] = X[c].astype("float32")
        X_test[c] = X_test[c].astype("float32")

    pair_cols = list(combinations(raw_cat_cols, 2))
    print(f"cat cols: {len(cat_cols)}  num cols: {len(num_cols)}  TE pairs: {len(pair_cols)}")

    base_pair_names = [f"te__{f1}__{f2}" for (f1, f2) in pair_cols]
    if USE_LOGIT3:
        te_col_names = ([f"{c}__z1" for c in base_pair_names] + [f"{c}__z2" for c in base_pair_names]
                        + [f"{c}__z3" for c in base_pair_names])
    else:
        te_col_names = base_pair_names

    fold_id = folds(y.values)
    oof = np.zeros(len(X), dtype=np.float32)
    pred = np.zeros(len(X_test), dtype=np.float32)

    for k in range(N_FOLDS):
        fold = k + 1
        t, v = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        print(f"fold {fold}", flush=True)

        Xt = X.iloc[t].copy().reset_index(drop=True)
        Xv = X.iloc[v].copy().reset_index(drop=True)
        yt = y.iloc[t].copy().reset_index(drop=True)
        yv = y.iloc[v].copy().reset_index(drop=True)
        X_test_fold = X_test.copy().reset_index(drop=True)

        tr_te, va_te, te_te = build_pair_te_features(
            Xt, yt.values, Xv, X_test_fold, pair_cols=pair_cols, inner_splits=TE_INNER_SPLITS, seed=SEED + fold,
            smooth=TE_SMOOTH, use_logit3=USE_LOGIT3)

        scaler_te = StandardScaler()
        tr_te = scaler_te.fit_transform(tr_te).astype(np.float32)
        va_te = scaler_te.transform(va_te).astype(np.float32)
        te_te = scaler_te.transform(te_te).astype(np.float32)

        Xt_model = pd.concat([Xt.reset_index(drop=True), pd.DataFrame(tr_te, columns=te_col_names)], axis=1)
        Xv_model = pd.concat([Xv.reset_index(drop=True), pd.DataFrame(va_te, columns=te_col_names)], axis=1)
        Xte_model = pd.concat([X_test_fold.reset_index(drop=True), pd.DataFrame(te_te, columns=te_col_names)], axis=1)

        for c in cat_cols:
            Xt_model[c] = Xt_model[c].astype("string")
            Xv_model[c] = Xv_model[c].astype("string")
            Xte_model[c] = Xte_model[c].astype("string")
        for c in te_col_names:
            Xt_model[c] = Xt_model[c].astype("float32")
            Xv_model[c] = Xv_model[c].astype("float32")
            Xte_model[c] = Xte_model[c].astype("float32")

        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(Xt_model, yt, Xv_model, yv, cat_col_names=cat_cols)

        val_preds = model.predict_proba(Xv_model)[:, 1]
        test_preds = model.predict_proba(Xte_model)[:, 1]
        oof[v] = val_preds
        pred += test_preds / N_FOLDS
        print(f"fold {fold} auc: {roc_auc_score(yv, val_preds):.6f}", flush=True)

        del model, Xt_model, Xv_model, Xte_model, tr_te, va_te, te_te
        gc.collect()
        torch.cuda.empty_cache()

    print("OOF AUC: %.6f" % roc_auc_score(y, oof))
    save_pair(model_dir(FOLDER), PAIR, oof, pred)


if __name__ == "__main__":
    main()
