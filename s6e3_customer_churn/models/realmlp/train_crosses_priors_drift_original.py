#!/usr/bin/env python
"""RealMLP on frequency / rank / transform / service / cross-category / original-prior / drift features plus the digit /
bin / rounded-category block, original rows appended before the split. Two models (presets, pair name):

  realmlp_3config_crosses_priors_drift_original      the three tuned RealMLP configurations of lib/appended_original.py,
                                                     averaged per fold
  realmlp_risk_score_crosses_priors_drift_original   + a handcrafted churn-risk score and the original churn rate of each
                                                     raw value; one RealMLP configuration (2 x 512, p_drop 0.42, 30 epochs)

Rows: the 594,194 competition rows followed by the 7,032 original rows with a TotalCharges (numerics float32, NaN ->
median). Folds: StratifiedKFold(5, shuffle=True, random_state=42) over these 601,226 rows; a fold trains on the
competition and original rows of the 4 other folds and is validated on all its own rows. The pair keeps the OOF of the
competition rows.
Features, built for the 601,226 rows, the test rows and the original rows (the 19 raw columns plus):
- frequency and percentile rank of tenure / MonthlyCharges / TotalCharges over the three frames; log1p, sqrt,
  1 / (1 + x) and missing flag;
- TotalCharges - tenure x MonthlyCharges (and its absolute value), MonthlyCharges / (TotalCharges + 1),
  TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure x MonthlyCharges, tenure x TotalCharges;
- counts of Yes / No / no-service answers over the 8 service columns, has_internet, has_phone; Yes / No / other flags of
  11 columns;
- tenure bins (0, 1, 3, 6, 12, 24, 36, 48, 60, 72), 40 / 60 quantile bins of MonthlyCharges / TotalCharges, 6
  categorical pairs and Contract x InternetService x PaymentMethod as categories;
- counts and rare flags (<= 50) of the 16 categoricals and these 10 categories over the three frames;
- original churn rates (0.5 if unseen) of the 16 categoricals, the 10 categories and the 3 numeric values;
- drift log1p(train frequency / (original frequency + 1e-6)) of the 26 categories;
- the digit / bin / rounded-category / rare-flag block of lib/appended_original.py over the 601,226 rows + test.
Test = mean of the fold models.
Usage: python train_crosses_priors_drift_original.py <pair>"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, model_dir, read_original, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from lib.appended_original import FEATS, THREE_CONFIGS, add_digit_bin_cat_rare_features, cross_validate  # noqa: E402

warnings.simplefilter("ignore")

FOLDER = "realmlp"
N_SPLITS = 5
SEED = 42

BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
BASE_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
MC_BINS = 40
TC_BINS = 60

CROSS_PAIRS = [
    ("Contract", "InternetService"),
    ("PaymentMethod", "Contract"),
    ("InternetService", "OnlineSecurity"),
    ("PaymentMethod", "PaperlessBilling"),
    ("Contract", "PaperlessBilling"),
    ("InternetService", "TechSupport"),
]
TRIPLE = [("Contract", "InternetService", "PaymentMethod")]

CAT_RARE_THRESH = 50
DRIFT_EPS = 1e-6

ONE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_epochs": 30,
    "random_state": 42,
    "val_metric_name": "1-auc_ovr",
    "verbosity": 2,
    "hidden_sizes": "rectangular",
    "max_one_hot_cat_size": 9,
    "embedding_size": 8,
    "weight_param": "ntk",
    "weight_init_mode": "std",
    "bias_init_mode": "he+5",
    "bias_lr_factor": 0.1,
    "act": "mish",
    "use_parametric_act": True,
    "act_lr_factor": 0.1,
    "wd": 0.013148218987344902,
    "wd_sched": "flat_cos",
    "bias_wd_factor": 0.0,
    "block_str": "w-b-a-d",
    "p_drop": 0.4201348420594983,
    "p_drop_sched": "flat_cos",
    "add_front_scale": True,
    "scale_lr_factor": 3.7236960970475845,
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
    "num_emb_type": "pbld",
    "plr_sigma": 14.70686604689513,
    "plr_hidden_1": 64,
    "plr_hidden_2": 16,
    "plr_lr_factor": 0.05152926209149046,
    "lr": 0.0682536435787459,
    "lr_sched": "coslog4",
    "opt": "adam",
    "sq_mom": 0.9940457705991664,
    "n_hidden_layers": 2,
    "hidden_width": 512,
    "first_layer_lr_factor": 0.5086979438288887,
    "ls_eps_sched": "coslog4",
    "ls_eps": 0.08315114923465,
    "use_ls": False,
    "use_early_stopping": False,
    "n_ens": 8,
}

PRESETS = {
    "realmlp_3config_crosses_priors_drift_original": dict(risk_score=False, configs=THREE_CONFIGS, test_folds=15),
    "realmlp_risk_score_crosses_priors_drift_original": dict(risk_score=True, configs=[ONE_CONFIG], test_folds=5),
}


def dedup_keep_order(seq):
    return list(dict.fromkeys(seq))


def to_bin_target(s: pd.Series) -> np.ndarray:
    return (s.astype(str).str.strip().str.lower() == "yes").astype(np.int32).values


def coerce_totalcharges(df):
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype("float32")
    return df


def clean_total_charges_original(orig_df):
    """Drops the 11 original customers with a blank TotalCharges (NaN after read_original)."""
    out = orig_df.loc[orig_df["TotalCharges"].notna()].copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").astype("float32")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())
    return out


def safe_str_inplace(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("missing").str.strip()


def add_global_rank_and_freq(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)

    add_tr, add_te, add_or = {}, {}, {}
    for c in num_cols:
        freq = all_num[c].value_counts(normalize=True)
        add_tr[f"FREQ_{c}"] = train_df[c].map(freq).fillna(0).astype("float32").values
        add_te[f"FREQ_{c}"] = test_df[c].map(freq).fillna(0).astype("float32").values
        add_or[f"FREQ_{c}"] = orig_df[c].map(freq).fillna(0).astype("float32").values

    for c in num_cols:
        r = all_num[c].rank(method="average", pct=True)
        n_tr, n_te = len(train_df), len(test_df)
        add_tr[f"RANK_{c}"] = r.iloc[:n_tr].astype("float32").values
        add_te[f"RANK_{c}"] = r.iloc[n_tr:n_tr + n_te].astype("float32").values
        add_or[f"RANK_{c}"] = r.iloc[n_tr + n_te:].astype("float32").values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_num_transforms(train_df, test_df, orig_df, num_cols):
    def build(df):
        out = {}
        for c in num_cols:
            v = df[c].astype("float32")
            out[f"LOG1P_{c}"] = np.log1p(v.clip(lower=0)).astype("float32")
            out[f"SQRT_{c}"] = np.sqrt(v.clip(lower=0)).astype("float32")
            out[f"INV1P_{c}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")
            out[f"MISS_{c}"] = pd.to_numeric(df[c], errors="coerce").isna().astype("int8")
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_charge_relationships(train_df, test_df, orig_df):
    def build(df):
        out = {}
        out["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        out["abs_charges_dev"] = np.abs(out["charges_deviation"]).astype("float32")
        out["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        out["total_to_monthly_ratio"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1)).astype("float32")
        out["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
        out["tenure_x_monthly"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
        out["tenure_x_total"] = (df["tenure"] * df["TotalCharges"]).astype("float32")
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_service_counts(train_df, test_df, orig_df):
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for df in (train_df, test_df, orig_df):
        safe_str_inplace(df, service_cols + ["InternetService"])

    def build(df):
        out = {}
        out["service_yes_count"] = (df[service_cols] == "Yes").sum(axis=1).astype("float32")
        out["service_no_count"] = (df[service_cols] == "No").sum(axis=1).astype("float32")
        out["service_other_count"] = (df[service_cols].isin(["No phone service", "No internet service"])).sum(axis=1).astype("float32")
        out["service_count"] = out["service_yes_count"].astype("float32")
        out["has_internet"] = (df["InternetService"] != "No").astype("float32")
        out["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_yes_no_other_flags(train_df, test_df, orig_df):
    yn_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "MultipleLines",
    ]
    for df in (train_df, test_df, orig_df):
        safe_str_inplace(df, yn_cols)

    def build(df):
        out = {}
        for c in yn_cols:
            if c in df.columns:
                s = df[c].astype(str)
                out[f"ISYES_{c}"] = (s == "Yes").astype("float32")
                out[f"ISNO_{c}"] = (s == "No").astype("float32")
                out[f"ISOTHER_{c}"] = (~pd.Series(s).isin(["Yes", "No"])).astype("float32").values
        return pd.DataFrame(out, index=df.index)

    train_df = pd.concat([train_df, build(train_df)], axis=1)
    test_df = pd.concat([test_df, build(test_df)], axis=1)
    orig_df = pd.concat([orig_df, build(orig_df)], axis=1)
    return train_df, test_df, orig_df


def add_bins_and_cross_cats(train_df, test_df, orig_df):
    for df in (train_df, test_df, orig_df):
        df["tenure_bin"] = pd.cut(df["tenure"], bins=TENURE_BINS, include_lowest=True).astype(str)

    all_mc = pd.concat([train_df["MonthlyCharges"], test_df["MonthlyCharges"], orig_df["MonthlyCharges"]], axis=0)
    all_tc = pd.concat([train_df["TotalCharges"], test_df["TotalCharges"], orig_df["TotalCharges"]], axis=0)
    mc_bins = pd.qcut(all_mc, q=MC_BINS, retbins=True, duplicates="drop")[1]
    tc_bins = pd.qcut(all_tc, q=TC_BINS, retbins=True, duplicates="drop")[1]

    for df in (train_df, test_df, orig_df):
        df["MonthlyCharges_bin"] = pd.cut(df["MonthlyCharges"], bins=mc_bins, include_lowest=True).astype(str)
        df["TotalCharges_bin"] = pd.cut(df["TotalCharges"], bins=tc_bins, include_lowest=True).astype(str)

    for a, b in CROSS_PAIRS:
        if a in train_df.columns and b in train_df.columns:
            name = f"{a}__{b}"
            for df in (train_df, test_df, orig_df):
                df[name] = (df[a].astype(str) + "|" + df[b].astype(str))

    for a, b, c in TRIPLE:
        if a in train_df.columns and b in train_df.columns and c in train_df.columns:
            name = f"{a}__{b}__{c}"
            for df in (train_df, test_df, orig_df):
                df[name] = (df[a].astype(str) + "|" + df[b].astype(str) + "|" + df[c].astype(str))

    return train_df, test_df, orig_df


def add_cat_counts_and_rare(train_df, test_df, orig_df, cat_cols, rare_thresh=50):
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0, ignore_index=True)

    add_tr, add_te, add_or = {}, {}, {}
    for c in cat_cols:
        vc = all_cat[c].astype("string").value_counts(dropna=False)
        tr_cnt = pd.to_numeric(train_df[c].astype("string").map(vc), errors="coerce").fillna(0).astype("float32")
        te_cnt = pd.to_numeric(test_df[c].astype("string").map(vc), errors="coerce").fillna(0).astype("float32")
        or_cnt = pd.to_numeric(orig_df[c].astype("string").map(vc), errors="coerce").fillna(0).astype("float32")

        add_tr[f"CAT_CNT_{c}"] = tr_cnt.values
        add_te[f"CAT_CNT_{c}"] = te_cnt.values
        add_or[f"CAT_CNT_{c}"] = or_cnt.values

        add_tr[f"CAT_RARE_{c}"] = (tr_cnt <= rare_thresh).astype("float32").values
        add_te[f"CAT_RARE_{c}"] = (te_cnt <= rare_thresh).astype("float32").values
        add_or[f"CAT_RARE_{c}"] = (or_cnt <= rare_thresh).astype("float32").values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    orig_df = pd.concat([orig_df, pd.DataFrame(add_or, index=orig_df.index)], axis=1)
    return train_df, test_df, orig_df


def add_orig_proba_features(train_df, test_df, orig_df, cols, y_orig_bin):
    orig_tmp = orig_df.copy()
    orig_tmp["_y_"] = y_orig_bin.astype("float32")
    add_tr, add_te = {}, {}
    for c in cols:
        if c not in orig_tmp.columns:
            continue
        m = orig_tmp.groupby(c, observed=False)["_y_"].mean()
        name = f"ORIG_proba_{c}"
        add_tr[name] = train_df[c].map(m).fillna(0.5).astype("float32").values
        add_te[name] = test_df[c].map(m).fillna(0.5).astype("float32").values
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    add_tr, add_te = {}, {}
    new_cols = []
    for c in cols:
        if c not in train_df.columns or c not in orig_df.columns:
            continue
        p_tr = train_df[c].value_counts(normalize=True)
        p_or = orig_df[c].value_counts(normalize=True)
        name = f"DRIFT_{c}"
        add_tr[name] = np.log1p(train_df[c].map(p_tr).fillna(0) / (train_df[c].map(p_or).fillna(eps) + eps)).astype("float32").values
        add_te[name] = np.log1p(test_df[c].map(p_tr).fillna(0) / (test_df[c].map(p_or).fillna(eps) + eps)).astype("float32").values
        new_cols.append(name)
    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df, new_cols


def add_risk_score(train_df, test_df, orig_df):
    """Handcrafted churn-risk score: weighted flags of contract, billing, payment, internet, missing security / support,
    short tenure, high MonthlyCharges (>= the frame's 75th percentile) and a large |TotalCharges - tenure x
    MonthlyCharges| (>= the frame's 80th percentile)."""
    def build(df):
        tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype("float32")
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype("float32")
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype("float32")

        contract = df.get("Contract", "missing").astype(str)
        pay = df.get("PaymentMethod", "missing").astype(str)
        internet = df.get("InternetService", "missing").astype(str)
        sec = df.get("OnlineSecurity", "missing").astype(str)
        tech = df.get("TechSupport", "missing").astype(str)
        paper = df.get("PaperlessBilling", "missing").astype(str)

        m = np.zeros(len(df), dtype=np.float32)
        m += 0.35 * (contract == "Month-to-month").astype(np.float32)
        m += 0.15 * (paper == "Yes").astype(np.float32)
        m += 0.25 * pay.str.contains("Electronic", case=False, na=False).astype(np.float32)
        m += 0.15 * (internet.str.contains("Fiber", case=False, na=False)).astype(np.float32)
        m += 0.10 * ((sec == "No") & (internet != "No")).astype(np.float32)
        m += 0.10 * ((tech == "No") & (internet != "No")).astype(np.float32)

        m += 0.20 * (tenure <= 6).astype(np.float32)
        m += 0.10 * (tenure <= 12).astype(np.float32)
        m += 0.10 * (mc >= np.nanpercentile(mc, 75)).astype(np.float32)

        dev = (tc - tenure * mc).astype(np.float32)
        m += 0.05 * (np.abs(dev) >= np.nanpercentile(np.abs(dev), 80)).astype(np.float32)
        return m.astype("float32")

    train_df = train_df.copy()
    test_df = test_df.copy()
    orig_df = orig_df.copy()
    train_df["risk_score"] = build(train_df)
    test_df["risk_score"] = build(test_df)
    orig_df["risk_score"] = build(orig_df)
    return train_df, test_df, orig_df


def add_orig_mean_by_value_features(train_df, test_df, orig_df, cols, y_orig_bin):
    """orig_{col} = original churn rate of the exact value of col (the original mean if unseen)."""
    orig_mean = float(np.mean(y_orig_bin))
    o = orig_df.copy()
    o["_y_"] = y_orig_bin.astype("float32")

    add_tr, add_te = {}, {}
    for c in cols:
        if c not in o.columns or c not in train_df.columns or c not in test_df.columns:
            continue
        g = o.groupby(c, observed=False)["_y_"].mean()
        nm = f"orig_{c}"
        add_tr[nm] = train_df[c].map(g).fillna(orig_mean).astype("float32").values
        add_te[nm] = test_df[c].map(g).fillna(orig_mean).astype("float32").values

    train_df = pd.concat([train_df, pd.DataFrame(add_tr, index=train_df.index)], axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(add_te, index=test_df.index)], axis=1)
    return train_df, test_df, list(add_tr.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=sorted(PRESETS))
    pair = ap.parse_args().pair
    preset = PRESETS[pair]

    train = read_train()
    test = read_test()
    orig = clean_total_charges_original(read_original())
    print("train:", train.shape, "test:", test.shape, "orig:", orig.shape)

    train = coerce_totalcharges(train)
    test = coerce_totalcharges(test)
    for col in ["MonthlyCharges", "tenure"]:
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")
        test[col] = pd.to_numeric(test[col], errors="coerce").astype("float32")
        orig[col] = pd.to_numeric(orig[col], errors="coerce").astype("float32")
    for df in (train, test, orig):
        for c in BASE_NUMS:
            df[c] = df[c].fillna(df[c].median()).astype("float32")

    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1}).astype(np.int32)
    n_comp = len(train)
    y_orig_bin = to_bin_target(orig[TARGET])

    orig_use = orig.loc[orig[TARGET].astype(str).isin(["Yes", "No"])].copy()
    orig_use[TARGET] = orig_use[TARGET].map({"No": 0, "Yes": 1}).astype(np.int32)

    # append the original rows
    train_model_df = pd.concat([train[FEATS + [TARGET]], orig_use[FEATS + [TARGET]]], axis=0, ignore_index=True)
    test_model_df = test[FEATS].copy()
    print("train after appending orig:", train_model_df.shape)

    train_fe0 = train_model_df[FEATS].copy()
    test_fe0 = test_model_df[FEATS].copy()
    orig_fe0 = orig_use[FEATS].copy()

    safe_str_inplace(train_fe0, BASE_CATS)
    safe_str_inplace(test_fe0, BASE_CATS)
    safe_str_inplace(orig_fe0, BASE_CATS)

    if preset["risk_score"]:
        train_fe0, test_fe0, orig_fe0 = add_risk_score(train_fe0, test_fe0, orig_fe0)

    train_fe0, test_fe0, orig_fe0 = add_global_rank_and_freq(train_fe0, test_fe0, orig_fe0, BASE_NUMS)
    train_fe0, test_fe0, orig_fe0 = add_num_transforms(train_fe0, test_fe0, orig_fe0, BASE_NUMS)
    train_fe0, test_fe0, orig_fe0 = add_charge_relationships(train_fe0, test_fe0, orig_fe0)
    train_fe0, test_fe0, orig_fe0 = add_service_counts(train_fe0, test_fe0, orig_fe0)
    train_fe0, test_fe0, orig_fe0 = add_yes_no_other_flags(train_fe0, test_fe0, orig_fe0)
    train_fe0, test_fe0, orig_fe0 = add_bins_and_cross_cats(train_fe0, test_fe0, orig_fe0)

    engine_cats = ["tenure_bin", "MonthlyCharges_bin", "TotalCharges_bin"]
    engine_cats += [f"{a}__{b}" for a, b in CROSS_PAIRS]
    engine_cats += [f"{a}__{b}__{c}" for a, b, c in TRIPLE]
    engine_cats = dedup_keep_order([c for c in engine_cats if c in train_fe0.columns])

    safe_str_inplace(train_fe0, engine_cats)
    safe_str_inplace(test_fe0, engine_cats)
    safe_str_inplace(orig_fe0, engine_cats)

    all_for_counts = dedup_keep_order(BASE_CATS + engine_cats)
    train_fe0, test_fe0, orig_fe0 = add_cat_counts_and_rare(train_fe0, test_fe0, orig_fe0, all_for_counts,
                                                            rare_thresh=CAT_RARE_THRESH)

    for df in (train_fe0, test_fe0, orig_fe0):
        df["CAT_tenure"] = df["tenure"].astype(str)
        df["CAT_MonthlyCharges"] = df["MonthlyCharges"].astype(str)
        df["CAT_TotalCharges"] = df["TotalCharges"].astype(str)

    orig_proba_keys = dedup_keep_order(BASE_CATS + engine_cats + ["CAT_tenure", "CAT_MonthlyCharges", "CAT_TotalCharges"])
    train_fe0, test_fe0 = add_orig_proba_features(train_fe0, test_fe0, orig_fe0, orig_proba_keys, y_orig_bin)

    orig_mean_cols = []
    if preset["risk_score"]:
        train_fe0, test_fe0, orig_mean_cols = add_orig_mean_by_value_features(
            train_fe0, test_fe0, orig_fe0, cols=FEATS, y_orig_bin=y_orig_bin)
        print(f"original value-rate features: {len(orig_mean_cols)}")

    drift_base = dedup_keep_order(BASE_CATS + engine_cats)
    train_fe0, test_fe0, drift_cols = add_drift_ratio(train_fe0, test_fe0, orig_fe0, drift_base, eps=DRIFT_EPS)
    print(f"drift features: {len(drift_cols)}")

    combined_df = pd.concat([train_fe0.reset_index(drop=True), test_fe0.reset_index(drop=True)],
                            axis=0, ignore_index=True)
    digit, binning, all_cats, rare_flags = add_digit_bin_cat_rare_features(combined_df, FEATS)

    xgb_nums = []
    for c in BASE_NUMS:
        xgb_nums += [f"FREQ_{c}", f"RANK_{c}", f"LOG1P_{c}", f"SQRT_{c}", f"INV1P_{c}", f"MISS_{c}"]
    if preset["risk_score"]:
        xgb_nums += ["risk_score"]
    xgb_nums += [
        "charges_deviation", "abs_charges_dev", "monthly_to_total_ratio", "total_to_monthly_ratio",
        "avg_monthly_charges", "tenure_x_monthly", "tenure_x_total",
        "service_yes_count", "service_no_count", "service_other_count", "service_count", "has_internet", "has_phone",
    ]
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
              "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "MultipleLines"]:
        if f"ISYES_{c}" in combined_df.columns:
            xgb_nums += [f"ISYES_{c}", f"ISNO_{c}", f"ISOTHER_{c}"]
    for c in all_for_counts:
        xgb_nums += [f"CAT_CNT_{c}", f"CAT_RARE_{c}"]
    for c in orig_proba_keys:
        col = f"ORIG_proba_{c}"
        if col in combined_df.columns:
            xgb_nums.append(col)
    xgb_nums += orig_mean_cols
    xgb_nums += drift_cols
    xgb_nums = dedup_keep_order([c for c in xgb_nums if c in combined_df.columns])

    xgb_cats = dedup_keep_order([c for c in engine_cats if c in combined_df.columns])

    features = FEATS + xgb_nums + digit + binning + all_cats + rare_flags + xgb_cats
    features = dedup_keep_order([c for c in features if c in combined_df.columns])

    for c in xgb_cats:
        combined_df[c] = combined_df[c].astype(str)
    str_cols = [c for c in combined_df.columns if combined_df[c].dtype == "object"]
    cat_cols = dedup_keep_order(str_cols + all_cats + xgb_cats)
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype(str).astype("category")

    train_fe = combined_df.iloc[: len(train_model_df)].reset_index(drop=True)
    test_fe = combined_df.iloc[len(train_model_df):].reset_index(drop=True)
    y = train_model_df[TARGET].values.astype(np.int32)
    if not train_fe.columns.is_unique:
        dup = train_fe.columns[train_fe.columns.duplicated()].tolist()
        raise ValueError(f"duplicate columns: {dup[:30]}")
    print("train_fe:", train_fe.shape, "test_fe:", test_fe.shape, "features:", len(features), "cat cols:", len(cat_cols))

    configs = preset["configs"]
    oof, test_pred = cross_validate(train_fe, test_fe, y, features, cat_cols, configs, N_SPLITS, SEED)
    print(f"OOF AUC (competition rows): {roc_auc_score(y[:n_comp], oof[:n_comp]):.6f}")
    if preset["test_folds"] != N_SPLITS:
        _, test_pred = cross_validate(train_fe, test_fe, y, features, cat_cols, configs, preset["test_folds"], SEED)
    save_pair(model_dir(FOLDER), pair, oof[:n_comp], test_pred)


if __name__ == "__main__":
    main()
