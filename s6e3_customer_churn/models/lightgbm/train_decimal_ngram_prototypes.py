#!/usr/bin/env python
"""LightGBM on decimal-string features: hashed character n-grams of the charge texts in a TF-IDF / SVD embedding,
distances to the original rows and to their k-means prototypes, fingerprint-key counts and target encodings
(pair lgbm_decimal_ngram_svd_prototypes).

Needs ../snap_features (cache/combo.csv) to run first.

Inputs: train / test / original rows with MonthlyCharges / TotalCharges as their CSV text; from combo.csv the decimal
strings (mc, mc0, mc1, mc_len, tc, tc0, tc1, tc_len) and the snapped charges of the train / test rows; the original
rows are their own snap values (difference 0) and get their decimal strings from their text.
Features of the train / test rows (the original rows are a reference, not training rows):
  - 18 numerics: the snap differences, the numeric charges and tenure, TotalCharges - tenure x MonthlyCharges and its
    absolute value, TotalCharges / (tenure + 1), the two charge ratios, the deviation ratio, service yes / other counts
    and balance, has internet / phone, original churn rate of the snapped MonthlyCharges / TotalCharges values
  - 5 fingerprint keys (Contract, InternetService, PaymentMethod, PaperlessBilling with the raw charge texts; Contract,
    InternetService, PaymentMethod with the snapped charges; the 4 categoricals with the tenure bin and the decimal
    digits; Contract, InternetService with the decimal strings; Contract, InternetService, 3 services with the
    service-count bin): counts in train (the row itself excluded) / test / original, their ratios, log counts, seen
    flags, original churn / non-churn counts, their log ratio and the smoothed (10) original churn rate
  - embedding: per-row tokens of 12 categoricals, the decimal strings, 9 quantile bins (8 to 16 quantiles over train +
    test + original), 4 fingerprint keys, 9 crosses and the exact text and character 2 / 3-grams of the raw and snapped
    charges; hashed (FeatureHasher, 2^16 columns), TF-IDF, TruncatedSVD (48 components, fitted on the original rows,
    220,000 train rows and 160,000 test rows, seed 42), standardized on the original rows (48 features)
  - in the embedding: distances and cosine similarities to the centroids of all / churned / retained original rows,
    distances to the 1st / 3rd / 5th / 10th nearest original row of each group and the mean distance to the k nearest,
    their gaps; MiniBatchKMeans prototypes of the original rows (96 all, 48 churned, 48 retained; seed 42): nearest
    prototype id, its original size (and log), distance to the nearest and mean distance to the 3 nearest prototypes
  - target encodings over the same 5 folds (each fold's rows encoded with the other 4 folds, test rows with all train
    rows) of 3 fingerprint keys (smoothing 40 / 30 / 30), the 3 prototype ids (20) and the tenure-bin / digit key x
    prototype id (50)
203 features. Model: lr 0.028, 127 leaves, min_data_in_leaf 180, feature_fraction 0.78, bagging 0.82 every round,
lambda_l2 4, max_bin 255, seed 42, 32 threads; up to 7,000 rounds with early stopping (300) on the validation fold.
Usage: python train_decimal_ngram_prototypes.py"""
import gc
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterator, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, ORIGINAL_FILE, TARGET, cache_dir, folds  # noqa: E402
from common.pairs import save_pair  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_decimal_ngram_svd_prototypes"
SEED = 42
N_FOLDS = 5
ID_COL = ID
COMBO_PATH = cache_dir("snap_features") / "combo.csv"

N_HASH_FEATURES = 2 ** 16
HASH_CHUNK_SIZE = 200_000
SVD_COMPONENTS = 48
SVD_FIT_TRAIN_SAMPLE = 220_000
SVD_FIT_TEST_SAMPLE = 160_000

ORIG_K_LIST = [1, 3, 5, 10]
PROTO_ALL_CLUSTERS = 96
PROTO_YES_CLUSTERS = 48
PROTO_NO_CLUSTERS = 48
PROTO_NEIGHBORS = 3

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.028,
    num_leaves=127,
    max_depth=-1,
    min_data_in_leaf=180,
    feature_fraction=0.78,
    bagging_fraction=0.82,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=4.0,
    max_bin=255,
    n_estimators=7000,
    seed=SEED,
    verbosity=-1,
    n_jobs=32,
)

BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
TOKEN_CROSS_PAIRS = [
    ("Contract", "InternetService"),
    ("Contract", "PaymentMethod"),
    ("InternetService", "OnlineSecurity"),
    ("InternetService", "TechSupport"),
    ("PaperlessBilling", "PaymentMethod"),
    ("mc0", "Contract"),
    ("mc1", "PaymentMethod"),
    ("tc0", "Contract"),
    ("tc1", "InternetService"),
]
SUPPORT_KEY_SPECS = {
    "fp_exact": [
        "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
        "MonthlyCharges_raw", "TotalCharges_raw",
    ],
    "fp_snap": [
        "Contract", "InternetService", "PaymentMethod",
        "MonthlyCharges_snap_str", "TotalCharges_snap_str",
    ],
    "fp_relaxed": [
        "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
        "tenure_bin", "mc0", "mc1", "tc0", "tc1",
    ],
    "fp_decimal_pair": [
        "Contract", "InternetService", "mc", "tc",
    ],
    "fp_service": [
        "Contract", "InternetService", "OnlineSecurity", "TechSupport",
        "DeviceProtection", "service_yes_count_bin",
    ],
}
NUMERIC_ARTIFACT_COLS = [
    "MonthlyCharges_snap_diff",
    "TotalCharges_snap_diff",
    "MonthlyCharges_num",
    "TotalCharges_num",
    "tenure_num",
    "charges_deviation",
    "abs_charges_deviation",
    "avg_monthly_charges",
    "monthly_to_total_ratio",
    "total_to_monthly_ratio",
    "charge_dev_ratio",
    "service_yes_count",
    "service_other_count",
    "service_balance",
    "has_internet",
    "has_phone",
    "orig_MonthlyCharges",
    "orig_TotalCharges",
]


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(np.int8).to_numpy()
    m = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0, "1": 1, "0": 0}
    return s.astype("string").map(m).astype("float32").fillna(0).astype(np.int8).to_numpy()


def safe_str_series(s, na_token="NA"):
    return s.astype("string").fillna(na_token).replace({"": na_token, " ": na_token})


def clean_numeric_from_raw(s):
    return pd.to_numeric(s.astype("string").str.strip().replace({"": pd.NA, " ": pd.NA}), errors="coerce")


def _ensure_unique_columns(df):
    cols = []
    seen = {}
    for c in map(str, df.columns):
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
    out = df.copy()
    out.columns = cols
    return out


def _safe_concat(left, right):
    keep = [c for c in right.columns if c not in left.columns]
    if not keep:
        return left.copy()
    return pd.concat([left.reset_index(drop=True), right[keep].reset_index(drop=True)], axis=1)


def load_combo_train_test(n_train, n_test):
    if not COMBO_PATH.exists():
        raise FileNotFoundError(f"{COMBO_PATH} is missing: run ../snap_features/build.py first")
    combo = pd.read_csv(COMBO_PATH, low_memory=False)
    combo = _ensure_unique_columns(combo)
    if len(combo) != n_train + n_test:
        raise ValueError(f"combo.csv has {len(combo)} rows, expected {n_train + n_test}")
    combo = combo.reset_index(drop=True)
    tr = combo.iloc[:n_train].reset_index(drop=True).copy()
    te = combo.iloc[n_train:].reset_index(drop=True).copy()
    return tr, te


# ============================================================
# Data
# ============================================================
def read_main_data():
    dtype = {"MonthlyCharges": "string", "TotalCharges": "string"}
    train = pd.read_csv(DATA_DIR / "train.csv", dtype=dtype, low_memory=False)
    test = pd.read_csv(DATA_DIR / "test.csv", dtype=dtype, low_memory=False)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=dtype, low_memory=False)
    if "customerID" in orig.columns and ID_COL not in orig.columns:
        orig = orig.rename(columns={"customerID": ID_COL})

    for df in (train, test, orig):
        df["MonthlyCharges_raw"] = safe_str_series(df["MonthlyCharges"])
        df["TotalCharges_raw"] = safe_str_series(df["TotalCharges"])
        df["MonthlyCharges_num"] = clean_numeric_from_raw(df["MonthlyCharges_raw"]).astype("float32")
        df["TotalCharges_num"] = clean_numeric_from_raw(df["TotalCharges_raw"]).astype("float32")
        df["tenure_num"] = pd.to_numeric(df["tenure"], errors="coerce").astype("float32")
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype("int8").astype("string")
        for c in BASE_CATS:
            if c in df.columns:
                df[c] = safe_str_series(df[c])

    orig["TotalCharges_num"] = orig["TotalCharges_num"].fillna(0.0)
    train["TotalCharges_num"] = train["TotalCharges_num"].fillna(train["TotalCharges_num"].median())
    test["TotalCharges_num"] = test["TotalCharges_num"].fillna(train["TotalCharges_num"].median())

    y = to_binary_target(train[TARGET])
    y_orig = to_binary_target(orig[TARGET])

    combo_tr, combo_te = load_combo_train_test(len(train), len(test))
    drop_dup = [c for c in ["MonthlyCharges", "TotalCharges", TARGET, ID_COL] if c in combo_tr.columns]
    combo_tr = combo_tr.drop(columns=drop_dup, errors="ignore")
    combo_te = combo_te.drop(columns=drop_dup, errors="ignore")
    train = _safe_concat(train, combo_tr)
    test = _safe_concat(test, combo_te)
    return train, test, orig, y, y_orig


# ============================================================
# Decimal-string / numeric features
# ============================================================
def add_decimal_cols_from_raw(df):
    out = pd.DataFrame(index=df.index)

    def split_frac(col_raw, prefix):
        s = safe_str_series(df[col_raw])
        frac = s.str.split(".", n=1).str[1].fillna("NA").astype("string")
        out[prefix] = frac
        out[f"{prefix}0"] = frac.str[0].fillna("-1").astype("string")
        out[f"{prefix}1"] = frac.str[1].fillna("-1").astype("string")
        out[f"{prefix}_len"] = frac.str.len().fillna(0).astype("int16")

    if "MonthlyCharges_raw" in df.columns:
        split_frac("MonthlyCharges_raw", "mc")
    if "TotalCharges_raw" in df.columns:
        split_frac("TotalCharges_raw", "tc")
    return out


def add_combo_like_orig_columns(orig, y_orig):
    out = pd.DataFrame(index=orig.index)
    out = _safe_concat(out, add_decimal_cols_from_raw(orig))
    out["MonthlyCharges_snap_str"] = safe_str_series(orig["MonthlyCharges_raw"])
    out["TotalCharges_snap_str"] = safe_str_series(orig["TotalCharges_raw"])
    out["MonthlyCharges_snap_diff"] = 0.0
    out["TotalCharges_snap_diff"] = 0.0

    tmp = orig.copy()
    tmp["_y_"] = y_orig.astype("float32")
    mc_prior = tmp.groupby("MonthlyCharges_raw")["_y_"].mean()
    tc_prior = tmp.groupby("TotalCharges_raw")["_y_"].mean()
    out["orig_MonthlyCharges"] = safe_str_series(orig["MonthlyCharges_raw"]).map(mc_prior).fillna(np.mean(y_orig)).astype("float32")
    out["orig_TotalCharges"] = safe_str_series(orig["TotalCharges_raw"]).map(tc_prior).fillna(np.mean(y_orig)).astype("float32")
    return out


def ensure_artifact_columns(train, test, orig, y_orig):
    def _process_main(df):
        df = df.copy()
        need = [c for c in ["mc", "mc0", "mc1", "mc_len", "tc", "tc0", "tc1", "tc_len"] if c not in df.columns]
        if need:
            df = _safe_concat(df, add_decimal_cols_from_raw(df))

        if "MonthlyCharges_snap_str" not in df.columns:
            if "MonthlyCharges_snap" in df.columns:
                df["MonthlyCharges_snap_str"] = safe_str_series(df["MonthlyCharges_snap"])
            else:
                df["MonthlyCharges_snap_str"] = safe_str_series(df["MonthlyCharges_raw"])
        else:
            df["MonthlyCharges_snap_str"] = safe_str_series(df["MonthlyCharges_snap_str"])

        if "TotalCharges_snap_str" not in df.columns:
            if "TotalCharges_snap" in df.columns:
                df["TotalCharges_snap_str"] = safe_str_series(df["TotalCharges_snap"])
            else:
                df["TotalCharges_snap_str"] = safe_str_series(df["TotalCharges_raw"])
        else:
            df["TotalCharges_snap_str"] = safe_str_series(df["TotalCharges_snap_str"])

        if "MonthlyCharges_snap_diff" not in df.columns:
            mc_raw = clean_numeric_from_raw(df["MonthlyCharges_raw"]).fillna(0)
            mc_snap = clean_numeric_from_raw(df["MonthlyCharges_snap_str"]).fillna(0)
            df["MonthlyCharges_snap_diff"] = (mc_snap - mc_raw).astype("float32")
        else:
            df["MonthlyCharges_snap_diff"] = pd.to_numeric(df["MonthlyCharges_snap_diff"], errors="coerce").fillna(0).astype("float32")

        if "TotalCharges_snap_diff" not in df.columns:
            tc_raw = clean_numeric_from_raw(df["TotalCharges_raw"]).fillna(0)
            tc_snap = clean_numeric_from_raw(df["TotalCharges_snap_str"]).fillna(0)
            df["TotalCharges_snap_diff"] = (tc_snap - tc_raw).astype("float32")
        else:
            df["TotalCharges_snap_diff"] = pd.to_numeric(df["TotalCharges_snap_diff"], errors="coerce").fillna(0).astype("float32")

        for c in ["orig_MonthlyCharges", "orig_TotalCharges"]:
            if c not in df.columns:
                df[c] = np.nan
        return df

    train = _process_main(train)
    test = _process_main(test)
    orig = _safe_concat(orig, add_combo_like_orig_columns(orig, y_orig))

    mc_prior = pd.Series(y_orig, index=safe_str_series(orig["MonthlyCharges_raw"])).groupby(level=0).mean()
    tc_prior = pd.Series(y_orig, index=safe_str_series(orig["TotalCharges_raw"])).groupby(level=0).mean()
    prior = float(np.mean(y_orig))

    def _fill_orig_priors(df):
        df = df.copy()
        if df["orig_MonthlyCharges"].isna().any():
            df["orig_MonthlyCharges"] = df["orig_MonthlyCharges"].fillna(
                safe_str_series(df["MonthlyCharges_snap_str"]).map(mc_prior).fillna(prior))
        df["orig_MonthlyCharges"] = pd.to_numeric(df["orig_MonthlyCharges"], errors="coerce").fillna(prior).astype("float32")
        if df["orig_TotalCharges"].isna().any():
            df["orig_TotalCharges"] = df["orig_TotalCharges"].fillna(
                safe_str_series(df["TotalCharges_snap_str"]).map(tc_prior).fillna(prior))
        df["orig_TotalCharges"] = pd.to_numeric(df["orig_TotalCharges"], errors="coerce").fillna(prior).astype("float32")
        return df

    return _fill_orig_priors(train), _fill_orig_priors(test), orig


def qcut_edges(values, q=16):
    x = pd.to_numeric(values, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    try:
        bins = pd.qcut(x, q=min(q, x.nunique()), duplicates="drop", retbins=True)[1]
    except Exception:
        bins = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    bins = np.unique(bins)
    if len(bins) < 2:
        lo, hi = float(x.min()), float(x.max())
        if lo == hi:
            hi = lo + 1.0
        bins = np.array([lo - 1e-9, hi + 1e-9], dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def apply_bin_str(s, bins):
    b = pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=False, include_lowest=True)
    return b.astype("float").fillna(-1).astype(int).astype("string")


def add_numeric_artifact_features(train, test, orig):
    def add_core(df):
        ten = pd.to_numeric(df["tenure_num"], errors="coerce").fillna(0).astype("float32")
        mc = pd.to_numeric(df["MonthlyCharges_num"], errors="coerce").fillna(0).astype("float32")
        tc = pd.to_numeric(df["TotalCharges_num"], errors="coerce").fillna(0).astype("float32")
        df["charges_deviation"] = (tc - ten * mc).astype("float32")
        df["abs_charges_deviation"] = np.abs(df["charges_deviation"]).astype("float32")
        df["avg_monthly_charges"] = (tc / (ten + 1.0)).astype("float32")
        df["monthly_to_total_ratio"] = (mc / (np.abs(tc) + 1.0)).astype("float32")
        df["total_to_monthly_ratio"] = (tc / (np.abs(mc) + 1.0)).astype("float32")
        df["charge_dev_ratio"] = (df["charges_deviation"] / (np.abs(tc) + 1.0)).astype("float32")

        svc = df[SERVICE_COLS].astype("string").fillna("NA")
        df["service_yes_count"] = (svc == "Yes").sum(axis=1).astype("float32")
        df["service_other_count"] = svc.isin(["No phone service", "No internet service"]).sum(axis=1).astype("float32")
        df["service_balance"] = ((svc == "Yes").sum(axis=1) - (svc == "No").sum(axis=1)).astype("float32")
        df["has_internet"] = (safe_str_series(df["InternetService"]) != "No").astype("float32")
        df["has_phone"] = (safe_str_series(df["PhoneService"]) == "Yes").astype("float32")
        return df

    train = add_core(train.copy())
    test = add_core(test.copy())
    orig = add_core(orig.copy())

    bin_src = [
        "tenure_num", "MonthlyCharges_num", "TotalCharges_num",
        "charges_deviation", "avg_monthly_charges", "monthly_to_total_ratio",
        "MonthlyCharges_snap_diff", "TotalCharges_snap_diff", "service_yes_count",
    ]
    comb = pd.concat([train[bin_src], test[bin_src], orig[bin_src]], axis=0, ignore_index=True)
    bin_specs = {
        "tenure_bin": ("tenure_num", qcut_edges(comb["tenure_num"], q=12)),
        "monthly_bin": ("MonthlyCharges_num", qcut_edges(comb["MonthlyCharges_num"], q=16)),
        "total_bin": ("TotalCharges_num", qcut_edges(comb["TotalCharges_num"], q=16)),
        "charge_dev_bin": ("charges_deviation", qcut_edges(comb["charges_deviation"], q=16)),
        "avg_monthly_bin": ("avg_monthly_charges", qcut_edges(comb["avg_monthly_charges"], q=16)),
        "ratio_bin": ("monthly_to_total_ratio", qcut_edges(comb["monthly_to_total_ratio"], q=16)),
        "mcsnapdiff_bin": ("MonthlyCharges_snap_diff", qcut_edges(comb["MonthlyCharges_snap_diff"], q=12)),
        "tcsnapdiff_bin": ("TotalCharges_snap_diff", qcut_edges(comb["TotalCharges_snap_diff"], q=12)),
        "service_yes_count_bin": ("service_yes_count", qcut_edges(comb["service_yes_count"], q=8)),
    }

    def add_bins(df):
        df = df.copy()
        for name, (src, edges) in bin_specs.items():
            df[name] = apply_bin_str(df[src], edges)
        return df

    return add_bins(train), add_bins(test), add_bins(orig)


# ============================================================
# Fingerprint keys + their counts
# ============================================================
def build_support_keys(df):
    out = pd.DataFrame(index=df.index)
    for name, cols in SUPPORT_KEY_SPECS.items():
        parts = []
        for c in cols:
            if c in df.columns:
                parts.append(safe_str_series(df[c]))
            else:
                parts.append(pd.Series(["NA"] * len(df), index=df.index, dtype="string"))
        out[name] = pd.concat(parts, axis=1).agg("||".join, axis=1)
    return out


def add_support_features(train_keys, test_keys, orig_keys, y_orig):
    out_tr = pd.DataFrame(index=train_keys.index)
    out_te = pd.DataFrame(index=test_keys.index)
    orig_prior = float(np.mean(y_orig))
    alpha = 10.0

    for key_name in train_keys.columns:
        s_tr = safe_str_series(train_keys[key_name])
        s_te = safe_str_series(test_keys[key_name])
        s_or = safe_str_series(orig_keys[key_name])

        vc_tr = s_tr.value_counts(dropna=False)
        vc_te = s_te.value_counts(dropna=False)
        vc_or = s_or.value_counts(dropna=False)
        vc_oy = s_or[pd.Series(y_orig, index=s_or.index) == 1].value_counts(dropna=False)
        vc_on = s_or[pd.Series(y_orig, index=s_or.index) == 0].value_counts(dropna=False)

        tr_ct_train = s_tr.map(vc_tr).fillna(0).astype("float32") - 1.0
        te_ct_train = s_te.map(vc_tr).fillna(0).astype("float32")
        tr_ct_test = s_tr.map(vc_te).fillna(0).astype("float32")
        te_ct_test = s_te.map(vc_te).fillna(0).astype("float32")
        tr_ct_orig = s_tr.map(vc_or).fillna(0).astype("float32")
        te_ct_orig = s_te.map(vc_or).fillna(0).astype("float32")
        tr_ct_oy = s_tr.map(vc_oy).fillna(0).astype("float32")
        te_ct_oy = s_te.map(vc_oy).fillna(0).astype("float32")
        tr_ct_on = s_tr.map(vc_on).fillna(0).astype("float32")
        te_ct_on = s_te.map(vc_on).fillna(0).astype("float32")

        for out, ct_train, ct_test, ct_orig, ct_oy_i, ct_on_i in [
            (out_tr, tr_ct_train, tr_ct_test, tr_ct_orig, tr_ct_oy, tr_ct_on),
            (out_te, te_ct_train, te_ct_test, te_ct_orig, te_ct_oy, te_ct_on),
        ]:
            out[f"{key_name}__ct_train"] = ct_train.astype("float32")
            out[f"{key_name}__ct_test"] = ct_test.astype("float32")
            out[f"{key_name}__ct_orig"] = ct_orig.astype("float32")
            out[f"{key_name}__ct_combo"] = (ct_train + ct_test).astype("float32")
            out[f"{key_name}__logct_train"] = np.log1p(np.clip(ct_train, 0, None)).astype("float32")
            out[f"{key_name}__logct_orig"] = np.log1p(np.clip(ct_orig, 0, None)).astype("float32")
            out[f"{key_name}__seen_orig"] = (ct_orig > 0).astype("float32")
            out[f"{key_name}__seen_train"] = (ct_train > 0).astype("float32")
            out[f"{key_name}__train_orig_ratio"] = ((ct_train + 1.0) / (ct_orig + 1.0)).astype("float32")
            out[f"{key_name}__test_orig_ratio"] = ((ct_test + 1.0) / (ct_orig + 1.0)).astype("float32")
            out[f"{key_name}__train_test_ratio"] = ((ct_train + 1.0) / (ct_test + 1.0)).astype("float32")
            out[f"{key_name}__orig_yes_ct"] = ct_oy_i.astype("float32")
            out[f"{key_name}__orig_no_ct"] = ct_on_i.astype("float32")
            out[f"{key_name}__orig_yes_no_logratio"] = np.log((ct_oy_i + 1.0) / (ct_on_i + 1.0)).astype("float32")
            out[f"{key_name}__orig_smoothed_rate"] = ((ct_oy_i + alpha * orig_prior) / (ct_orig + alpha)).astype("float32")
    return out_tr, out_te


# ============================================================
# Tokens -> hashed matrix -> TF-IDF / SVD embedding
# ============================================================
def _char_ngrams(val, prefix):
    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "<na>", "na"):
        return [f"{prefix}=MISSING"]
    s = s.replace(" ", "_").replace("-", "M")
    toks = [f"{prefix}=EXACT__{s}"]
    for n in (2, 3):
        if len(s) >= n:
            for i in range(len(s) - n + 1):
                toks.append(f"{prefix}{n}={s[i:i + n]}")
    return toks


def iter_row_tokens(df) -> Iterator[List[str]]:
    n = len(df)
    arrays: Dict[str, np.ndarray] = {}
    wanted = [
        "MonthlyCharges_raw", "TotalCharges_raw", "MonthlyCharges_snap_str", "TotalCharges_snap_str",
        "mc", "mc0", "mc1", "mc_len", "tc", "tc0", "tc1", "tc_len",
        "tenure_bin", "monthly_bin", "total_bin", "charge_dev_bin", "avg_monthly_bin",
        "ratio_bin", "mcsnapdiff_bin", "tcsnapdiff_bin", "service_yes_count_bin",
        "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
        "OnlineSecurity", "TechSupport", "DeviceProtection", "PhoneService",
        "SeniorCitizen", "Partner", "Dependents", "MultipleLines",
        "fp_exact", "fp_snap", "fp_relaxed", "fp_decimal_pair", "fp_service",
    ]
    for c in wanted:
        if c in df.columns:
            arrays[c] = safe_str_series(df[c]).to_numpy(dtype=object)
        else:
            arrays[c] = np.array(["NA"] * n, dtype=object)

    for i in range(n):
        toks: List[str] = []
        for c in [
            "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
            "OnlineSecurity", "TechSupport", "DeviceProtection", "PhoneService",
            "SeniorCitizen", "Partner", "Dependents", "MultipleLines",
        ]:
            toks.append(f"{c}={arrays[c][i]}")
        for c in ["mc", "mc0", "mc1", "mc_len", "tc", "tc0", "tc1", "tc_len"]:
            toks.append(f"{c}={arrays[c][i]}")
        for c in [
            "tenure_bin", "monthly_bin", "total_bin", "charge_dev_bin",
            "avg_monthly_bin", "ratio_bin", "mcsnapdiff_bin", "tcsnapdiff_bin",
            "service_yes_count_bin",
        ]:
            toks.append(f"{c}={arrays[c][i]}")
        for c in ["fp_snap", "fp_relaxed", "fp_decimal_pair", "fp_service"]:
            toks.append(f"{c}={arrays[c][i]}")
        for a, b in TOKEN_CROSS_PAIRS:
            toks.append(f"{a}__{b}={arrays[a][i]}||{arrays[b][i]}")
        toks.extend(_char_ngrams(arrays["MonthlyCharges_raw"][i], "MC_RAW"))
        toks.extend(_char_ngrams(arrays["TotalCharges_raw"][i], "TC_RAW"))
        toks.extend(_char_ngrams(arrays["MonthlyCharges_snap_str"][i], "MC_SNAP"))
        toks.extend(_char_ngrams(arrays["TotalCharges_snap_str"][i], "TC_SNAP"))
        yield toks


def hash_token_matrix(df, n_features=N_HASH_FEATURES, chunk_size=HASH_CHUNK_SIZE):
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    mats = []
    n = len(df)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = df.iloc[start:stop].reset_index(drop=True)
        mats.append(hasher.transform(iter_row_tokens(chunk)).astype(np.float32).tocsr())
    return sp.vstack(mats, format="csr") if len(mats) > 1 else mats[0]


def fit_tfidf_svd_embeddings(X_train_hash, X_test_hash, X_orig_hash, n_components=SVD_COMPONENTS,
                             train_sample=SVD_FIT_TRAIN_SAMPLE, test_sample=SVD_FIT_TEST_SAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    tr_idx = np.arange(X_train_hash.shape[0])
    te_idx = np.arange(X_test_hash.shape[0])
    if len(tr_idx) > train_sample:
        tr_idx = rng.choice(tr_idx, size=train_sample, replace=False)
    if len(te_idx) > test_sample:
        te_idx = rng.choice(te_idx, size=test_sample, replace=False)
    X_fit = sp.vstack([X_orig_hash, X_train_hash[tr_idx], X_test_hash[te_idx]], format="csr")

    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    X_fit_tfidf = tfidf.fit_transform(X_fit)
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=seed)
    svd.fit(X_fit_tfidf)

    X_train_emb = svd.transform(tfidf.transform(X_train_hash)).astype(np.float32)
    X_test_emb = svd.transform(tfidf.transform(X_test_hash)).astype(np.float32)
    X_orig_emb = svd.transform(tfidf.transform(X_orig_hash)).astype(np.float32)

    scaler = StandardScaler()
    X_orig_emb = scaler.fit_transform(X_orig_emb).astype(np.float32)
    X_train_emb = scaler.transform(X_train_emb).astype(np.float32)
    X_test_emb = scaler.transform(X_test_emb).astype(np.float32)
    return X_train_emb, X_test_emb, X_orig_emb


# ============================================================
# Embedding geometry against the original rows
# ============================================================
def _l2_normalize(x, eps=1e-9):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def add_centroid_knn_features(X_train_emb, X_test_emb, X_orig_emb, y_orig, k_list=ORIG_K_LIST):
    out_tr = pd.DataFrame(index=np.arange(X_train_emb.shape[0]))
    out_te = pd.DataFrame(index=np.arange(X_test_emb.shape[0]))
    ref_all = X_orig_emb
    ref_yes = X_orig_emb[y_orig == 1]
    ref_no = X_orig_emb[y_orig == 0]
    cent_all = ref_all.mean(axis=0, keepdims=True)
    cent_yes = ref_yes.mean(axis=0, keepdims=True)
    cent_no = ref_no.mean(axis=0, keepdims=True)

    for X, out in [(X_train_emb, out_tr), (X_test_emb, out_te)]:
        for name, cent in [("all", cent_all), ("yes", cent_yes), ("no", cent_no)]:
            out[f"emb_cent_dist_{name}"] = np.sqrt(((X - cent) ** 2).sum(axis=1)).astype(np.float32)
        out["emb_cent_gap_yes_no"] = (out["emb_cent_dist_yes"] - out["emb_cent_dist_no"]).astype(np.float32)
        out["emb_cent_ratio_yes_no"] = ((out["emb_cent_dist_yes"] + 1e-6) / (out["emb_cent_dist_no"] + 1e-6)).astype(np.float32)
        out["emb_cent_prob_like"] = (1.0 / (1.0 + np.exp(np.clip(out["emb_cent_gap_yes_no"], -20, 20)))).astype(np.float32)

    cent_stack = _l2_normalize(np.vstack([cent_all, cent_yes, cent_no]).astype(np.float32))
    for X, out in [(X_train_emb, out_tr), (X_test_emb, out_te)]:
        cos = _l2_normalize(X.astype(np.float32)) @ cent_stack.T
        out["emb_cos_all"] = cos[:, 0].astype(np.float32)
        out["emb_cos_yes"] = cos[:, 1].astype(np.float32)
        out["emb_cos_no"] = cos[:, 2].astype(np.float32)
        out["emb_cos_gap_yes_no"] = (out["emb_cos_yes"] - out["emb_cos_no"]).astype(np.float32)

    max_k = max(k_list)
    for grp, ref in {"all": ref_all, "yes": ref_yes, "no": ref_no}.items():
        use_k = min(max_k, ref.shape[0])
        if use_k < 1:
            continue
        nn = NearestNeighbors(n_neighbors=use_k, metric="euclidean", n_jobs=-1)
        nn.fit(ref)
        d_tr, _ = nn.kneighbors(X_train_emb)
        d_te, _ = nn.kneighbors(X_test_emb)
        for k in k_list:
            kk = min(k, use_k)
            out_tr[f"emb_knn_{grp}_dist_{k}"] = d_tr[:, kk - 1].astype(np.float32)
            out_te[f"emb_knn_{grp}_dist_{k}"] = d_te[:, kk - 1].astype(np.float32)
            out_tr[f"emb_knn_{grp}_mean_{k}"] = d_tr[:, :kk].mean(axis=1).astype(np.float32)
            out_te[f"emb_knn_{grp}_mean_{k}"] = d_te[:, :kk].mean(axis=1).astype(np.float32)

    for out in (out_tr, out_te):
        out["emb_knn_gap_yes_no_3"] = (out["emb_knn_yes_mean_3"] - out["emb_knn_no_mean_3"]).astype(np.float32)
        out["emb_knn_ratio_yes_no_3"] = ((out["emb_knn_yes_mean_3"] + 1e-6) / (out["emb_knn_no_mean_3"] + 1e-6)).astype(np.float32)
        out["emb_knn_yes_vs_all_3"] = (out["emb_knn_yes_mean_3"] - out["emb_knn_all_mean_3"]).astype(np.float32)
        out["emb_knn_no_vs_all_3"] = (out["emb_knn_no_mean_3"] - out["emb_knn_all_mean_3"]).astype(np.float32)
    return out_tr, out_te


def add_prototype_features(X_train_emb, X_test_emb, X_orig_emb, y_orig, seed=SEED):
    out_tr = pd.DataFrame(index=np.arange(X_train_emb.shape[0]))
    out_te = pd.DataFrame(index=np.arange(X_test_emb.shape[0]))
    proto_ids_tr: Dict[str, pd.Series] = {}
    proto_ids_te: Dict[str, pd.Series] = {}
    proto_specs = {
        "all": (X_orig_emb, PROTO_ALL_CLUSTERS),
        "yes": (X_orig_emb[y_orig == 1], PROTO_YES_CLUSTERS),
        "no": (X_orig_emb[y_orig == 0], PROTO_NO_CLUSTERS),
    }

    for name, (ref, n_clusters) in proto_specs.items():
        if ref.shape[0] == 0:
            continue
        n_clusters = int(min(n_clusters, max(2, ref.shape[0] // 15)))
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(4096, max(512, ref.shape[0])), random_state=seed,
                             n_init=10)
        km.fit(ref)
        centers = km.cluster_centers_.astype(np.float32)
        ref_sizes = pd.Series(km.predict(ref)).value_counts().sort_index()
        ref_sizes = ref_sizes.reindex(np.arange(n_clusters), fill_value=0)

        nn_cent = NearestNeighbors(n_neighbors=min(PROTO_NEIGHBORS, n_clusters), metric="euclidean", n_jobs=-1)
        nn_cent.fit(centers)
        d_tr, idx_tr = nn_cent.kneighbors(X_train_emb)
        d_te, idx_te = nn_cent.kneighbors(X_test_emb)
        assign_tr = idx_tr[:, 0]
        assign_te = idx_te[:, 0]

        out_tr[f"proto_{name}_id"] = assign_tr.astype(np.float32)
        out_te[f"proto_{name}_id"] = assign_te.astype(np.float32)
        out_tr[f"proto_{name}_size"] = pd.Series(assign_tr).map(ref_sizes).fillna(0).to_numpy(dtype=np.float32)
        out_te[f"proto_{name}_size"] = pd.Series(assign_te).map(ref_sizes).fillna(0).to_numpy(dtype=np.float32)
        out_tr[f"proto_{name}_logsize"] = np.log1p(out_tr[f"proto_{name}_size"]).astype(np.float32)
        out_te[f"proto_{name}_logsize"] = np.log1p(out_te[f"proto_{name}_size"]).astype(np.float32)
        out_tr[f"proto_{name}_dist_1"] = d_tr[:, 0].astype(np.float32)
        out_te[f"proto_{name}_dist_1"] = d_te[:, 0].astype(np.float32)
        out_tr[f"proto_{name}_dist_mean"] = d_tr.mean(axis=1).astype(np.float32)
        out_te[f"proto_{name}_dist_mean"] = d_te.mean(axis=1).astype(np.float32)
        proto_ids_tr[f"proto_{name}_id"] = pd.Series(assign_tr, index=out_tr.index).astype("int32")
        proto_ids_te[f"proto_{name}_id"] = pd.Series(assign_te, index=out_te.index).astype("int32")

    for out in (out_tr, out_te):
        out["proto_gap_yes_no"] = (out["proto_yes_dist_1"] - out["proto_no_dist_1"]).astype(np.float32)
        out["proto_ratio_yes_no"] = ((out["proto_yes_dist_1"] + 1e-6) / (out["proto_no_dist_1"] + 1e-6)).astype(np.float32)
    return out_tr, out_te, proto_ids_tr, proto_ids_te


# ============================================================
# Target encoding over the shared folds
# ============================================================
def make_oof_target_encoding(train_key, test_key, y, fold_list, smoothing=20.0):
    train_key = safe_str_series(pd.Series(train_key).reset_index(drop=True))
    test_key = safe_str_series(pd.Series(test_key).reset_index(drop=True))
    y_s = pd.Series(y).reset_index(drop=True)
    prior = float(y_s.mean())

    oof = np.zeros(len(train_key), dtype=np.float32)
    for tr_idx, va_idx in fold_list:
        stats = pd.DataFrame({"key": train_key.iloc[tr_idx].values, "y": y_s.iloc[tr_idx].values}).groupby("key")["y"].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + smoothing * prior) / (stats["count"] + smoothing)
        oof[va_idx] = train_key.iloc[va_idx].map(enc).fillna(prior).astype(np.float32).values

    stats_full = pd.DataFrame({"key": train_key.values, "y": y_s.values}).groupby("key")["y"].agg(["mean", "count"])
    enc_full = (stats_full["mean"] * stats_full["count"] + smoothing * prior) / (stats_full["count"] + smoothing)
    test_enc = test_key.map(enc_full).fillna(prior).astype(np.float32).values
    return oof, test_enc


# ============================================================
# Assembly + model
# ============================================================
def build_feature_frames(train, test, orig, y, y_orig, fold_list):
    train, test, orig = ensure_artifact_columns(train, test, orig, y_orig)
    train, test, orig = add_numeric_artifact_features(train, test, orig)

    train_keys = build_support_keys(train)
    test_keys = build_support_keys(test)
    orig_keys = build_support_keys(orig)
    train = _safe_concat(train, train_keys)
    test = _safe_concat(test, test_keys)
    orig = _safe_concat(orig, orig_keys)
    f_tr_support, f_te_support = add_support_features(train_keys, test_keys, orig_keys, y_orig)

    keep_num = [c for c in NUMERIC_ARTIFACT_COLS if c in train.columns]
    f_tr_num = train[keep_num].copy()
    f_te_num = test[keep_num].copy()

    print("hashing tokens...", flush=True)
    Xh_train = hash_token_matrix(train)
    Xh_test = hash_token_matrix(test)
    Xh_orig = hash_token_matrix(orig)
    X_train_emb, X_test_emb, X_orig_emb = fit_tfidf_svd_embeddings(Xh_train, Xh_test, Xh_orig)
    emb_cols = [f"emb_{i:02d}" for i in range(X_train_emb.shape[1])]
    f_tr_emb = pd.DataFrame(X_train_emb, columns=emb_cols, index=train.index)
    f_te_emb = pd.DataFrame(X_test_emb, columns=emb_cols, index=test.index)

    f_tr_knn, f_te_knn = add_centroid_knn_features(X_train_emb, X_test_emb, X_orig_emb, y_orig)
    f_tr_proto, f_te_proto, proto_ids_tr, proto_ids_te = add_prototype_features(X_train_emb, X_test_emb, X_orig_emb, y_orig)

    te_tr = pd.DataFrame(index=train.index)
    te_te = pd.DataFrame(index=test.index)
    for key_name, smoothing in [("fp_relaxed", 40.0), ("fp_snap", 30.0), ("fp_service", 30.0)]:
        oof_te, tst_te = make_oof_target_encoding(train[key_name], test[key_name], y, fold_list, smoothing=smoothing)
        te_tr[f"te_{key_name}"] = oof_te
        te_te[f"te_{key_name}"] = tst_te
    for proto_name in sorted(proto_ids_tr.keys()):
        oof_te, tst_te = make_oof_target_encoding(proto_ids_tr[proto_name], proto_ids_te[proto_name], y, fold_list,
                                                  smoothing=20.0)
        te_tr[f"te_{proto_name}"] = oof_te
        te_te[f"te_{proto_name}"] = tst_te
    hybrid_tr = safe_str_series(train["fp_relaxed"]) + "__P" + proto_ids_tr["proto_all_id"].astype("string")
    hybrid_te = safe_str_series(test["fp_relaxed"]) + "__P" + proto_ids_te["proto_all_id"].astype("string")
    oof_te, tst_te = make_oof_target_encoding(hybrid_tr, hybrid_te, y, fold_list, smoothing=50.0)
    te_tr["te_fp_relaxed_proto_all"] = oof_te
    te_te["te_fp_relaxed_proto_all"] = tst_te

    X_train = pd.concat([
        f_tr_num.reset_index(drop=True),
        f_tr_support.reset_index(drop=True),
        f_tr_emb.reset_index(drop=True),
        f_tr_knn.reset_index(drop=True),
        f_tr_proto.reset_index(drop=True),
        te_tr.reset_index(drop=True),
    ], axis=1)
    X_test = pd.concat([
        f_te_num.reset_index(drop=True),
        f_te_support.reset_index(drop=True),
        f_te_emb.reset_index(drop=True),
        f_te_knn.reset_index(drop=True),
        f_te_proto.reset_index(drop=True),
        te_te.reset_index(drop=True),
    ], axis=1)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype(np.float32)

    del Xh_train, Xh_test, Xh_orig, X_train_emb, X_test_emb, X_orig_emb
    gc.collect()
    return X_train, X_test


def fit_lightgbm_oof(X_train, y, X_test, fold_list, params):
    oof = np.zeros(len(X_train), dtype=np.float64)
    test_pred_folds = np.zeros((len(X_test), len(fold_list)), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(fold_list, 1):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y[va_idx]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(250)],
        )
        va_pred = clf.predict_proba(X_va)[:, 1]
        oof[va_idx] = va_pred
        test_pred_folds[:, fold - 1] = clf.predict_proba(X_test)[:, 1]
        print(f"fold {fold}/{len(fold_list)}: AUC {roc_auc_score(y_va, va_pred):.6f} | best_iteration "
              f"{clf.best_iteration_}", flush=True)
        del X_tr, y_tr, X_va, y_va, clf, va_pred
        gc.collect()
    return oof.astype(np.float32), test_pred_folds.mean(axis=1).astype(np.float32)


def main():
    seed_everything(SEED)
    train, test, orig, y, y_orig = read_main_data()
    print(f"train {train.shape} | test {test.shape} | original {orig.shape} | churn rate {y.mean():.6f} "
          f"(original {y_orig.mean():.6f})", flush=True)

    fold_ids = folds(y)
    fold_list = [(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0]) for k in range(N_FOLDS)]

    X_train, X_test = build_feature_frames(train, test, orig, y, y_orig, fold_list)
    print(f"X_train {X_train.shape} | X_test {X_test.shape}", flush=True)

    oof, pred = fit_lightgbm_oof(X_train, y, X_test, fold_list, LGB_PARAMS)
    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    save_pair(HERE, PAIR, oof, pred)


if __name__ == "__main__":
    main()
