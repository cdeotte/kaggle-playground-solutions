#!/usr/bin/env python
"""LightGBM on a feature union with snap features, pair keys and nested target encodings
(pair lgbm_union_snap_pairs_nested_te).

Needs ../snap_features (cache/combo.csv) to run first.

Inputs: train / test / original rows (MonthlyCharges / TotalCharges read as text, then numeric); from combo.csv the snap
columns of the train and test rows (MonthlyCharges_snap, TotalCharges_snap, their differences to the raw values and the
decimal strings mc, mc0, mc1, mc_len, tc, tc0, tc1, tc_len); the original rows get their own values as snap values
(difference 0) and their own decimal strings.
Label-free features (over train + test + original rows):
  - frequency and percentile rank of tenure, the two charges and the two snapped charges; log1p, sqrt, 1 / (1 + x) and
    missing flag of these, of the snap differences and of the decimal counts
  - tenure x MonthlyCharges, TotalCharges / tenure, TotalCharges - tenure x MonthlyCharges, MonthlyCharges /
    (1 + TotalCharges), tenure / (1 + MonthlyCharges), the same products / differences with the snapped charges and
    the absolute snap differences
  - service yes / no / other counts, has-family flag, yes / no / other flags of 10 columns
  - keys: 11 tenure bins, 40 / 60-quantile bins of the charges and the snapped charges, roundings (tenure 1, charges
    0.5 / 5) and integer parts, the decimal strings, 6 category pairs and Contract x InternetService x PaymentMethod
  - count and rare flag (count <= 50) of the 16 categoricals and the tenure bin, their original churn rate and row
    count; log train / original frequency ratio of the 16 categoricals; original churn rate and row count of the
    snapped charges x Contract / InternetService
  - pair keys of 15 columns (Contract, InternetService, PaymentMethod, PaperlessBilling, OnlineSecurity, TechSupport,
    tenure bin, the 4 quantile bins, mc0, mc1, tc0, tc1) with at most 20,000 levels
  - up to 5000-quantile, integer and round-to-2 keys of tenure, the charges and the snapped charges; counts of Contract,
    InternetService, PaymentMethod and Contract x InternetService and their original churn rate and row count
Per fold: every text column that ends in `_te`, starts with `B_pair_` / `L_` or has more than 7 values is replaced by its
nested target mean (inner StratifiedKFold(5, shuffled, seed 42 + fold number) of the training rows; the validation and
test rows get the means of all training rows, unseen values the global mean); the other text columns are factorized.
374 model columns (202 numeric, 154 encoded, 18 factorized).
Model: lr 0.01, 63 leaves, max_depth 6, min_child_samples 200, subsample 0.72 every round, colsample_bytree 0.22,
reg_alpha 8, reg_lambda 14, max_bin 511, seed 42, 32 threads; up to 12,000 rounds with early stopping (400) on the
validation fold.
Usage: python train_union_snap_nested_te.py"""
import gc
import itertools
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import DATA_DIR, ID, ORIGINAL_FILE, TARGET, cache_dir, folds  # noqa: E402
from common.pairs import save_pair  # noqa: E402
from models.lightgbm.lib.kfold import stratified_fold_ids  # noqa: E402

warnings.filterwarnings("ignore")

PAIR = "lgbm_union_snap_pairs_nested_te"
SEED = 42
N_FOLDS = 5
INNER_FOLDS = 5
TEST_FOLDS = 15
TEST_INNER_FOLDS = 10
N_THREADS = 32
ID_COL = ID
COMBO_PATH = cache_dir("snap_features") / "combo.csv"

PAIR_MAX_UNIQUE = 20_000
CAT_RARE_THRESH = 50
HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000

LGB_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=12000,
    learning_rate=0.01,
    num_leaves=63,
    max_depth=6,
    min_child_samples=200,
    subsample=0.72,
    subsample_freq=1,
    colsample_bytree=0.22,
    reg_alpha=8.0,
    reg_lambda=14.0,
    min_split_gain=0.0,
    max_bin=511,
    random_state=SEED,
    verbosity=-1,
    num_threads=N_THREADS,
)

BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
BASE_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
YN_LIKE_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
CROSS_PAIRS = [
    ("Contract", "InternetService"),
    ("PaymentMethod", "Contract"),
    ("InternetService", "OnlineSecurity"),
    ("PaymentMethod", "PaperlessBilling"),
    ("Contract", "PaperlessBilling"),
    ("InternetService", "TechSupport"),
]
TRIPLE = [("Contract", "InternetService", "PaymentMethod")]
SNAP_PAIR_PRIORS = [
    ("MonthlyCharges_snap", "Contract"),
    ("MonthlyCharges_snap", "InternetService"),
    ("TotalCharges_snap", "Contract"),
    ("TotalCharges_snap", "InternetService"),
]


# -----------------------------
# Utilities
# -----------------------------
def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def to_bin_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int8").values
    return s.astype("string").str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype("int8").values


def clean_totalcharges(df):
    s = df["TotalCharges"].astype("string")
    s = s.str.strip()
    s = s.replace({"": pd.NA, " ": pd.NA})
    df["TotalCharges"] = s
    return df


def ensure_str_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing").str.strip()
    return df


def ensure_num_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def consistent_factorize(train_df, valid_df, test_df, cols):
    out_tr = train_df.copy()
    out_va = valid_df.copy()
    out_te = test_df.copy()
    for c in cols:
        allv = pd.concat([
            out_tr[c].astype("string").fillna("missing"),
            out_va[c].astype("string").fillna("missing"),
            out_te[c].astype("string").fillna("missing"),
        ], axis=0, ignore_index=True)
        codes, _ = pd.factorize(allv, sort=True)
        n_tr = len(out_tr)
        n_va = len(out_va)
        out_tr[c] = pd.Series(codes[:n_tr], index=out_tr.index, dtype="int32")
        out_va[c] = pd.Series(codes[n_tr:n_tr + n_va], index=out_va.index, dtype="int32")
        out_te[c] = pd.Series(codes[n_tr + n_va:], index=out_te.index, dtype="int32")
    return out_tr, out_va, out_te


# -----------------------------
# Data + snap features
# -----------------------------
def load_data():
    charge_dtypes = {"TotalCharges": "string", "MonthlyCharges": "string"}
    train = pd.read_csv(DATA_DIR / "train.csv", dtype=charge_dtypes)
    test = pd.read_csv(DATA_DIR / "test.csv", dtype=charge_dtypes)
    orig = pd.read_csv(DATA_DIR / ORIGINAL_FILE, dtype=charge_dtypes)
    orig[ID_COL] = np.arange(len(orig), dtype=np.int32)
    orig = clean_totalcharges(orig)
    train = clean_totalcharges(train)
    test = clean_totalcharges(test)

    train[TARGET] = to_bin_target(train[TARGET])
    orig[TARGET] = to_bin_target(orig[TARGET])
    for df in (train, test, orig):
        ensure_str_cols(df, [c for c in BASE_CATS if c in df.columns])
        ensure_num_cols(df, [c for c in BASE_NUMS if c in df.columns])

    if not COMBO_PATH.exists():
        raise FileNotFoundError(f"{COMBO_PATH} is missing: run ../snap_features/build.py first")
    combo = pd.read_csv(
        COMBO_PATH,
        dtype={
            "TotalCharges": "string",
            "MonthlyCharges": "string",
            "tc": "object",
            "tc0": "object",
            "tc1": "object",
            "mc": "object",
            "mc0": "object",
            "mc1": "object",
            "MonthlyCharges_snap": "string",
            "TotalCharges_snap": "string",
        },
    )
    combo_train = combo.iloc[:len(train)].copy()
    combo_test = combo.iloc[len(train):len(train) + len(test)].copy()
    snap_take = [
        c for c in [
            "MonthlyCharges_snap", "TotalCharges_snap",
            "MonthlyCharges_snap_diff", "TotalCharges_snap_diff",
            "mc", "mc0", "mc1", "mc_len",
            "tc", "tc0", "tc1", "tc_len",
        ] if c in combo.columns
    ]
    train = pd.concat([train.reset_index(drop=True), combo_train[snap_take].reset_index(drop=True)], axis=1)
    test = pd.concat([test.reset_index(drop=True), combo_test[snap_take].reset_index(drop=True)], axis=1)

    # the original rows are their own snap values
    orig["MonthlyCharges_snap"] = orig["MonthlyCharges"].astype("string")
    orig["TotalCharges_snap"] = orig["TotalCharges"].astype("string")
    orig["MonthlyCharges_snap_diff"] = np.float32(0.0)
    orig["TotalCharges_snap_diff"] = np.float32(0.0)
    for base, prefix in [("MonthlyCharges", "mc"), ("TotalCharges", "tc")]:
        s = orig[base].astype("string")
        frac = s.str.split(".").str[1]
        orig[prefix] = frac
        orig[f"{prefix}0"] = frac.str[0].fillna("-1")
        orig[f"{prefix}1"] = "-1"
        frac_len = frac.str.len()
        orig[f"{prefix}_len"] = frac_len.fillna(-1).astype("float32")
        mask = frac_len.eq(2)
        orig.loc[mask, f"{prefix}1"] = frac[mask].str[1].fillna("-1")

    str_snap = ["MonthlyCharges_snap", "TotalCharges_snap", "mc", "mc0", "mc1", "tc", "tc0", "tc1"]
    num_snap = ["MonthlyCharges_snap", "TotalCharges_snap", "MonthlyCharges_snap_diff", "TotalCharges_snap_diff",
                "mc_len", "tc_len"]
    for df in (train, test, orig):
        ensure_str_cols(df, [c for c in str_snap if c in df.columns])
    for df in (train, test, orig):
        ensure_num_cols(df, [c for c in num_snap if c in df.columns])
    print(f"train {train.shape} | test {test.shape} | original {orig.shape}", flush=True)
    return train, test, orig


# -----------------------------
# Feature builders
# -----------------------------
def add_global_rank_freq_num(train_df, test_df, orig_df, num_cols):
    all_num = pd.concat([train_df[num_cols], test_df[num_cols], orig_df[num_cols]], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    out_tr, out_te = {}, {}
    for c in num_cols:
        s = pd.to_numeric(all_num[c], errors="coerce")
        freq = s.value_counts(normalize=True, dropna=False)
        rank = s.rank(method="average", pct=True)
        out_tr[f"A_FREQ_{c}"] = s.iloc[:n_tr].map(freq).astype("float32").values
        out_te[f"A_FREQ_{c}"] = s.iloc[n_tr:n_tr + n_te].map(freq).astype("float32").values
        out_tr[f"A_RANK_{c}"] = rank.iloc[:n_tr].astype("float32").values
        out_te[f"A_RANK_{c}"] = rank.iloc[n_tr:n_tr + n_te].astype("float32").values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def add_num_transforms(df, num_cols, prefix="A"):
    out = {}
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce").astype("float32")
        out[f"{prefix}_LOG1P_{c}"] = np.log1p(v.clip(lower=0)).astype("float32")
        out[f"{prefix}_SQRT_{c}"] = np.sqrt(v.clip(lower=0)).astype("float32")
        out[f"{prefix}_INV1P_{c}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")
        out[f"{prefix}_MISS_{c}"] = v.isna().astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_charge_relationships(df, prefix="A"):
    out = {}
    tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype("float32")
    mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0).astype("float32")
    tc = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0).astype("float32")
    t1 = tenure.clip(lower=1)
    out[f"{prefix}_TENURE_X_MONTHLY"] = (tenure * mc).astype("float32")
    out[f"{prefix}_TOTAL_DIV_TENURE"] = (tc / t1).astype("float32")
    out[f"{prefix}_TOTAL_MINUS_TENURE_MONTHLY"] = (tc - tenure * mc).astype("float32")
    out[f"{prefix}_MONTHLY_DIV_TOTAL1P"] = (mc / (1.0 + tc.clip(lower=0))).astype("float32")
    out[f"{prefix}_TENURE_DIV_MONTHLY1P"] = (tenure / (1.0 + mc.clip(lower=0))).astype("float32")

    if "MonthlyCharges_snap" in df.columns and "TotalCharges_snap" in df.columns:
        mcs = pd.to_numeric(df["MonthlyCharges_snap"], errors="coerce").fillna(0).astype("float32")
        tcs = pd.to_numeric(df["TotalCharges_snap"], errors="coerce").fillna(0).astype("float32")
        out[f"{prefix}_TENURE_X_MONTHLY_SNAP"] = (tenure * mcs).astype("float32")
        out[f"{prefix}_TOTAL_SNAP_MINUS_TENURE_MONTHLY_SNAP"] = (tcs - tenure * mcs).astype("float32")
        out[f"{prefix}_MONTHLY_SNAP_DIFF_ABS"] = pd.to_numeric(df.get("MonthlyCharges_snap_diff", 0), errors="coerce").fillna(0).abs().astype("float32")
        out[f"{prefix}_TOTAL_SNAP_DIFF_ABS"] = pd.to_numeric(df.get("TotalCharges_snap_diff", 0), errors="coerce").fillna(0).abs().astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_service_counts(df, prefix="A"):
    out = {}
    yes_count = np.zeros(len(df), dtype=np.float32)
    no_count = np.zeros(len(df), dtype=np.float32)
    other_count = np.zeros(len(df), dtype=np.float32)
    for c in SERVICE_COLS:
        if c in df.columns:
            s = df[c].astype("string").fillna("missing").str.lower()
            yes = s.eq("yes")
            no = s.eq("no") | s.eq("no internet service") | s.eq("no phone service")
            yes_count += yes.astype("float32").values
            no_count += no.astype("float32").values
            other_count += (~(yes | no)).astype("float32").values
    out[f"{prefix}_SERVICE_YES_COUNT"] = yes_count
    out[f"{prefix}_SERVICE_NO_COUNT"] = no_count
    out[f"{prefix}_SERVICE_OTHER_COUNT"] = other_count
    out[f"{prefix}_HAS_FAMILY"] = (
        df["Partner"].astype("string").str.lower().eq("yes") |
        df["Dependents"].astype("string").str.lower().eq("yes")
    ).astype("float32").values
    return pd.DataFrame(out, index=df.index)


def add_yes_no_other_flags(df, cols, prefix="A"):
    out = {}
    for c in cols:
        if c in df.columns:
            s = df[c].astype("string").fillna("missing").str.lower()
            out[f"{prefix}_{c}_is_yes"] = s.eq("yes").astype("float32").values
            out[f"{prefix}_{c}_is_no"] = s.isin(["no", "no internet service", "no phone service"]).astype("float32").values
            out[f"{prefix}_{c}_is_other"] = (~s.isin(["yes", "no", "no internet service", "no phone service"]).fillna(False)).astype("float32").values
    return pd.DataFrame(out, index=df.index)


def add_bins_cross_and_digits(train_df, test_df, orig_df):
    use_nums = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap"]
                if c in train_df.columns and c in test_df.columns and c in orig_df.columns]
    all_num = pd.concat([train_df[use_nums], test_df[use_nums], orig_df[use_nums]], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    out_all = {}

    if "tenure" in use_nums:
        bins = [-1, 0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 999999]
        out_all["A_tenure_bin"] = pd.cut(all_num["tenure"], bins=bins, labels=False, include_lowest=True).astype("Int64").astype("string")

    qspec = {"MonthlyCharges": 40, "TotalCharges": 60}
    if "MonthlyCharges_snap" in use_nums:
        qspec["MonthlyCharges_snap"] = 40
    if "TotalCharges_snap" in use_nums:
        qspec["TotalCharges_snap"] = 60
    for c, q in qspec.items():
        s = pd.to_numeric(all_num[c], errors="coerce")
        q = int(min(q, max(2, s.nunique(dropna=True))))
        out_all[f"A_qcut_{c}"] = pd.qcut(s, q=q, duplicates="drop").astype("string")

    round_map = {"tenure": 1.0, "MonthlyCharges": 0.5, "TotalCharges": 5.0}
    if "MonthlyCharges_snap" in use_nums:
        round_map["MonthlyCharges_snap"] = 0.5
    if "TotalCharges_snap" in use_nums:
        round_map["TotalCharges_snap"] = 5.0
    for c, step in round_map.items():
        if c in all_num.columns:
            s = pd.to_numeric(all_num[c], errors="coerce")
            out_all[f"A_round_{c}"] = (np.round(s / step) * step).astype("float32").astype("string")
            out_all[f"A_int_{c}"] = np.floor(s.fillna(-9999)).astype("int32").astype("string")

    digit_cols = [c for c in ["mc", "mc0", "mc1", "mc_len", "tc", "tc0", "tc1", "tc_len"]
                  if c in train_df.columns and c in test_df.columns and c in orig_df.columns]
    for c in digit_cols:
        out_all[f"A_digit_{c}"] = pd.concat([train_df[c], test_df[c], orig_df[c]], axis=0, ignore_index=True).astype("string")

    all_cat_source = pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True)
    for a, b in CROSS_PAIRS:
        if a in all_cat_source.columns and b in all_cat_source.columns:
            aa = all_cat_source[a].astype("string").fillna("missing")
            bb = all_cat_source[b].astype("string").fillna("missing")
            out_all[f"A_{a}__{b}_te"] = (aa + "|" + bb).astype("string")
    for a, b, c in TRIPLE:
        if a in all_cat_source.columns and b in all_cat_source.columns and c in all_cat_source.columns:
            aa = all_cat_source[a].astype("string").fillna("missing")
            bb = all_cat_source[b].astype("string").fillna("missing")
            cc = all_cat_source[c].astype("string").fillna("missing")
            out_all[f"A_{a}__{b}__{c}_te"] = (aa + "|" + bb + "|" + cc).astype("string")

    out_all = pd.DataFrame(out_all)
    out_tr = out_all.iloc[:n_tr].set_index(train_df.index)
    out_te = out_all.iloc[n_tr:n_tr + n_te].set_index(test_df.index)
    out_or = out_all.iloc[n_tr + n_te:].set_index(orig_df.index)
    return out_tr, out_te, out_or


def add_cat_count_and_rare(train_df, test_df, orig_df, cols, rare_thresh=50):
    all_cat = pd.concat([
        train_df[cols].astype("string"),
        test_df[cols].astype("string"),
        orig_df[cols].astype("string"),
    ], axis=0, ignore_index=True)
    n_tr, n_te = len(train_df), len(test_df)
    out_tr, out_te = {}, {}
    for c in cols:
        vc = all_cat[c].value_counts(dropna=False)
        cnt = all_cat[c].map(vc).astype("float32")
        out_tr[f"A_CAT_CNT_{c}"] = cnt.iloc[:n_tr].values
        out_te[f"A_CAT_CNT_{c}"] = cnt.iloc[n_tr:n_tr + n_te].values
        out_tr[f"A_CAT_RARE_{c}"] = (cnt.iloc[:n_tr] <= rare_thresh).astype("float32").values
        out_te[f"A_CAT_RARE_{c}"] = (cnt.iloc[n_tr:n_tr + n_te] <= rare_thresh).astype("float32").values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def add_orig_single_priors(train_df, test_df, orig_df, cols):
    y_orig = orig_df[TARGET].astype("float32")
    out_tr, out_te = {}, {}
    for c in cols:
        grp = pd.DataFrame({c: orig_df[c], "y": y_orig}).groupby(c, dropna=False)["y"]
        mean_map = grp.mean()
        size_map = grp.size()
        out_tr[f"A_ORIG_MEAN_{c}"] = train_df[c].map(mean_map).fillna(y_orig.mean()).astype("float32").values
        out_te[f"A_ORIG_MEAN_{c}"] = test_df[c].map(mean_map).fillna(y_orig.mean()).astype("float32").values
        out_tr[f"A_ORIG_CNT_{c}"] = train_df[c].map(size_map).fillna(0).astype("float32").values
        out_te[f"A_ORIG_CNT_{c}"] = test_df[c].map(size_map).fillna(0).astype("float32").values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def add_orig_pair_priors(train_df, test_df, orig_df, pairs):
    out_tr, out_te = {}, {}
    base_mean = orig_df[TARGET].mean()
    for c1, c2 in pairs:
        if c1 not in orig_df.columns or c2 not in orig_df.columns:
            continue
        name = f"A_PAIR_{c1}__{c2}"
        ok = pd.DataFrame({
            c1: orig_df[c1].astype("string"),
            c2: orig_df[c2].astype("string"),
            "y": orig_df[TARGET].astype("float32"),
        })
        ok[name] = ok[c1] + "|" + ok[c2]
        grp_mean = ok.groupby(name, dropna=False)["y"].mean()
        grp_size = ok.groupby(name, dropna=False).size()

        trk = train_df[c1].astype("string") + "|" + train_df[c2].astype("string")
        tek = test_df[c1].astype("string") + "|" + test_df[c2].astype("string")
        out_tr[f"{name}_orig_mean"] = trk.map(grp_mean).fillna(base_mean).astype("float32").values
        out_te[f"{name}_orig_mean"] = tek.map(grp_mean).fillna(base_mean).astype("float32").values
        out_tr[f"{name}_orig_cnt"] = trk.map(grp_size).fillna(0).astype("float32").values
        out_te[f"{name}_orig_cnt"] = tek.map(grp_size).fillna(0).astype("float32").values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def add_drift_ratio(train_df, test_df, orig_df, cols, eps=1e-6):
    out_tr, out_te = {}, {}
    for c in cols:
        p_tr = train_df[c].value_counts(normalize=True, dropna=False)
        p_or = orig_df[c].value_counts(normalize=True, dropna=False)
        out_tr[f"A_DRIFT_{c}"] = np.log1p(train_df[c].map(p_tr).fillna(0) / (train_df[c].map(p_or).fillna(eps) + eps)).astype("float32").values
        out_te[f"A_DRIFT_{c}"] = np.log1p(test_df[c].map(p_tr).fillna(0) / (test_df[c].map(p_or).fillna(eps) + eps)).astype("float32").values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def create_pair_features(train_df, test_df, orig_df, base_cols, pair_max_unique=20000):
    out_tr, out_te = {}, {}
    n_tr, n_te = len(train_df), len(test_df)
    for a, b in itertools.combinations(base_cols, 2):
        sa = pd.concat([train_df[a], test_df[a], orig_df[a]], axis=0, ignore_index=True).astype("string")
        sb = pd.concat([train_df[b], test_df[b], orig_df[b]], axis=0, ignore_index=True).astype("string")
        key = (sa + "|" + sb).astype("string")
        if key.nunique(dropna=False) <= pair_max_unique:
            name = f"B_pair_{a}__{b}"
            out_tr[name] = key.iloc[:n_tr].values
            out_te[name] = key.iloc[n_tr:n_tr + n_te].values
    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def add_quantile_int_round_keys(train_df, test_df, orig_df):
    num_candidates = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap"]
                      if c in train_df.columns]
    n_tr, n_te = len(train_df), len(test_df)
    out_tr, out_te = {}, {}

    for col in num_candidates:
        s_all = pd.concat([train_df[col], test_df[col], orig_df[col]], axis=0, ignore_index=True)
        s_num = pd.to_numeric(s_all, errors="coerce")

        q = int(min(QCUT_BINS, max(2, s_num.nunique(dropna=True))))
        qcat = pd.qcut(s_num, q=q, duplicates="drop").astype("string")
        out_tr[f"L_{col}_quantiled_te"] = qcat.iloc[:n_tr].values
        out_te[f"L_{col}_quantiled_te"] = qcat.iloc[n_tr:n_tr + n_te].values

        intval = np.floor(s_num.fillna(-9999)).astype("int32").astype("string")
        out_tr[f"L_{col}_int_te"] = intval.iloc[:n_tr].values
        out_te[f"L_{col}_int_te"] = intval.iloc[n_tr:n_tr + n_te].values

        round_div2 = (np.round((s_num / 2.0)) * 2.0).astype("float32").astype("string")
        out_tr[f"L_{col}_round_div2_te"] = round_div2.iloc[:n_tr].values
        out_te[f"L_{col}_round_div2_te"] = round_div2.iloc[n_tr:n_tr + n_te].values

    all_df = pd.concat([train_df, test_df, orig_df], axis=0, ignore_index=True)
    for cols in [("Contract",), ("InternetService",), ("PaymentMethod",), ("Contract", "InternetService")]:
        name = f"L_count({'|'.join(cols)})"
        key = all_df[cols[0]].astype("string").fillna("missing")
        for c in cols[1:]:
            key = key + "|" + all_df[c].astype("string").fillna("missing")
        counts = key.map(key.value_counts(dropna=False)).astype("float32")
        out_tr[name] = counts.iloc[:n_tr].values
        out_te[name] = counts.iloc[n_tr:n_tr + n_te].values

    for cols in [("Contract",), ("InternetService",), ("PaymentMethod",), ("Contract", "InternetService")]:
        if len(cols) == 1:
            col = cols[0]
            mapping_mean = orig_df.groupby(col, dropna=False)[TARGET].mean()
            mapping_size = orig_df.groupby(col, dropna=False)[TARGET].size()
            mean_name = f"L_groupby({col})_mean_orig"
            size_name = f"L_groupby({col})_size_orig"
            out_tr[mean_name] = train_df[col].map(mapping_mean).fillna(orig_df[TARGET].mean()).astype("float32").values
            out_te[mean_name] = test_df[col].map(mapping_mean).fillna(orig_df[TARGET].mean()).astype("float32").values
            out_tr[size_name] = train_df[col].map(mapping_size).fillna(0).astype("float32").values
            out_te[size_name] = test_df[col].map(mapping_size).fillna(0).astype("float32").values
        else:
            name = "__".join(cols)
            key_o = orig_df[cols[0]].astype("string").fillna("missing")
            for c in cols[1:]:
                key_o = key_o + "|" + orig_df[c].astype("string").fillna("missing")
            mapping_mean = pd.DataFrame({"k": key_o, "y": orig_df[TARGET]}).groupby("k", dropna=False)["y"].mean()
            mapping_size = pd.DataFrame({"k": key_o, "y": orig_df[TARGET]}).groupby("k", dropna=False).size()

            trk = train_df[cols[0]].astype("string").fillna("missing")
            tek = test_df[cols[0]].astype("string").fillna("missing")
            for c in cols[1:]:
                trk = trk + "|" + train_df[c].astype("string").fillna("missing")
                tek = tek + "|" + test_df[c].astype("string").fillna("missing")
            mean_name = f"L_groupby({name})_mean_orig"
            size_name = f"L_groupby({name})_size_orig"
            out_tr[mean_name] = trk.map(mapping_mean).fillna(orig_df[TARGET].mean()).astype("float32").values
            out_te[mean_name] = tek.map(mapping_mean).fillna(orig_df[TARGET].mean()).astype("float32").values
            out_tr[size_name] = trk.map(mapping_size).fillna(0).astype("float32").values
            out_te[size_name] = tek.map(mapping_size).fillna(0).astype("float32").values

    return pd.DataFrame(out_tr, index=train_df.index), pd.DataFrame(out_te, index=test_df.index)


def build_features():
    train, test, orig = load_data()
    work_tr = train.drop(columns=[TARGET]).copy()
    work_te = test.copy()
    work_or = orig.copy()

    all_nums = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap",
                            "MonthlyCharges_snap_diff", "TotalCharges_snap_diff", "mc_len", "tc_len"] if c in work_tr.columns]
    all_cats = [c for c in BASE_CATS if c in work_tr.columns]

    A_freq_tr, A_freq_te = add_global_rank_freq_num(
        work_tr, work_te, work_or,
        [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap"] if c in work_tr.columns])
    A_num_tr = add_num_transforms(work_tr, all_nums)
    A_num_te = add_num_transforms(work_te, all_nums)
    A_rel_tr = add_charge_relationships(work_tr)
    A_rel_te = add_charge_relationships(work_te)
    A_srv_tr = add_service_counts(work_tr)
    A_srv_te = add_service_counts(work_te)
    A_flag_tr = add_yes_no_other_flags(work_tr, YN_LIKE_COLS)
    A_flag_te = add_yes_no_other_flags(work_te, YN_LIKE_COLS)
    A_bin_tr, A_bin_te, A_bin_or = add_bins_cross_and_digits(work_tr, work_te, work_or)

    tenure_bin = [c for c in ["A_tenure_bin"] if c in A_bin_tr.columns]
    orig_tr_base = pd.concat([work_tr[all_cats], A_bin_tr[tenure_bin]], axis=1)
    orig_te_base = pd.concat([work_te[all_cats], A_bin_te[tenure_bin]], axis=1)
    orig_or_base = pd.concat([work_or[all_cats], A_bin_or[tenure_bin]], axis=1)
    A_cnt_tr, A_cnt_te = add_cat_count_and_rare(orig_tr_base, orig_te_base, orig_or_base,
                                                dedup_keep_order(all_cats + tenure_bin), rare_thresh=CAT_RARE_THRESH)
    orig_prior_cols = dedup_keep_order([c for c in all_cats + ["A_tenure_bin"] if c in pd.concat([work_tr, A_bin_tr], axis=1).columns])
    A_op_tr, A_op_te = add_orig_single_priors(orig_tr_base, orig_te_base, pd.concat([orig_or_base, work_or[[TARGET]]], axis=1),
                                              orig_prior_cols)
    A_dr_tr, A_dr_te = add_drift_ratio(orig_tr_base, orig_te_base, orig_or_base, [c for c in all_cats if c in orig_tr_base.columns])
    A_pp_tr, A_pp_te = add_orig_pair_priors(work_tr, work_te, work_or,
                                            [p for p in SNAP_PAIR_PRIORS if p[0] in work_tr.columns and p[1] in work_tr.columns])

    pair_src_tr = pd.concat([work_tr, A_bin_tr], axis=1)
    pair_src_te = pd.concat([work_te, A_bin_te], axis=1)
    pair_src_or = pd.concat([work_or, A_bin_or], axis=1)
    pair_base = dedup_keep_order([
        c for c in [
            "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
            "OnlineSecurity", "TechSupport", "A_tenure_bin",
            "A_qcut_MonthlyCharges", "A_qcut_TotalCharges",
            "A_qcut_MonthlyCharges_snap", "A_qcut_TotalCharges_snap",
            "mc0", "mc1", "tc0", "tc1",
        ] if c in pair_src_tr.columns
    ])
    B_pair_tr, B_pair_te = create_pair_features(pair_src_tr, pair_src_te, pair_src_or, pair_base, pair_max_unique=PAIR_MAX_UNIQUE)

    L_tr, L_te = add_quantile_int_round_keys(work_tr, work_te, work_or)

    X_train = pd.concat([
        work_tr.reset_index(drop=True),
        A_freq_tr.reset_index(drop=True), A_num_tr.reset_index(drop=True), A_rel_tr.reset_index(drop=True),
        A_srv_tr.reset_index(drop=True), A_flag_tr.reset_index(drop=True), A_bin_tr.reset_index(drop=True),
        A_cnt_tr.reset_index(drop=True), A_op_tr.reset_index(drop=True), A_dr_tr.reset_index(drop=True),
        A_pp_tr.reset_index(drop=True), B_pair_tr.reset_index(drop=True), L_tr.reset_index(drop=True),
    ], axis=1)
    X_test = pd.concat([
        work_te.reset_index(drop=True),
        A_freq_te.reset_index(drop=True), A_num_te.reset_index(drop=True), A_rel_te.reset_index(drop=True),
        A_srv_te.reset_index(drop=True), A_flag_te.reset_index(drop=True), A_bin_te.reset_index(drop=True),
        A_cnt_te.reset_index(drop=True), A_op_te.reset_index(drop=True), A_dr_te.reset_index(drop=True),
        A_pp_te.reset_index(drop=True), B_pair_te.reset_index(drop=True), L_te.reset_index(drop=True),
    ], axis=1)
    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, X_train.columns].copy()

    all_cols = X_train.columns.tolist()
    string_cols = [c for c in all_cols if str(X_train[c].dtype) in ("object", "string") or pd.api.types.is_string_dtype(X_train[c])]
    num_cols_final = [c for c in all_cols if c not in string_cols and c != ID_COL]
    te_cols = []
    for c in string_cols:
        nunq = pd.concat([X_train[c], X_test[c]], axis=0).astype("string").nunique(dropna=False)
        if c.endswith("_te") or c.startswith("B_pair_") or c.startswith("L_") or nunq > HIGH_CARD_THRESHOLD:
            te_cols.append(c)
    te_cols = dedup_keep_order(te_cols)
    cat_factor_cols = [c for c in string_cols if c not in te_cols and c != ID_COL]
    print(f"X_train {X_train.shape} | numeric {len(num_cols_final)} | encoded {len(te_cols)} | "
          f"factorized {len(cat_factor_cols)}", flush=True)

    y = train[TARGET].astype("int8").values
    return X_train, X_test, y, te_cols, cat_factor_cols


# -----------------------------
# Nested target encoding + CV
# -----------------------------
def nested_target_encode(train_col, y_train, valid_col, test_col, n_splits=5, seed=42):
    train_col = train_col.astype("string").fillna("missing")
    valid_col = valid_col.astype("string").fillna("missing")
    test_col = test_col.astype("string").fillna("missing")
    global_mean = float(np.mean(y_train))

    oof = np.full(len(train_col), global_mean, dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_s = pd.Series(y_train, index=train_col.index)
    for sub_tr_idx, sub_va_idx in skf.split(train_col, y_train):
        sub_tr = train_col.iloc[sub_tr_idx]
        sub_y = y_s.iloc[sub_tr_idx]
        sub_map = sub_y.groupby(sub_tr).mean()
        oof[sub_va_idx] = train_col.iloc[sub_va_idx].map(sub_map).fillna(global_mean).astype("float32").values

    full_map = y_s.groupby(train_col).mean()
    va = valid_col.map(full_map).fillna(global_mean).astype("float32").values
    te = test_col.map(full_map).fillna(global_mean).astype("float32").values
    return oof, va, te


def cross_validate(X_train, X_test, y, te_cols, cat_factor_cols, fold_ids, n_folds, inner_folds):
    oof = np.zeros(len(X_train), dtype=np.float32)
    pred = np.zeros(len(X_test), dtype=np.float32)

    for fold in range(1, n_folds + 1):
        tr_idx = np.where(fold_ids != fold - 1)[0]
        va_idx = np.where(fold_ids == fold - 1)[0]
        X_tr = X_train.iloc[tr_idx].copy()
        y_tr = y[tr_idx]
        X_va = X_train.iloc[va_idx].copy()
        y_va = y[va_idx]
        X_te = X_test.copy()

        te_blocks_tr, te_blocks_va, te_blocks_te = {}, {}, {}
        for c in te_cols:
            tr_enc, va_enc, te_enc = nested_target_encode(X_tr[c], y_tr, X_va[c], X_te[c], n_splits=inner_folds,
                                                          seed=SEED + fold)
            te_blocks_tr[f"TE_{c}"] = tr_enc
            te_blocks_va[f"TE_{c}"] = va_enc
            te_blocks_te[f"TE_{c}"] = te_enc
        if len(te_blocks_tr) > 0:
            X_tr = pd.concat([X_tr.drop(columns=te_cols), pd.DataFrame(te_blocks_tr, index=X_tr.index)], axis=1)
            X_va = pd.concat([X_va.drop(columns=te_cols), pd.DataFrame(te_blocks_va, index=X_va.index)], axis=1)
            X_te = pd.concat([X_te.drop(columns=te_cols), pd.DataFrame(te_blocks_te, index=X_te.index)], axis=1)

        fac_cols = [c for c in X_tr.columns if c in cat_factor_cols]
        if len(fac_cols) > 0:
            X_tr, X_va, X_te = consistent_factorize(X_tr, X_va, X_te, fac_cols)

        for df in (X_tr, X_va, X_te):
            for c in df.columns:
                if c == ID_COL:
                    continue
                if pd.api.types.is_bool_dtype(df[c]):
                    df[c] = df[c].astype("int8")
                elif pd.api.types.is_integer_dtype(df[c]):
                    df[c] = df[c].astype("int32")
                else:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            if ID_COL in df.columns:
                df.drop(columns=[ID_COL], inplace=True)

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(400, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
        pred += model.predict_proba(X_te, num_iteration=model.best_iteration_)[:, 1] / n_folds
        print(f"fold {fold}/{n_folds}: AUC {roc_auc_score(y_va, oof[va_idx]):.6f} | best_iteration {model.best_iteration_}",
              flush=True)

        del X_tr, X_va, X_te, model
        gc.collect()
    return oof, pred


def main():
    X_train, X_test, y, te_cols, cat_factor_cols = build_features()
    oof, pred = cross_validate(X_train, X_test, y, te_cols, cat_factor_cols, folds(y), N_FOLDS, INNER_FOLDS)
    print(f"{PAIR}: OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)
    if TEST_FOLDS != N_FOLDS:
        _, pred = cross_validate(X_train, X_test, y, te_cols, cat_factor_cols, stratified_fold_ids(y, TEST_FOLDS),
                                 TEST_FOLDS, TEST_INNER_FOLDS)
    save_pair(HERE, PAIR, oof, pred)


if __name__ == "__main__":
    main()
