#!/usr/bin/env python
"""GPU XGBoost on union features with fold-safe target / count encodings and LapTime-anchor statistics (pair
xgb_union_te_laptime_anchor). Only competition rows are trained on; the original rows feed frequencies, ranks, drift and
original target priors.

Label-free features (train + test + original):
  - Pipeline A: for the 4 numerics with >= 100 levels, value frequency, log1p, sqrt, 1 / (1 + |x|), percentile rank among
    the original values, 50 quantile-bin keys and exact keys (4 decimals); 10 race-state ratios / flags; 7 category
    crosses; counts and rare flags (<= 50) of every category; original target mean and count of every category;
  - Pipeline B: for the 11 numerics, 5 quantile bins (edges from train), exact (4 decimals) / rounded (2, 1, 0, -1) keys
    and 3 digits after the decimal point, factorized over all rows; rounding to tens / units and digits 10^-2..10^2 of 4
    timing columns; log train / original frequency ratios, original target mean and count of every key, category means
    of the numerics over train; pairs of the categories and the first 20 numeric keys with <= 20,000 levels (429 kept);
  - anchor extras: LapTime (s) in thousandths as the anchor, category x anchor keys, 6 anchor digits, 10 digit pairs, 45
    category-pair ids.
Per fold (inner StratifiedKFold(5, shuffle=True, random_state=42) for the training rows): target std / min / max and
smoothed (20) target mean of the A keys; target mean / std / count / nunique / median / min / max / skew and 8 quantiles
per anchor value and the target mean of the 65 category x anchor, digit-pair and category-pair keys (inner KFold(5,
shuffle=True, random_state=42)); counts of the B pairs and smoothed (20) target means of the B keys; the raw keys are
then dropped (1541 columns).
Model: depth 3, lr 0.02, subsample 0.8, colsample_bytree 0.8, up to 20,000 rounds, early stopping 200 on the validation
fold, random_state 42. Test = mean of the 5 fold models; predictions clipped to [1e-7, 1 - 1e-7].
Usage: python train_laptime_anchor_union.py [--folds 0,1,2,3,4] [--force]"""
import argparse
import gc
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, folds, read_original, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402

PAIR = "xgb_union_te_laptime_anchor"
ID_COL = ID
INNER_FOLDS = 5
SEED = 42

MAKE_PAIR_FEATURES = True
PAIR_MAX_UNIQUE = 20_000
Q_LIST = [5]
DRIFT_EPS = 1e-6
ANCHOR_COL = "LapTime (s)"
ANCHOR_CENTS = 1000
MAX_PAIR_CATS = 25
ANCHOR_DIGITS_K = 6
STATS_TE_ANCHOR = ["mean", "std", "count", "nunique", "median", "min", "max", "skew"]
ANCHOR_QUANTILES = [5, 10, 40, 45, 55, 60, 90, 95]
TE2_MAX_COMBOS = 250

XGB_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.02,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': 16,
    'random_state': SEED,
    'early_stopping_rounds': 200,
    'device': 'cuda',
    'tree_method': 'hist',
    'enable_categorical': True,
}


def log(msg):
    print(msg, flush=True)


def safe_name(col):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
    s = re.sub(r"_+", "_", s)
    return s or "feature"


def to_binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("int8")
    return s.astype("string").str.strip().str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0}).fillna(0).astype("int8")


def sanitize_series(s, fill="missing"):
    return s.astype("string").fillna(fill).str.strip()


def as_key(s):
    return pd.Series(s).astype("string").fillna("missing")


def infer_groups(train, test):
    features = [c for c in train.columns if c not in [ID_COL, TARGET]]
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
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def add_qcut_bins(train, test, orig, cols, q_list):
    out = []
    for c in cols:
        if c not in train.columns:
            continue
        p = safe_name(c)
        for q in q_list:
            name = f"B_qcut_{p}_q{q}"
            try:
                b_train, bins = pd.qcut(pd.to_numeric(train[c], errors="coerce"), q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = pd.Series(b_train).astype("float32")
                test[name] = pd.cut(pd.to_numeric(test[c], errors="coerce"), bins=bins, labels=False, include_lowest=True).astype("float32")
                orig[name] = pd.cut(pd.to_numeric(orig[c], errors="coerce"), bins=bins, labels=False, include_lowest=True).astype("float32")
            except Exception:
                train[name] = 0.0
                test[name] = 0.0
                orig[name] = 0.0
            out.append(name)
    return out


def add_round_features(train, test, orig, round_specs):
    cols = []
    for c, rr in round_specs.items():
        if c not in train.columns:
            continue
        p = safe_name(c)
        for r in rr:
            n = f"B_round_{p}_r{r}"
            for df in (train, test, orig):
                df[n] = pd.to_numeric(df[c], errors="coerce").round(r).astype("float32")
            cols.append(n)
    return cols


def add_digit_features(train, test, orig, digit_specs):
    cols = []
    for c, ks in digit_specs.items():
        if c not in train.columns:
            continue
        p = safe_name(c)
        for k in ks:
            n = f"B_digit_{p}_d{k}"
            for df in (train, test, orig):
                x = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64")
                df[n] = (((x * (10 ** k)) % 10).fillna(-1)).astype("int16")
            cols.append(n)
    return cols


def factorize_across(train, test, orig, cols):
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        combined = pd.concat([as_key(train[c]), as_key(test[c]), as_key(orig[c])], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        train[c] = codes[:len(train)].astype("int32")
        test[c] = codes[len(train):len(train) + len(test)].astype("int32")
        orig[c] = codes[len(train) + len(test):].astype("int32")


def make_pair_feature(train, test, orig, cols, max_unique=20_000):
    name = "B_pair_" + "__".join(safe_name(c) for c in cols)
    parts = []
    for df in (train, test, orig):
        s = as_key(df[cols[0]])
        for c in cols[1:]:
            s = s + "|" + as_key(df[c])
        df[name] = s
        parts.append(s)
    combined = pd.concat(parts, ignore_index=True)
    codes, _ = pd.factorize(combined, sort=False)
    if len(np.unique(codes)) > max_unique:
        for df in (train, test, orig):
            df.drop(columns=[name], inplace=True, errors="ignore")
        return None
    train[name] = codes[:len(train)].astype("int32")
    test[name] = codes[len(train):len(train) + len(test)].astype("int32")
    orig[name] = codes[len(train) + len(test):].astype("int32")
    return name


def add_drift(train, test, orig, cols):
    out = []
    for c in cols:
        if c not in train.columns or c not in orig.columns:
            continue
        p_tr = train[c].value_counts(normalize=True, dropna=False)
        p_or = orig[c].value_counts(normalize=True, dropna=False)
        n = f"B_DRIFT_{safe_name(c)}"
        train[n] = np.log1p(train[c].map(p_tr).fillna(0) / (train[c].map(p_or).fillna(DRIFT_EPS) + DRIFT_EPS)).astype("float32")
        test[n] = np.log1p(test[c].map(p_tr).fillna(0) / (test[c].map(p_or).fillna(DRIFT_EPS) + DRIFT_EPS)).astype("float32")
        out.append(n)
    return out


def add_orig_te_ce(train, test, orig, cols, y_orig):
    out = []
    ot = orig.copy()
    ot["_y_"] = y_orig.astype("float32")
    for c in cols:
        if c not in ot.columns:
            continue
        p = safe_name(c)
        n1 = f"B_TEORIG_{p}"
        m = ot.groupby(c, observed=False)["_y_"].mean()
        train[n1] = train[c].map(m).fillna(float(y_orig.mean())).astype("float32")
        test[n1] = test[c].map(m).fillna(float(y_orig.mean())).astype("float32")
        n2 = f"B_CEORIG_{p}"
        vc = ot[c].value_counts(dropna=False)
        train[n2] = train[c].map(vc).fillna(0).astype("float32")
        test[n2] = test[c].map(vc).fillna(0).astype("float32")
        out += [n1, n2]
    return out


def add_group_means(train, test, group_cols, num_cols):
    out = []
    for g in group_cols:
        if g not in train.columns:
            continue
        for ncol in num_cols:
            if ncol not in train.columns or g == ncol:
                continue
            name = f"B_MEAN_{safe_name(ncol)}_BY_{safe_name(g)}"
            mp = train.groupby(g, observed=False)[ncol].mean()
            fill = float(pd.to_numeric(train[ncol], errors="coerce").mean())
            train[name] = train[g].map(mp).fillna(fill).astype("float32")
            test[name] = test[g].map(mp).fillna(fill).astype("float32")
            out.append(name)
    return out


def make_anchor_int(s, cents=1000):
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    return pd.Series(np.rint(v * cents), index=s.index).fillna(-1).astype("int32")


def make_cat_anchor_key(cat_code, anchor_int, scale=50_000_000):
    offset = 10_000_000
    return ((cat_code.astype("int64") + 1) * scale + (anchor_int.astype("int64") + offset)).astype("int64")


def add_anchor_digits(df, anchor_col, k=6):
    cols = []
    v = pd.to_numeric(df[anchor_col], errors="coerce").astype("float64").fillna(0.0)
    p = safe_name(anchor_col)
    for i in range(1, k + 1):
        n = f"X_{p}_digit{i}"
        df[n] = (((v * (10 ** i)) % 10)).astype("int16")
        cols.append(n)
    return cols


def add_digit_pair_cols(df, digit_cols):
    out = []
    use = digit_cols[:5]
    for i in range(len(use) - 1):
        for j in range(i + 1, len(use)):
            n = f"X_digit_{i+1}_{j+1}"
            df[n] = ((df[use[i]].astype("int16") + 1) * 11 + (df[use[j]].astype("int16") + 1)).astype("int16")
            out.append(n)
    return out


def add_cat_pair_id_cols(train, test, cat_cols, max_pair_cats=25):
    use = cat_cols[:max_pair_cats]
    out = []
    for i, c1 in enumerate(use[:-1]):
        for c2 in use[i + 1:]:
            n = f"X_pair_{safe_name(c1)}__{safe_name(c2)}"
            m2 = int(max(train[c2].max(), test[c2].max()) + 2)
            train[n] = ((train[c1].astype("int64") + 1) * m2 + (train[c2].astype("int64") + 1)).astype("int64")
            test[n] = ((test[c1].astype("int64") + 1) * m2 + (test[c2].astype("int64") + 1)).astype("int64")
            out.append(n)
    return out


def safe_map(keys, mapping, fill_value=np.nan):
    out = pd.Series(keys).map(mapping)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out.astype("float32").to_numpy()


def concat_block(df, block):
    if isinstance(block, dict):
        block = pd.DataFrame(block, index=df.index)
    else:
        block = block.copy()
        block.index = df.index
    return pd.concat([df, block], axis=1)


def target_encoder_mean(X_tr, y_tr, X_va, X_te, cols, prefix):
    """Per-column smoothed (20) target mean, out-of-fold over an inner StratifiedKFold for the training rows."""
    out_cols = []
    if not cols:
        return out_cols
    y_tr = np.asarray(y_tr, dtype=np.float32)
    prior = float(y_tr.mean())
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    for c in cols:
        if c not in X_tr.columns or c not in X_va.columns or c not in X_te.columns:
            continue
        out_col = f"{prefix}_{safe_name(c)}"
        # Output names must be unique: pair names can collide after sanitizing.
        if out_col in X_tr.columns or out_col in out_cols:
            out_col = f"{out_col}_{len(out_cols)}"
        out_cols.append(out_col)
        key_all = as_key(X_tr[c]).reset_index(drop=True)
        enc_oof = np.full(len(X_tr), prior, dtype=np.float32)
        for it, iv in inner.split(X_tr, y_tr):
            key_tr = key_all.iloc[it]
            key_va = key_all.iloc[iv]
            stats = pd.DataFrame({"k": key_tr.values, "y": y_tr[it]}).groupby("k")["y"].agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + prior * 20.0) / (stats["count"] + 20.0)
            enc_oof[iv] = key_va.map(enc).fillna(prior).astype("float32").values
        stats_full = pd.DataFrame({"k": key_all.values, "y": y_tr}).groupby("k")["y"].agg(["mean", "count"])
        enc_full = (stats_full["mean"] * stats_full["count"] + prior * 20.0) / (stats_full["count"] + 20.0)
        X_tr[out_col] = enc_oof.astype("float32")
        X_va[out_col] = as_key(X_va[c]).map(enc_full).fillna(prior).astype("float32").values
        X_te[out_col] = as_key(X_te[c]).map(enc_full).fillna(prior).astype("float32").values
    return out_cols


def count_encode_fold(X_tr, X_va, X_te, cols):
    out = []
    for c in cols:
        if c not in X_tr.columns:
            continue
        n = f"CE_{safe_name(c)}"
        vc = X_tr[c].value_counts(dropna=False)
        X_tr[n] = X_tr[c].map(vc).fillna(0).astype("float32")
        X_va[n] = X_va[c].map(vc).fillna(0).astype("float32")
        X_te[n] = X_te[c].map(vc).fillna(0).astype("float32")
        out.append(n)
    return out


def numeric_matrix(df):
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            codes, _ = pd.factorize(as_key(s), sort=False)
            out[c] = codes.astype("int32")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float32")
        else:
            out[c] = out[c].astype("int32")
    return out


def build_features():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = to_binary_target(train[TARGET])
    orig[TARGET] = to_binary_target(orig[TARGET])
    y = train[TARGET].values.astype("int8")
    y_orig = orig[TARGET].values.astype("int8")

    CATS, NUMS, NUMERIC_LIKE = infer_groups(train, test)
    log(f"CATS={CATS}\nNUMS={NUMS}")

    for df in (train, test, orig):
        for c in CATS:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
        for c in NUMERIC_LIKE:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pipeline A.
    A_CATS = list(CATS)
    A_NUMS = list(NUMS)
    A_NEW_CATS, A_NUM_AS_CAT = [], []

    for col in A_NUMS:
        p = safe_name(col)
        freq = pd.concat([train[col], test[col], orig[col]], ignore_index=True).value_counts(normalize=True, dropna=False)
        for df in (train, test, orig):
            v = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
            df[f"A_FREQ_{p}"] = df[col].map(freq).fillna(0).astype("float32")
            df[f"A_LOG1P_{p}"] = np.log1p(v.clip(lower=0)).astype("float32")
            df[f"A_SQRT_{p}"] = np.sqrt(v.clip(lower=0)).astype("float32")
            df[f"A_INV1P_{p}"] = (1.0 / (1.0 + np.abs(v))).astype("float32")
            df[f"A_ORIG_RANK_{p}"] = pctrank_against(v, orig[col])

    for df in (train, test, orig):
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
        df["A_tyre_lap_ratio"] = (tyre / (lap + 1)).astype("float32")
        df["A_lap_per_stint"] = (lap / (stint + 1)).astype("float32")
        df["A_tyre_after_pit"] = (tyre / (pit + 1)).astype("float32")
        df["A_degr_per_tyre"] = (degr / (tyre + 1)).astype("float32")
        df["A_delta_per_degr"] = (delta / (np.abs(degr) + 1)).astype("float32")
        df["A_laptime_per_tyre"] = (lapt / (tyre + 1)).astype("float32")
        df["A_laptime_x_progress"] = (lapt * prog).astype("float32")
        df["A_degr_x_progress"] = (degr * prog).astype("float32")
        df["A_position_risk"] = ((21 - pos).clip(lower=0) + np.abs(pchg)).astype("float32")
        df["A_late_stint"] = ((prog > 0.55).astype("int8") * (tyre > 12).astype("int8")).astype("int8")

    for col in A_NUMS:
        p = safe_name(col)
        allv = pd.concat([train[col], test[col], orig[col]], ignore_index=True)
        try:
            bins = pd.qcut(allv, q=50, retbins=True, duplicates="drop")[1]
        except Exception:
            bins = np.linspace(float(allv.min()), float(allv.max()), 51)
        n = f"A_{p}_bin"
        for df in (train, test, orig):
            df[n] = pd.cut(pd.to_numeric(df[col], errors="coerce"), bins=bins, include_lowest=True).astype(str)
        A_NEW_CATS.append(n)

    cross_pairs = [("Driver", "Race"), ("Driver", "Compound"), ("Race", "Compound"), ("Year", "Race"), ("Stint", "Compound"), ("PitStop", "Stint"), ("Position", "Position_Change")]
    for a, b in cross_pairs:
        if a in train.columns and b in train.columns:
            n = f"A_{safe_name(a)}__{safe_name(b)}"
            for df in (train, test, orig):
                df[n] = as_key(df[a]) + "|" + as_key(df[b])
            A_NEW_CATS.append(n)

    for c in [c for c in A_CATS + A_NEW_CATS if c in train.columns]:
        vc = pd.concat([as_key(train[c]), as_key(test[c]), as_key(orig[c])], ignore_index=True).value_counts(dropna=False)
        p = safe_name(c)
        for df in (train, test, orig):
            df[f"A_CAT_CNT_{p}"] = as_key(df[c]).map(vc).fillna(0).astype("float32")
            df[f"A_CAT_RARE_{p}"] = (df[f"A_CAT_CNT_{p}"] <= 50).astype("float32")

    for col in [c for c in A_CATS + A_NEW_CATS if c in orig.columns]:
        p = safe_name(col)
        m = orig.groupby(col, observed=False)[TARGET].mean()
        vc = orig[col].value_counts(dropna=False)
        for df in (train, test, orig):
            df[f"A_ORIG_proba_{p}"] = df[col].map(m).fillna(float(y_orig.mean())).astype("float32")
            df[f"A_ORIG_count_{p}"] = df[col].map(vc).fillna(0).astype("float32")

    for col in A_NUMS:
        p = safe_name(col)
        n = f"A_CAT_{p}"
        for df in (train, test, orig):
            df[n] = pd.to_numeric(df[col], errors="coerce").round(4).astype(str)
        A_NUM_AS_CAT.append(n)

    # Pipeline B.
    B_NUMS = [c for c in NUMERIC_LIKE if c in train.columns and c in test.columns and c in orig.columns]
    B_CATS = [c for c in CATS if c in train.columns and c in test.columns and c in orig.columns]
    ROUND_SPECS = {c: [-1, 0] for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"] if c in train.columns}
    DIGIT_SPECS = {c: list(range(-2, 3)) for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress"] if c in train.columns}
    add_qcut_bins(train, test, orig, B_NUMS, Q_LIST)
    add_round_features(train, test, orig, ROUND_SPECS)
    B_DIGITS = add_digit_features(train, test, orig, DIGIT_SPECS)

    B_NUM_CAT = []
    for c in B_NUMS:
        p = safe_name(c)
        n = f"B_NUMCAT_{p}"
        for df in (train, test, orig):
            df[n] = pd.to_numeric(df[c], errors="coerce").round(4).astype(str)
        B_NUM_CAT.append(n)
        for r in [2, 1, 0, -1]:
            nr = f"B_{p}_round_{r}"
            for df in (train, test, orig):
                df[nr] = pd.to_numeric(df[c], errors="coerce").round(r).astype(str)
            B_NUM_CAT.append(nr)
        nf = f"B_{p}_frac3"
        for df in (train, test, orig):
            x = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float64")
            df[nf] = np.floor((np.abs(x) * 1000.0) % 1000.0).astype("int32")
        B_NUM_CAT.append(nf)

    factorize_across(train, test, orig, B_CATS + B_NUM_CAT)
    add_drift(train, test, orig, B_CATS + B_NUM_CAT)
    add_orig_te_ce(train, test, orig, B_CATS + B_NUM_CAT + B_DIGITS, y_orig)
    add_group_means(train, test, B_CATS, B_NUMS)

    B_PAIRS = []
    if MAKE_PAIR_FEATURES:
        pair_base = B_CATS + B_NUM_CAT[:20]
        log(f"Building B pair features from {len(pair_base)} columns")
        for cols2 in combinations(pair_base, 2):
            nm = make_pair_feature(train, test, orig, list(cols2), PAIR_MAX_UNIQUE)
            if nm is not None:
                B_PAIRS.append(nm)
        log(f"Kept B pair features: {len(B_PAIRS)}")

    # Anchor extras: LapTime (s) in thousandths.
    train["X_anchor_int"] = make_anchor_int(train[ANCHOR_COL], ANCHOR_CENTS)
    test["X_anchor_int"] = make_anchor_int(test[ANCHOR_COL], ANCHOR_CENTS)
    X_CAT_COLS = list(B_CATS)
    X_COMBO_KEYS = []
    for c in X_CAT_COLS:
        n = f"X_{safe_name(c)}_anc"
        train[n] = make_cat_anchor_key(train[c], train["X_anchor_int"])
        test[n] = make_cat_anchor_key(test[c], test["X_anchor_int"])
        X_COMBO_KEYS.append(n)
    X_DIGITS = add_anchor_digits(train, ANCHOR_COL, ANCHOR_DIGITS_K)
    add_anchor_digits(test, ANCHOR_COL, ANCHOR_DIGITS_K)
    X_DIGIT_PAIRS = add_digit_pair_cols(train, X_DIGITS[:5])
    add_digit_pair_cols(test, X_DIGITS[:5])
    X_CATPAIR_KEYS = add_cat_pair_id_cols(train, test, X_CAT_COLS, MAX_PAIR_CATS)
    X_TE2_KEYS = (X_COMBO_KEYS + X_DIGIT_PAIRS + X_CATPAIR_KEYS)[:TE2_MAX_COMBOS]
    log(f"X_COMBO_KEYS={len(X_COMBO_KEYS)} X_DIGIT_PAIRS={len(X_DIGIT_PAIRS)} X_CATPAIR_KEYS={len(X_CATPAIR_KEYS)} X_TE2_KEYS={len(X_TE2_KEYS)}")

    base_features = [c for c in train.columns if c not in [TARGET, ID_COL]]
    keys = dict(
        a_te=[c for c in A_NUM_AS_CAT + A_CATS + A_NEW_CATS if c in train.columns],
        b_te=[c for c in B_NUM_CAT + B_CATS + B_PAIRS if c in train.columns],
        b_ce=[c for c in B_PAIRS if c in train.columns],
        x_te2=X_TE2_KEYS,
    )
    keys['drop'] = list(dict.fromkeys(keys['a_te'] + keys['b_te'] + X_TE2_KEYS + ["X_anchor_int"]))
    log(f"BASE_FEATURES={len(base_features)} A_TE={len(keys['a_te'])} B_TE={len(keys['b_te'])} B_CE={len(keys['b_ce'])} drop_keys={len(keys['drop'])}")
    return train, test, y, base_features, keys


def run_fold(tr_idx, va_idx, train, test, y, base_features, keys):
    from xgboost import XGBClassifier

    a_te_cols, x_te2_keys = keys['a_te'], keys['x_te2']
    A_STATS = ["std", "min", "max"]
    X_tr = train.iloc[tr_idx][base_features].copy().reset_index(drop=True)
    y_tr = y[tr_idx].copy()
    X_va = train.iloc[va_idx][base_features].copy().reset_index(drop=True)
    y_va = y[va_idx].copy()
    X_te = test[base_features].copy().reset_index(drop=True)

    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    # Target std / min / max of the A keys.
    te1_store = {f"A_TE1_{safe_name(col)}_{stat}": np.full(len(X_tr), np.nan, dtype="float32") for col in a_te_cols for stat in A_STATS}
    for in_tr, in_va in inner.split(X_tr, y_tr):
        yy = y_tr[in_tr]
        for col in a_te_cols:
            tmp = pd.DataFrame({"k": as_key(X_tr.iloc[in_tr][col]).values, "y": yy}).groupby("k")["y"].agg(A_STATS)
            keys_va = as_key(X_tr.iloc[in_va][col])
            for stat in A_STATS:
                te1_store[f"A_TE1_{safe_name(col)}_{stat}"][in_va] = safe_map(keys_va, tmp[stat], fill_value=np.nan)
    X_tr = concat_block(X_tr, te1_store)
    for col in a_te_cols:
        tmp = pd.DataFrame({"k": as_key(X_tr.iloc[:len(y_tr)][col]).values, "y": y_tr}).groupby("k")["y"].agg(A_STATS)
        for stat in A_STATS:
            n = f"A_TE1_{safe_name(col)}_{stat}"
            X_va[n] = safe_map(as_key(X_va[col]), tmp[stat], fill_value=np.nan)
            X_te[n] = safe_map(as_key(X_te[col]), tmp[stat], fill_value=np.nan)
    for df in (X_tr, X_va, X_te):
        cols = [c for c in df.columns if c.startswith("A_TE1_")]
        if cols:
            df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

    target_encoder_mean(X_tr, y_tr, X_va, X_te, a_te_cols, "A_TE")

    # Anchor statistics and anchor / digit / pair key means.
    anchor_key_tr = X_tr["X_anchor_int"].astype("int32")
    inner2 = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
    te_anchor_names = [f"X_TE1_anchor_{s}" for s in STATS_TE_ANCHOR]
    te_q_names = [f"X_Q_{q}" for q in ANCHOR_QUANTILES]
    te2_names = [f"X_TE2_{safe_name(c)}_mean" for c in x_te2_keys]
    store_a = {n: np.full(len(X_tr), np.nan, dtype="float32") for n in te_anchor_names}
    store_q = {n: np.full(len(X_tr), np.nan, dtype="float32") for n in te_q_names}
    store_2 = {n: np.full(len(X_tr), np.nan, dtype="float32") for n in te2_names}
    for in_tr, in_va in inner2.split(X_tr):
        yy = y_tr[in_tr]
        g = pd.DataFrame({"k": anchor_key_tr.iloc[in_tr].values, "y": yy})
        agg = g.groupby("k")["y"].agg(STATS_TE_ANCHOR)
        kva = anchor_key_tr.iloc[in_va]
        for stat in STATS_TE_ANCHOR:
            store_a[f"X_TE1_anchor_{stat}"][in_va] = safe_map(kva, agg[stat], fill_value=np.nan)
        for q in ANCHOR_QUANTILES:
            qmap = g.groupby("k")["y"].quantile(q / 100.0)
            store_q[f"X_Q_{q}"][in_va] = safe_map(kva, qmap, fill_value=np.nan)
        for c in x_te2_keys:
            m = pd.DataFrame({"k": X_tr.iloc[in_tr][c].values, "y": yy}).groupby("k")["y"].mean()
            store_2[f"X_TE2_{safe_name(c)}_mean"][in_va] = safe_map(X_tr.iloc[in_va][c], m, fill_value=np.nan)
    X_tr = concat_block(X_tr, store_a)
    X_tr = concat_block(X_tr, store_q)
    X_tr = concat_block(X_tr, store_2)

    gfull = pd.DataFrame({"k": anchor_key_tr.values, "y": y_tr})
    agg_full = gfull.groupby("k")["y"].agg(STATS_TE_ANCHOR)
    for stat in STATS_TE_ANCHOR:
        n = f"X_TE1_anchor_{stat}"
        X_va[n] = safe_map(X_va["X_anchor_int"], agg_full[stat], fill_value=np.nan)
        X_te[n] = safe_map(X_te["X_anchor_int"], agg_full[stat], fill_value=np.nan)
    for q in ANCHOR_QUANTILES:
        qmap = gfull.groupby("k")["y"].quantile(q / 100.0)
        n = f"X_Q_{q}"
        X_va[n] = safe_map(X_va["X_anchor_int"], qmap, fill_value=np.nan)
        X_te[n] = safe_map(X_te["X_anchor_int"], qmap, fill_value=np.nan)
    for c in x_te2_keys:
        m = pd.DataFrame({"k": X_tr[c].iloc[:len(y_tr)].values, "y": y_tr}).groupby("k")["y"].mean()
        n = f"X_TE2_{safe_name(c)}_mean"
        X_va[n] = safe_map(X_va[c], m, fill_value=np.nan)
        X_te[n] = safe_map(X_te[c], m, fill_value=np.nan)
    for df in (X_tr, X_va, X_te):
        xcols = [c for c in df.columns if c.startswith("X_TE1_") or c.startswith("X_Q_") or c.startswith("X_TE2_")]
        if xcols:
            df[xcols] = df[xcols].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
        if "X_TE1_anchor_count" in df.columns and "X_TE1_anchor_nunique" in df.columns:
            df["X_TE1_count_per_nunique"] = (df["X_TE1_anchor_count"] / df["X_TE1_anchor_nunique"].replace(0, np.nan)).fillna(0).astype("float32")
        if "X_TE1_anchor_std" in df.columns and "X_TE1_anchor_count" in df.columns:
            df["X_TE1_std_per_count"] = (df["X_TE1_anchor_std"] / df["X_TE1_anchor_count"].replace(0, np.nan)).fillna(0).astype("float32")

    count_encode_fold(X_tr, X_va, X_te, keys['b_ce'])
    target_encoder_mean(X_tr, y_tr, X_va, X_te, keys['b_te'], "B_TE")

    drop_now = [c for c in keys['drop'] if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_now, errors="ignore")
    X_va = X_va.drop(columns=drop_now, errors="ignore")
    X_te = X_te.drop(columns=drop_now, errors="ignore")

    # Align and force a numeric-only matrix.
    common_cols = [c for c in X_tr.columns if c in X_va.columns and c in X_te.columns]
    X_tr = numeric_matrix(X_tr[common_cols])
    X_va = numeric_matrix(X_va[common_cols])
    X_te = numeric_matrix(X_te[common_cols])
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_va = model.predict_proba(X_va)[:, 1].astype("float32")
    p_te = model.predict_proba(X_te)[:, 1].astype("float32")
    info = f"features={X_tr.shape[1]} best_iter={getattr(model, 'best_iteration', None)}"
    del X_tr, X_va, X_te, model
    gc.collect()
    return p_va, p_te, y_va, info


def main():
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, PAIR, k)]
    if todo:
        train, test, y, base_features, keys = build_features()
        f = folds(y)
        for k in todo:
            p_va, p_te, y_va, info = run_fold(np.flatnonzero(f != k), np.flatnonzero(f == k), train, test, y, base_features, keys)
            save_part(HERE, PAIR, k, np.clip(p_va, 1e-7, 1 - 1e-7), np.clip(p_te, 1e-7, 1 - 1e-7))
            log(f"{PAIR} fold {k}: AUC {roc_auc_score(y_va, p_va):.6f} {info}")
    y = read_train()[TARGET].values.astype("int8")
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
