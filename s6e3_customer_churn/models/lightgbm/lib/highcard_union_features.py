"""Label-free features of the high-cardinality key recipe (../train_union_te.py), built on the train + test rows; the
original rows give priors, counts and reference distributions.

build_features(union_blocks, extra_blocks) returns (X_train, X_test, y). Every column whose name ends in `_te` is
target-encoded per fold by the training script; string columns become pandas categories (native in LightGBM).

Base blocks (always):
  - the 19 columns (the 15 string categoricals as categories; SeniorCitizen, tenure and the two charges numeric)
  - the columns with more than 7 values in train + test (tenure, MonthlyCharges, TotalCharges): as text keys, their
    train + test row counts, their original churn rate and the ratio of the train + test count to the original count
  - MonthlyCharges, TotalCharges, tenure, SeniorCitizen as keys: round(x) // 2, int(x), up to 5000 quantile bins,
    equal-width bins (5000 / 8000 / 200) and equal-width bins of log1p(x)
  - up to 4 integer and 4 decimal digits of MonthlyCharges and TotalCharges (6-decimal text); the 6 pairs of the 4
    TotalCharges decimal digits as keys
Union blocks (union_blocks=True):
  - frequency over train + test + original and percentile rank of tenure, MonthlyCharges, TotalCharges; log1p, sqrt,
    1 / (1 + x); TotalCharges - tenure * MonthlyCharges and its absolute value, MonthlyCharges / (TotalCharges + 1),
    TotalCharges / (MonthlyCharges + 1), TotalCharges / (tenure + 1), tenure * MonthlyCharges, tenure * TotalCharges;
    service yes / no / other counts, has internet / phone; missing flags; yes / no / other flags of 11 columns
  - keys: 10 tenure bins, 40-quantile bins of MonthlyCharges and 60-quantile bins of TotalCharges (edges over train +
    test + original), 6 category pairs and Contract x InternetService x PaymentMethod
  - original churn rate of the 16 categoricals and the 3 numerics; count and rare flag (count <= 50) of the 16
    categoricals (over train + test + original) and of the 10 new keys (over train + test; these count / flag columns
    keep the `_te` suffix and are encoded too)
  - 5-quantile bins (train edges), roundings to 10 / 1, the hundreds .. tenths digits of MonthlyCharges and the
    hundreds .. hundredths digits of TotalCharges; the 3 numerics, the two charges rounded to 0.1 / 1 / 10 and their
    cents as codes; the 16 categoricals and these codes factorized over train + test + original; log train / original
    frequency ratio (drift) of the categoricals and codes; original churn rate and row count of the categoricals, codes
    and digits; train mean of the 3 numerics per value of every categorical
Distribution / digit / n-gram blocks (extra_blocks=True):
  - TotalCharges percentile rank against the original churners, non-churners and all rows, z-score against the
    non-churners, churner / non-churner gaps of |z| and rank; percentile rank among the original rows of the same
    InternetService / Contract; MonthlyCharges minus the original InternetService mean; distances to the original
    churner / non-churner TotalCharges quartiles and their gaps
  - first / second / last digit, digit count, modulo 10 / 12 / 100, multiple-of flags, roundings to 10 / 100 and the
    distance to them, fractional parts of tenure, MonthlyCharges, TotalCharges; tenure years and months
  - keys: the 3 numerics as text, the 15 pairs of Contract, InternetService, PaymentMethod, OnlineSecurity,
    TechSupport, PaperlessBilling and the 4 triples of the first four"""
import itertools
from itertools import combinations

import numpy as np
import pandas as pd

from common.data import ID, TARGET, labels, read_original, read_test, read_train

pd.options.mode.chained_assignment = None

HIGH_CARD_THRESHOLD = 7
QCUT_BINS = 5000
CUT_BINS_MONTHLY = 5000
CUT_BINS_TOTAL = 8000

NUMS_CAND = ["tenure", "MonthlyCharges", "TotalCharges"]
Q_LIST = [5]
DRIFT_EPS = 1e-6
DIGIT_SPECS = {
    "MonthlyCharges": list(range(-2, 2)),
    "TotalCharges": list(range(-2, 3)),
}
ROUND_SPECS = {
    "MonthlyCharges": [-1, 0],
    "TotalCharges": [-1, 0],
}

TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]
A_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
A_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
YN_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines",
]
TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
CROSS_PAIRS = [
    ("Contract", "InternetService"),
    ("PaymentMethod", "Contract"),
    ("InternetService", "OnlineSecurity"),
    ("PaymentMethod", "PaperlessBilling"),
    ("Contract", "PaperlessBilling"),
    ("InternetService", "TechSupport"),
]
TRIPLE = [("Contract", "InternetService", "PaymentMethod")]


# -----------------------------
# Helpers
# -----------------------------
def sanitize_series(s, fill="missing"):
    return s.astype("string").fillna(fill).str.strip()


def sanitize_str_df(df, fill="missing"):
    out = df.astype("string")
    return out.apply(lambda s: s.fillna(fill).str.strip())


def _reset(df):
    return df.reset_index(drop=True)


def create_categorical_combs(X, list_of_col_tuples, delimiter="|"):
    combs = {}
    for cols in list_of_col_tuples:
        name = delimiter.join(cols)
        combs[name] = X[list(cols)].astype(str).agg(delimiter.join, axis=1)
    return pd.DataFrame(combs)


def create_count_features(X, list_of_inputs):
    list_of_inputs = [[col] if isinstance(col, str) else col for col in list_of_inputs]
    new_features = {}
    for cols in list_of_inputs:
        name = f"count({','.join(cols)})"
        new_features[name] = X.groupby(list(cols), dropna=False).transform("size")
    return pd.DataFrame(new_features)


def create_groupby_features_original(X, list_of_inputs, original_data):
    new_features = {}
    for cols, col, agg in list_of_inputs:
        mapping = original_data.groupby(list(cols), dropna=False)[col].agg(agg).rename(agg)
        name = f"groupby({','.join(cols)})_{agg}_{col}_original"
        new_features[name] = X[list(cols)].merge(mapping, on=cols, how="left")[agg]
    return pd.DataFrame(new_features)


def to_float_totalcharges(s):
    return pd.to_numeric(s.replace(" ", np.nan), errors="coerce")


def add_qcut_bins_3way(train, test, orig, num_cols, q_list):
    new_cols = []
    for c in num_cols:
        if c not in train.columns:
            continue
        for q in q_list:
            name = f"B_qcut_{c}_q{q}"
            try:
                b_tr, bins = pd.qcut(train[c], q=q, labels=False, retbins=True, duplicates="drop")
                train[name] = b_tr.astype("float32")
                test[name] = pd.cut(test[c], bins=bins, labels=False, include_lowest=True).astype("float32")
                orig[name] = pd.cut(orig[c], bins=bins, labels=False, include_lowest=True).astype("float32")
            except Exception:
                train[name] = 0.0
                test[name] = 0.0
                orig[name] = 0.0
            new_cols.append(name)
    return train, test, orig, new_cols


def add_round_features_3way(train, test, orig, round_specs):
    out = []
    for c, rr in round_specs.items():
        if c not in train.columns:
            continue
        for r in rr:
            n = f"B_round_{c}_r{r}"
            train[n] = pd.to_numeric(train[c], errors="coerce").round(r)
            test[n] = pd.to_numeric(test[c], errors="coerce").round(r)
            orig[n] = pd.to_numeric(orig[c], errors="coerce").round(r)
            out.append(n)
    return train, test, orig, out


def add_digit_features_3way(train, test, orig, digit_specs):
    out = []
    for c, ks in digit_specs.items():
        if c not in train.columns:
            continue
        for k in ks:
            n = f"B_digit_{c}_d{k}"
            for df in (train, test, orig):
                v = pd.to_numeric(df[c], errors="coerce")
                df[n] = (((v * (10 ** k)) % 10).fillna(-1)).astype("int8")
            out.append(n)
    return train, test, orig, out


def factorize_across_all_3way(train, test, orig, cols):
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        codes, _ = combined.factorize(sort=False)
        train[c] = codes[:len(train)]
        test[c] = codes[len(train):len(train) + len(test)]
        orig[c] = codes[len(train) + len(test):]
    return train, test, orig


def add_drift_ratio(train, test, orig, cols, eps=1e-6):
    new_cols = []
    for c in cols:
        if c not in train.columns or c not in test.columns or c not in orig.columns:
            continue
        p_tr = train[c].value_counts(normalize=True)
        p_or = orig[c].value_counts(normalize=True)
        name = f"B_DRIFT_{c}"
        train[name] = np.log1p(train[c].map(p_tr).fillna(0) / (train[c].map(p_or).fillna(eps) + eps)).astype("float32")
        test[name] = np.log1p(test[c].map(p_tr).fillna(0) / (test[c].map(p_or).fillna(eps) + eps)).astype("float32")
        new_cols.append(name)
    return train, test, new_cols


def add_orig_te_ce(train, test, orig, cols, y_orig):
    te_cols, ce_cols = [], []
    tmp = orig.copy()
    tmp["_y_"] = y_orig.astype("float32")
    for c in cols:
        if c not in tmp.columns:
            continue
        te_name = f"B_TEORIG_{c}"
        m = tmp.groupby(c, dropna=False)["_y_"].mean().astype("float32")
        train[te_name] = train[c].map(m).astype("float32")
        test[te_name] = test[c].map(m).astype("float32")
        fillv = float(np.nanmean(train[te_name].values))
        train[te_name] = train[te_name].fillna(fillv).astype("float32")
        test[te_name] = test[te_name].fillna(fillv).astype("float32")
        te_cols.append(te_name)

        ce_name = f"B_CEORIG_{c}"
        vc = tmp[c].value_counts(dropna=False)
        train[ce_name] = train[c].map(vc).fillna(0).astype("float32")
        test[ce_name] = test[c].map(vc).fillna(0).astype("float32")
        ce_cols.append(ce_name)
    return train, test, te_cols, ce_cols


def add_group_means(train, test, group_cols, num_cols):
    new_cols = []
    for g in group_cols:
        if g not in train.columns:
            continue
        for n in num_cols:
            if n not in train.columns or g == n:
                continue
            name = f"B_MEAN_{n}_BY_{g}"
            m = train.groupby(g, dropna=False)[n].mean()
            train[name] = train[g].map(m).astype("float32")
            test[name] = test[g].map(m).astype("float32")
            fillv = float(np.nanmean(train[name].values))
            train[name] = train[name].fillna(fillv).astype("float32")
            test[name] = test[name].fillna(fillv).astype("float32")
            new_cols.append(name)
    return train, test, new_cols


def pctrank_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    reference = np.asarray(reference)
    if len(reference) == 0:
        return np.zeros(len(values), dtype="float32")
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")


# -----------------------------
# Data
# -----------------------------
def load_data():
    train = read_train()
    test = read_test()
    orig = read_original()
    train[TARGET] = labels(train[TARGET])
    orig[TARGET] = labels(orig[TARGET])
    for df in (train, test, orig):
        if ID in df.columns:
            df.drop(columns=[ID], inplace=True)
    for df in (train, test, orig):
        df["TotalCharges"] = to_float_totalcharges(df["TotalCharges"])
        for c in ["MonthlyCharges", "tenure", "SeniorCitizen"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return train, test, orig


# -----------------------------
# Base blocks
# -----------------------------
def base_blocks(combined, orig):
    n_combined = len(combined)
    high_cardinality = [col for col in combined.columns if combined[col].nunique(dropna=False) > HIGH_CARD_THRESHOLD]
    print(f"columns with > {HIGH_CARD_THRESHOLD} values: {high_cardinality}", flush=True)

    te1 = _reset(combined[high_cardinality].astype(str).add_suffix("_te"))
    ce1 = _reset(create_count_features(combined, high_cardinality))
    inputs_mean = [([col], TARGET, "mean") for col in high_cardinality]
    te1_original = _reset(create_groupby_features_original(combined, inputs_mean, orig))
    inputs_size = [([col], TARGET, "size") for col in high_cardinality]
    ce1_original = _reset(create_groupby_features_original(combined, inputs_size, orig))

    count_ratios = {}
    for j, colname in enumerate(ce1.columns):
        denom = ce1_original.iloc[:, j].astype("float32")
        numer = ce1.iloc[:, j].astype("float32")
        ratio = numer / (denom.replace(0, np.nan))
        count_ratios[f"{colname}_ratio"] = ratio.fillna(0.0)
    count_ratios = _reset(pd.DataFrame(count_ratios))

    num_bins = {}
    for col in ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"]:
        s = combined[col]
        num_bins[f"{col}_round_div2_te"] = (pd.to_numeric(s, errors="coerce").round().fillna(-1) // 2).astype("int64").astype(str)
        num_bins[f"{col}_int_te"] = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").astype(str)
        try:
            q = int(min(QCUT_BINS, max(2, s.nunique(dropna=True))))
            num_bins[f"{col}_quantiled_te"] = pd.qcut(pd.to_numeric(s, errors="coerce"), q=q, duplicates="drop").astype(str)
        except Exception:
            num_bins[f"{col}_quantiled_te"] = pd.Series(["NA"] * n_combined)
        bins = CUT_BINS_MONTHLY if col == "MonthlyCharges" else (CUT_BINS_TOTAL if col == "TotalCharges" else 200)
        try:
            num_bins[f"{col}_binned_te"] = pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, duplicates="drop").astype(str)
        except Exception:
            num_bins[f"{col}_binned_te"] = pd.Series(["NA"] * n_combined)
        try:
            num_bins[f"{col}_log_binned_te"] = pd.cut(
                np.log1p(pd.to_numeric(s, errors="coerce").clip(lower=0)),
                bins=max(200, bins // 2),
                duplicates="drop",
            ).astype(str)
        except Exception:
            num_bins[f"{col}_log_binned_te"] = pd.Series(["NA"] * n_combined)
    num_bins = _reset(pd.DataFrame(num_bins))

    digits = {}
    float_cols = [c for c in combined.columns if pd.api.types.is_float_dtype(combined[c])]
    for col in float_cols:
        s = combined[col]
        ss = pd.to_numeric(s, errors="coerce").fillna(0).astype("float64").map(lambda x: f"{x:.6f}")
        splitted = ss.str.split(".", expand=True)
        non_decimal_part = splitted[0]
        decimal_part = splitted[1] if splitted.shape[1] > 1 else pd.Series(["0"] * len(ss))
        max_len_non_decimal = int(non_decimal_part.str.len().max())
        max_len_decimal = int(decimal_part.str.len().max())
        non_decimal_part_padded = non_decimal_part.str.rjust(max_len_non_decimal, "0")
        decimal_part_padded = decimal_part.str.ljust(max_len_decimal, "0")
        keep_non = min(max_len_non_decimal, 4)
        keep_dec = min(max_len_decimal, 4)
        for i in range(keep_non):
            digits[f"{col}_digit{i}"] = non_decimal_part_padded.str[i].astype(int)
        for i in range(keep_dec):
            digits[f"{col}_decimal_digit{i}"] = decimal_part_padded.str[i].astype(int)
    digits = _reset(pd.DataFrame(digits))

    digits_combs = pd.DataFrame(index=pd.RangeIndex(n_combined))
    candidate = [c for c in digits.columns if c.startswith("TotalCharges_")] or [c for c in digits.columns if c.startswith("MonthlyCharges_")]
    candidate = [c for c in candidate if "decimal_digit" in c][:6]
    if len(candidate) >= 2:
        dig_pairs = list(itertools.combinations(candidate, 2))
        digits_combs = create_categorical_combs(digits[candidate].astype(str), dig_pairs).add_suffix("_te")
    digits_combs = _reset(digits_combs)

    return [te1, ce1, te1_original, num_bins, digits, digits_combs, count_ratios]


# -----------------------------
# Union blocks
# -----------------------------
def union_blocks_a(train, test, orig):
    n_train, n_test = len(train), len(test)
    trA = _reset(train.drop(columns=[TARGET]))
    teA = _reset(test.copy())
    orA = orig.copy()

    for df in (trA, teA, orA):
        for c in A_CATS:
            if c in df.columns:
                df[c] = sanitize_series(df[c])
        for c in A_NUMS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    A_blocks = {}
    for col in A_NUMS:
        freq = pd.concat([trA[col], teA[col], orA[col]], axis=0).value_counts(normalize=True)
        tt = pd.concat([trA[col], teA[col]], axis=0, ignore_index=True)
        A_blocks[f"A_FREQ_{col}"] = tt.map(freq).fillna(0).astype("float32")

    _all_num = pd.concat([trA[A_NUMS], teA[A_NUMS], orA[A_NUMS]], axis=0, ignore_index=True)
    for col in A_NUMS:
        r = _all_num[col].rank(method="average", pct=True)
        A_blocks[f"A_RANK_{col}"] = r.iloc[:n_train + n_test].astype("float32").values

    for col in A_NUMS:
        v = pd.to_numeric(pd.concat([trA[col], teA[col]], axis=0, ignore_index=True), errors="coerce").fillna(0).astype("float32")
        A_blocks[f"A_LOG1P_{col}"] = np.log1p(v.clip(lower=0)).astype("float32")
        A_blocks[f"A_SQRT_{col}"] = np.sqrt(v.clip(lower=0)).astype("float32")
        A_blocks[f"A_INV1P_{col}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")

    tc = pd.concat([trA["TotalCharges"], teA["TotalCharges"]], axis=0, ignore_index=True).astype("float32")
    ten = pd.concat([trA["tenure"], teA["tenure"]], axis=0, ignore_index=True).astype("float32")
    mc = pd.concat([trA["MonthlyCharges"], teA["MonthlyCharges"]], axis=0, ignore_index=True).astype("float32")
    dev = (tc - ten * mc).astype("float32")
    A_blocks["A_charges_deviation"] = dev
    A_blocks["A_abs_charges_dev"] = np.abs(dev).astype("float32")
    A_blocks["A_monthly_to_total_ratio"] = (mc / (tc + 1)).astype("float32")
    A_blocks["A_total_to_monthly_ratio"] = (tc / (mc + 1)).astype("float32")
    A_blocks["A_avg_monthly_charges"] = (tc / (ten + 1)).astype("float32")
    A_blocks["A_tenure_x_monthly"] = (ten * mc).astype("float32")
    A_blocks["A_tenure_x_total"] = (ten * tc).astype("float32")

    combA = pd.concat([trA[SERVICE_COLS + ["InternetService"]], teA[SERVICE_COLS + ["InternetService"]]], axis=0, ignore_index=True)
    for c in SERVICE_COLS + ["InternetService"]:
        combA[c] = sanitize_series(combA[c])
    A_blocks["A_service_yes_count"] = (combA[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
    A_blocks["A_service_no_count"] = (combA[SERVICE_COLS] == "No").sum(axis=1).astype("float32")
    A_blocks["A_service_other_count"] = (combA[SERVICE_COLS].isin(["No phone service", "No internet service"])).sum(axis=1).astype("float32")
    A_blocks["A_service_count"] = A_blocks["A_service_yes_count"].astype("float32")
    A_blocks["A_has_internet"] = (combA["InternetService"] != "No").astype("float32")
    A_blocks["A_has_phone"] = (combA["PhoneService"] == "Yes").astype("float32")

    for col in A_NUMS:
        v = pd.to_numeric(pd.concat([trA[col], teA[col]], axis=0, ignore_index=True), errors="coerce")
        A_blocks[f"A_MISS_{col}"] = v.isna().astype("int8")

    for c in YN_COLS:
        s = sanitize_series(pd.concat([trA[c], teA[c]], axis=0, ignore_index=True))
        A_blocks[f"A_ISYES_{c}"] = (s == "Yes").astype("float32")
        A_blocks[f"A_ISNO_{c}"] = (s == "No").astype("float32")
        A_blocks[f"A_ISOTHER_{c}"] = (~s.isin(["Yes", "No"])).astype("float32")
    A_num_df = _reset(pd.DataFrame(A_blocks))

    A_te_extra = {}
    ten = pd.to_numeric(pd.concat([trA["tenure"], teA["tenure"]], axis=0, ignore_index=True), errors="coerce")
    A_te_extra["A_tenure_bin_te"] = pd.cut(ten, bins=TENURE_BINS, include_lowest=True).astype(str)

    mc_all = pd.concat([trA["MonthlyCharges"], teA["MonthlyCharges"], orA["MonthlyCharges"]], axis=0, ignore_index=True)
    mc_bins = pd.qcut(mc_all, q=40, retbins=True, duplicates="drop")[1]
    mc_tt = pd.concat([trA["MonthlyCharges"], teA["MonthlyCharges"]], axis=0, ignore_index=True)
    A_te_extra["A_MonthlyCharges_bin_te"] = pd.cut(mc_tt, bins=mc_bins, include_lowest=True).astype(str)

    tc_all = pd.concat([trA["TotalCharges"], teA["TotalCharges"], orA["TotalCharges"]], axis=0, ignore_index=True)
    tc_bins = pd.qcut(tc_all, q=60, retbins=True, duplicates="drop")[1]
    tc_tt = pd.concat([trA["TotalCharges"], teA["TotalCharges"]], axis=0, ignore_index=True)
    A_te_extra["A_TotalCharges_bin_te"] = pd.cut(tc_tt, bins=tc_bins, include_lowest=True).astype(str)

    for a, b in CROSS_PAIRS:
        aa = sanitize_series(pd.concat([trA[a], teA[a]], axis=0, ignore_index=True))
        bb = sanitize_series(pd.concat([trA[b], teA[b]], axis=0, ignore_index=True))
        A_te_extra[f"A_{a}__{b}_te"] = (aa + "|" + bb).astype(str)
    for a, b, c in TRIPLE:
        aa = sanitize_series(pd.concat([trA[a], teA[a]], axis=0, ignore_index=True))
        bb = sanitize_series(pd.concat([trA[b], teA[b]], axis=0, ignore_index=True))
        cc = sanitize_series(pd.concat([trA[c], teA[c]], axis=0, ignore_index=True))
        A_te_extra[f"A_{a}__{b}__{c}_te"] = (aa + "|" + bb + "|" + cc).astype(str)
    A_te_df = _reset(pd.DataFrame(A_te_extra))

    A_orig_map = {}
    for col in A_CATS + A_NUMS:
        if col in A_CATS:
            key_o = sanitize_series(orig[col])
            key_tt = sanitize_series(pd.concat([trA[col], teA[col]], axis=0, ignore_index=True))
        else:
            key_o = pd.to_numeric(orig[col], errors="coerce")
            key_tt = pd.to_numeric(pd.concat([trA[col], teA[col]], axis=0, ignore_index=True), errors="coerce")
        tmp = pd.DataFrame({"k": key_o, "y": orig[TARGET].values}).groupby("k", dropna=False)["y"].mean()
        A_orig_map[f"A_ORIG_proba_{col}"] = key_tt.map(tmp).fillna(0.5).astype("float32")
    A_orig_df = _reset(pd.DataFrame(A_orig_map))

    cat_for_counts = {}
    for c in A_CATS:
        cat_for_counts[c] = sanitize_series(pd.concat([trA[c], teA[c]], axis=0, ignore_index=True))
    for c in A_te_df.columns:
        cat_for_counts[c] = sanitize_series(A_te_df[c])
    cat_for_counts = pd.DataFrame(cat_for_counts)
    orig_cats_df = sanitize_str_df(orig[[c for c in A_CATS if c in orig.columns]].copy(), fill="missing")
    A_cnt = {}
    for c in cat_for_counts.columns:
        all_s = pd.concat([cat_for_counts[c], orig_cats_df[c] if c in orig_cats_df.columns else pd.Series([], dtype="string")],
                          axis=0, ignore_index=True)
        vc = all_s.value_counts(dropna=False)
        cnt = cat_for_counts[c].map(vc).fillna(0).astype("float32")
        A_cnt[f"A_CAT_CNT_{c}"] = cnt
        A_cnt[f"A_CAT_RARE_{c}"] = (cnt <= 50).astype("float32")
    A_cnt_df = _reset(pd.DataFrame(A_cnt))

    return [A_num_df, A_te_df, A_orig_df, A_cnt_df]


def union_blocks_b(train, test, orig):
    trB = train.copy()
    teB = test.copy()
    orB = orig.copy()
    y_orig = orB[TARGET].values.astype(np.int32)

    B_NUMS = [c for c in NUMS_CAND if c in trB.columns and c in teB.columns and c in orB.columns]
    B_CATS = [c for c in trB.columns if c not in [TARGET] + B_NUMS]
    for df in (trB, teB, orB):
        for c in B_CATS:
            df[c] = sanitize_series(df[c])

    trB, teB, orB, B_QCUT = add_qcut_bins_3way(trB, teB, orB, B_NUMS, Q_LIST)
    trB, teB, orB, B_ROUND = add_round_features_3way(trB, teB, orB, ROUND_SPECS)
    trB, teB, orB, B_DIGITS = add_digit_features_3way(trB, teB, orB, DIGIT_SPECS)

    B_NUM_CAT = []
    for c in B_NUMS:
        n = f"B_NUMCAT_{c}"
        for df in (trB, teB, orB):
            df[n] = pd.to_numeric(df[c], errors="coerce")
        B_NUM_CAT.append(n)

    B_NEW_NUM_CAT = []
    for r in [1, 0, -1]:
        n = f"B_MonthlyCharges_round_{r}"
        trB[n] = pd.to_numeric(trB["MonthlyCharges"], errors="coerce").round(r)
        teB[n] = pd.to_numeric(teB["MonthlyCharges"], errors="coerce").round(r)
        orB[n] = pd.to_numeric(orB["MonthlyCharges"], errors="coerce").round(r)
        B_NEW_NUM_CAT.append(n)
    for r in [1, 0, -1]:
        n = f"B_TotalCharges_round_{r}"
        trB[n] = pd.to_numeric(trB["TotalCharges"], errors="coerce").round(r)
        teB[n] = pd.to_numeric(teB["TotalCharges"], errors="coerce").round(r)
        orB[n] = pd.to_numeric(orB["TotalCharges"], errors="coerce").round(r)
        B_NEW_NUM_CAT.append(n)
    for base in ["MonthlyCharges", "TotalCharges"]:
        n = f"B_{base}_frac2"
        for df in (trB, teB, orB):
            x = pd.to_numeric(df[base], errors="coerce").astype("float64")
            df[n] = (np.floor((x * 100.0) % 100.0)).fillna(-1).astype("int16")
        B_NEW_NUM_CAT.append(n)
    B_NUM_CAT = B_NUM_CAT + B_NEW_NUM_CAT

    trB, teB, orB = factorize_across_all_3way(trB, teB, orB, B_CATS + B_NUM_CAT)
    trB, teB, B_DRIFT = add_drift_ratio(trB, teB, orB, cols=(B_CATS + B_NUM_CAT), eps=DRIFT_EPS)
    trB, teB, B_TEORIG, B_CEORIG = add_orig_te_ce(trB, teB, orB, B_CATS + B_NUM_CAT + B_DIGITS, y_orig)
    trB, teB, B_GM = add_group_means(trB, teB, group_cols=B_CATS, num_cols=B_NUMS)

    B_attach_cols = B_QCUT + B_ROUND + B_DIGITS + B_NUM_CAT + B_DRIFT + B_TEORIG + B_CEORIG + B_GM
    B_attach_df = pd.concat(
        [trB[B_attach_cols].reset_index(drop=True), teB[B_attach_cols].reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )
    return [_reset(B_attach_df)]


# -----------------------------
# Distribution / digit / n-gram blocks
# -----------------------------
def distribution_digit_ngram_blocks(combined, orig):
    orig_tc_all = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(orig["TotalCharges"].median()).values
    orig_churner_tc = pd.to_numeric(orig.loc[orig[TARGET] == 1, "TotalCharges"], errors="coerce").dropna().values
    orig_nonchurner_tc = pd.to_numeric(orig.loc[orig[TARGET] == 0, "TotalCharges"], errors="coerce").dropna().values
    orig_is_mc_mean = orig.groupby("InternetService", dropna=False)["MonthlyCharges"].mean()

    tc_comb = pd.to_numeric(combined["TotalCharges"], errors="coerce").fillna(0).values
    mc_comb = pd.to_numeric(combined["MonthlyCharges"], errors="coerce").fillna(0).values

    dist = {}
    dist["X_pctrank_nonchurner_TC"] = pctrank_against(tc_comb, orig_nonchurner_tc)
    dist["X_pctrank_churner_TC"] = pctrank_against(tc_comb, orig_churner_tc)
    dist["X_pctrank_orig_TC"] = pctrank_against(tc_comb, orig_tc_all)
    dist["X_zscore_nonchurner_TC"] = zscore_against(tc_comb, orig_nonchurner_tc)
    dist["X_zscore_churn_gap_TC"] = (
        np.abs(zscore_against(tc_comb, orig_churner_tc))
        - np.abs(zscore_against(tc_comb, orig_nonchurner_tc))
    ).astype("float32")
    dist["X_pctrank_churn_gap_TC"] = (
        pctrank_against(tc_comb, orig_churner_tc)
        - pctrank_against(tc_comb, orig_nonchurner_tc)
    ).astype("float32")
    dist["X_resid_IS_MC"] = (
        mc_comb - combined["InternetService"].map(orig_is_mc_mean).fillna(0).values
    ).astype("float32")

    cond_is = np.zeros(len(combined), dtype="float32")
    for cat_val in orig["InternetService"].dropna().unique():
        mask = (combined["InternetService"] == cat_val).values
        ref = pd.to_numeric(orig.loc[orig["InternetService"] == cat_val, "TotalCharges"], errors="coerce").dropna().values
        if len(ref) > 0 and mask.sum() > 0:
            cond_is[mask] = pctrank_against(pd.to_numeric(combined.loc[mask, "TotalCharges"], errors="coerce").fillna(0).values, ref)
    dist["X_cond_pctrank_IS_TC"] = cond_is

    cond_c = np.zeros(len(combined), dtype="float32")
    for cat_val in orig["Contract"].dropna().unique():
        mask = (combined["Contract"] == cat_val).values
        ref = pd.to_numeric(orig.loc[orig["Contract"] == cat_val, "TotalCharges"], errors="coerce").dropna().values
        if len(ref) > 0 and mask.sum() > 0:
            cond_c[mask] = pctrank_against(pd.to_numeric(combined.loc[mask, "TotalCharges"], errors="coerce").fillna(0).values, ref)
    dist["X_cond_pctrank_C_TC"] = cond_c

    tc_num = pd.to_numeric(combined["TotalCharges"], errors="coerce").fillna(0)
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        dist[f"X_dist_To_ch_{q_label}"] = np.abs(tc_num - ch_q).astype("float32")
        dist[f"X_dist_To_nc_{q_label}"] = np.abs(tc_num - nc_q).astype("float32")
        dist[f"X_qdist_gap_To_{q_label}"] = (np.abs(tc_num - nc_q) - np.abs(tc_num - ch_q)).astype("float32")
    dist_df = _reset(pd.DataFrame(dist))

    tenure_s = pd.to_numeric(combined["tenure"], errors="coerce").fillna(0)
    mc_s = pd.to_numeric(combined["MonthlyCharges"], errors="coerce").fillna(0)
    tc_s = pd.to_numeric(combined["TotalCharges"], errors="coerce").fillna(0)
    t_str = tenure_s.astype(int).astype(str)
    mc_floor = np.floor(mc_s).astype(int)
    tc_floor = np.floor(tc_s).astype(int)
    mc_str = mc_s.astype(str).str.replace(".", "", regex=False)
    tc_str = tc_s.astype(str).str.replace(".", "", regex=False)

    digit_explicit = {
        "X_tenure_first_digit": t_str.str[0].astype(int),
        "X_tenure_last_digit": t_str.str[-1].astype(int),
        "X_tenure_second_digit": t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0).astype("int16"),
        "X_tenure_mod10": (tenure_s % 10).astype("float32"),
        "X_tenure_mod12": (tenure_s % 12).astype("float32"),
        "X_tenure_num_digits": t_str.str.len().astype("int16"),
        "X_tenure_is_multiple_10": (tenure_s % 10 == 0).astype("float32"),
        "X_tenure_rounded_10": (np.round(tenure_s / 10) * 10).astype("float32"),
        "X_tenure_dev_from_round10": np.abs(tenure_s - np.round(tenure_s / 10) * 10).astype("float32"),

        "X_mc_first_digit": mc_str.str[0].replace("-", "0").fillna("0").astype(int),
        "X_mc_last_digit": mc_str.str[-1].replace("-", "0").fillna("0").astype(int),
        "X_mc_second_digit": mc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype("int16"),
        "X_mc_mod10": (mc_floor % 10).astype("float32"),
        "X_mc_mod100": (mc_floor % 100).astype("float32"),
        "X_mc_num_digits": mc_floor.astype(str).str.len().astype("int16"),
        "X_mc_is_multiple_10": (mc_floor % 10 == 0).astype("float32"),
        "X_mc_is_multiple_50": (mc_floor % 50 == 0).astype("float32"),
        "X_mc_rounded_10": (np.round(mc_s / 10) * 10).astype("float32"),
        "X_mc_fractional": (mc_s - np.floor(mc_s)).astype("float32"),
        "X_mc_dev_from_round10": np.abs(mc_s - np.round(mc_s / 10) * 10).astype("float32"),

        "X_tc_first_digit": tc_str.str[0].replace("-", "0").fillna("0").astype(int),
        "X_tc_last_digit": tc_str.str[-1].replace("-", "0").fillna("0").astype(int),
        "X_tc_second_digit": tc_str.apply(lambda x: int(x[1]) if len(x) > 1 and x[1].isdigit() else 0).astype("int16"),
        "X_tc_mod10": (tc_floor % 10).astype("float32"),
        "X_tc_mod100": (tc_floor % 100).astype("float32"),
        "X_tc_num_digits": tc_floor.astype(str).str.len().astype("int16"),
        "X_tc_is_multiple_10": (tc_floor % 10 == 0).astype("float32"),
        "X_tc_is_multiple_100": (tc_floor % 100 == 0).astype("float32"),
        "X_tc_rounded_100": (np.round(tc_s / 100) * 100).astype("float32"),
        "X_tc_fractional": (tc_s - np.floor(tc_s)).astype("float32"),
        "X_tc_dev_from_round100": np.abs(tc_s - np.round(tc_s / 100) * 100).astype("float32"),

        "X_tenure_years": (tenure_s // 12).astype("float32"),
        "X_tenure_months_in_year": (tenure_s % 12).astype("float32"),
        "X_mc_per_digit": (mc_s / (mc_floor.astype(str).str.len().astype(float) + 0.001)).astype("float32"),
        "X_tc_per_digit": (tc_s / (tc_floor.astype(str).str.len().astype(float) + 0.001)).astype("float32"),
    }
    digit_df = _reset(pd.DataFrame(digit_explicit))

    ngram_keys = {}
    for col in A_NUMS:
        ngram_keys[f"X_CAT_{col}_te"] = pd.to_numeric(combined[col], errors="coerce").fillna(-1).astype(str)
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        ngram_keys[f"X_BG_{c1}_{c2}_te"] = (sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2])).astype(str)
    for c1, c2, c3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        ngram_keys[f"X_TG_{c1}_{c2}_{c3}_te"] = (
            sanitize_series(combined[c1]) + "_" + sanitize_series(combined[c2]) + "_" + sanitize_series(combined[c3])
        ).astype(str)
    ngram_df = _reset(pd.DataFrame(ngram_keys))

    return [dist_df, digit_df, ngram_df]


# -----------------------------
# Assembly
# -----------------------------
def build_features(union_blocks=False, extra_blocks=False):
    train, test, orig = load_data()
    y = train[TARGET].values
    X = train.drop(columns=[TARGET]).copy()
    combined = pd.concat([_reset(X), _reset(test)], axis=0, ignore_index=True)
    n_train = len(X)

    dfs = [combined] + base_blocks(combined, orig)
    if union_blocks:
        dfs += union_blocks_a(train, test, orig)
        dfs += union_blocks_b(train, test, orig)
    if extra_blocks:
        dfs += distribution_digit_ngram_blocks(combined, orig)
    assert all(d.index.is_unique for d in dfs)
    assert all(len(d) == len(combined) for d in dfs)

    combined_fe = pd.concat(dfs, axis=1)
    combined_fe = combined_fe.loc[:, ~combined_fe.columns.duplicated()].copy()
    cat_cols = combined_fe.select_dtypes("object").columns.tolist()
    str_cols = combined_fe.select_dtypes(include=["string"]).columns.tolist()
    cat_cols = sorted(set(cat_cols + str_cols))
    if len(cat_cols) > 0:
        combined_fe[cat_cols] = combined_fe[cat_cols].astype("category")

    X_fe = combined_fe.iloc[:n_train].copy()
    T_fe = combined_fe.iloc[n_train:].copy()
    print(f"features: {X_fe.shape[1]} columns, {sum(c.endswith('_te') for c in X_fe.columns)} to encode, "
          f"{len(cat_cols)} categories", flush=True)
    return X_fe, T_fe, y
