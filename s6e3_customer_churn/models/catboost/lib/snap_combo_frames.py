"""Feature frames of train_snap_combo.py: the label-free snap / decimal-string columns of models/snap_features merged
onto the raw data, plus snap / frequency / rank / transform / relation / service / bin / cross / digit / n-gram /
count / group-mean / original-prior / drift blocks built over train + test, and original class-centroid distances."""
import os
import random
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from common.data import ID, TARGET, cache_dir, read_original, read_test, read_train

warnings.filterwarnings('ignore')

ID_COL = ID
SEED = 42
N_FOLDS = 5
EPS = 1e-6

BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
BASE_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
YN_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines",
]
PAIR_COLS = [
    ("Contract", "InternetService"),
    ("Contract", "PaymentMethod"),
    ("Contract", "PaperlessBilling"),
    ("InternetService", "PaymentMethod"),
    ("InternetService", "OnlineSecurity"),
    ("InternetService", "TechSupport"),
    ("InternetService", "DeviceProtection"),
    ("PaymentMethod", "SeniorCitizen"),
    ("tenure_bin", "Contract"),
    ("tenure_bin", "InternetService"),
    ("MonthlyCharges_bin", "Contract"),
    ("TotalCharges_bin", "Contract"),
    ("mc0", "Contract"),
    ("mc1", "PaymentMethod"),
    ("tc0", "Contract"),
    ("tc1", "InternetService"),
]
TRIPLE_COLS = [
    ("Contract", "InternetService", "PaymentMethod"),
    ("PaperlessBilling", "Contract", "PaymentMethod"),
    ("InternetService", "OnlineSecurity", "TechSupport"),
    ("tenure_bin", "Contract", "InternetService"),
]
TOP_CATS_FOR_NGRAMS = [
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
    "OnlineSecurity", "TechSupport", "DeviceProtection", "StreamingMovies",
    "StreamingTV", "MultipleLines",
]
TENURE_BINS = [-1, 0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10**9]
NATIVE_COMBO_KEYWORDS = [
    "charges_", "monthly_", "total_", "service_", "ORIG_", "DRIFT_", "AUX_",
    "RANK_", "FREQ_", "MEAN_", "DIFF_", "COUNT_", "CAT_CNT_", "CAT_RARE_",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_binary_target(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int).values
    m = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0, "1": 1, "0": 0}
    return s.astype("string").map(m).astype("float32").fillna(0).astype(int).values


def clean_totalcharges_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.strip().replace({"": np.nan, " ": np.nan}), errors="coerce")


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def _coerce_mixed_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) == "object":
            try:
                out[c] = out[c].astype("string")
            except Exception:
                pass
    return out


def load_combo_features(n_train: int, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The snap / decimal-string frame of train + test rows written by models/snap_features."""
    path = cache_dir("snap_features") / "combo.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run models/snap_features first")
    combo = pd.read_csv(path, low_memory=False)
    combo = _ensure_unique_columns(combo)
    combo = _coerce_mixed_object_columns(combo)
    if len(combo) != n_train + n_test:
        raise ValueError(f"combo.csv length mismatch: got {len(combo)}, expected {n_train + n_test}")
    combo = combo.reset_index(drop=True)
    ctrain = combo.iloc[:n_train].reset_index(drop=True).copy()
    ctest = combo.iloc[n_train:].reset_index(drop=True).copy()
    return ctrain, ctest


def _safe_qcut_edges(s: pd.Series, q: int):
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    nunique = int(x.nunique(dropna=True))
    qq = max(2, min(q, max(2, nunique))) if nunique else 2
    try:
        bins = pd.qcut(x, q=qq, duplicates="drop", retbins=True)[1]
    except Exception:
        xx = x.fillna(x.median() if x.notna().any() else 0)
        bins = np.unique(np.nanpercentile(xx, np.linspace(0, 100, qq + 1)))
    bins = np.unique(bins)
    if len(bins) < 2:
        lo = float(x.min()) if x.notna().any() else 0.0
        hi = float(x.max()) if x.notna().any() else 1.0
        if lo == hi:
            hi = lo + 1.0
        bins = np.array([lo - 1e-9, hi + 1e-9], dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    return bins


def _series(df: pd.DataFrame, c: str, default=None) -> pd.Series:
    if c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    if default is None:
        default = np.nan
    return pd.Series([default] * len(df), index=df.index)


def _safe_concat(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in right.columns if c not in left.columns]
    return pd.concat([left.reset_index(drop=True), right[keep].reset_index(drop=True)], axis=1)


def read_data():
    train, test, orig = read_train(), read_test(), read_original()
    for df in (train, test, orig):
        df["TotalCharges"] = clean_totalcharges_series(df["TotalCharges"])
        if df["TotalCharges"].isna().any():
            med = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(0 if pd.isna(med) else med)
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype("int8").astype("string")
        )
        for c in BASE_CATS:
            df[c] = df[c].astype("string").fillna("NA")

    combo_tr, combo_te = load_combo_features(len(train), len(test))
    train = _safe_concat(train, combo_tr)
    test = _safe_concat(test, combo_te)

    y = to_binary_target(train[TARGET])
    y_orig = to_binary_target(orig[TARGET])
    return train, test, orig, y, y_orig


def add_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    if "MonthlyCharges" in df.columns:
        mc = pd.to_numeric(_series(df, "MonthlyCharges"), errors="coerce").fillna(0)
        out["MonthlyCharges_snap"] = mc.round(0).astype("float32")
        out["MonthlyCharges_floor"] = np.floor(mc).astype("float32")
        out["MonthlyCharges_frac"] = (mc - np.floor(mc)).astype("float32")
    if "TotalCharges" in df.columns:
        tc = pd.to_numeric(_series(df, "TotalCharges"), errors="coerce").fillna(0)
        out["TotalCharges_snap"] = tc.round(0).astype("float32")
        out["TotalCharges_floor"] = np.floor(tc).astype("float32")
        out["TotalCharges_frac"] = (tc - np.floor(tc)).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_freq_rank_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(_series(df, c), errors="coerce").astype("float32")
        vc = s.value_counts(normalize=True, dropna=False)
        out[f"FREQ_{c}"] = s.map(vc).fillna(0).astype("float32")
        out[f"RANK_{c}"] = s.rank(method="average", pct=True).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_num_transforms(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(_series(df, c), errors="coerce").astype("float32")
        vv = v.fillna(0)
        out[f"LOG1P_{c}"] = np.log1p(vv.clip(lower=0)).astype("float32")
        out[f"SQRT_{c}"] = np.sqrt(vv.clip(lower=0)).astype("float32")
        out[f"INV1P_{c}"] = (1.0 / (1.0 + vv.clip(lower=0))).astype("float32")
        out[f"MISS_{c}"] = v.isna().astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_charge_relationships(df: pd.DataFrame) -> pd.DataFrame:
    ten = pd.to_numeric(_series(df, "tenure"), errors="coerce").fillna(0).astype("float32")
    mc = pd.to_numeric(_series(df, "MonthlyCharges"), errors="coerce").fillna(0).astype("float32")
    tc = pd.to_numeric(_series(df, "TotalCharges"), errors="coerce").fillna(0).astype("float32")
    out = {}
    expected = ten * mc
    out["charges_deviation"] = (tc - expected).astype("float32")
    out["abs_charges_dev"] = np.abs(out["charges_deviation"]).astype("float32")
    out["monthly_to_total_ratio"] = (mc / (tc.abs() + 1.0)).astype("float32")
    out["total_to_monthly_ratio"] = (tc / (mc.abs() + 1.0)).astype("float32")
    out["avg_monthly_charges"] = (tc / (ten + 1.0)).astype("float32")
    out["tenure_x_monthly"] = (ten * mc).astype("float32")
    out["tenure_x_total"] = (ten * tc).astype("float32")
    out["monthly_minus_avg"] = (mc - out["avg_monthly_charges"]).astype("float32")
    out["charge_dev_ratio"] = (out["charges_deviation"] / (tc.abs() + 1.0)).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_service_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    valid = [c for c in SERVICE_COLS if c in df.columns]
    if not valid:
        return pd.DataFrame(index=df.index)
    tmp = df[valid].astype("string").fillna("NA")
    out["service_yes_count"] = (tmp == "Yes").sum(axis=1).astype("float32")
    out["service_no_count"] = (tmp == "No").sum(axis=1).astype("float32")
    out["service_other_count"] = tmp.isin(["No phone service", "No internet service"]).sum(axis=1).astype("float32")
    out["service_balance"] = (out["service_yes_count"] - out["service_no_count"]).astype("float32")
    if "InternetService" in df.columns:
        out["has_internet"] = (_series(df, "InternetService").astype("string") != "No").astype("float32")
    if "PhoneService" in df.columns:
        out["has_phone"] = (_series(df, "PhoneService").astype("string") == "Yes").astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_yes_no_other_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in YN_COLS:
        if c not in df.columns:
            continue
        s = _series(df, c).astype("string").fillna("NA")
        out[f"FLAG_YES_{c}"] = (s == "Yes").astype("float32")
        out[f"FLAG_NO_{c}"] = (s == "No").astype("float32")
        out[f"FLAG_OTHER_{c}"] = (~s.isin(["Yes", "No"])).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_bins_and_cross_cats(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "tenure" in df.columns:
        out["tenure_bin"] = pd.cut(pd.to_numeric(_series(df, "tenure"), errors="coerce"), bins=TENURE_BINS,
                                   include_lowest=True).astype("string")
    if "MonthlyCharges" in df.columns:
        s = pd.to_numeric(_series(df, "MonthlyCharges"), errors="coerce")
        out["MonthlyCharges_bin"] = pd.cut(s, bins=_safe_qcut_edges(s, 14), include_lowest=True).astype("string")
    if "TotalCharges" in df.columns:
        s = pd.to_numeric(_series(df, "TotalCharges"), errors="coerce").fillna(-1)
        out["TotalCharges_bin"] = pd.cut(s, bins=_safe_qcut_edges(s, 14), include_lowest=True).astype("string")
    if "MonthlyCharges_snap" in df.columns:
        s = pd.to_numeric(_series(df, "MonthlyCharges_snap"), errors="coerce")
        out["MonthlyCharges_snap_bin"] = pd.cut(s, bins=_safe_qcut_edges(s, 14), include_lowest=True).astype("string")
    if "TotalCharges_snap" in df.columns:
        s = pd.to_numeric(_series(df, "TotalCharges_snap"), errors="coerce").fillna(-1)
        out["TotalCharges_snap_bin"] = pd.cut(s, bins=_safe_qcut_edges(s, 14), include_lowest=True).astype("string")

    def src(col):
        if col in out.columns:
            return _series(out, col).astype("string").fillna("NA")
        if col in df.columns:
            return _series(df, col).astype("string").fillna("NA")
        return None

    for a, b in PAIR_COLS:
        s1, s2 = src(a), src(b)
        if s1 is None or s2 is None:
            continue
        name = f"{a}__{b}"
        if name not in out.columns:
            out[name] = s1 + "__" + s2
    for a, b, c in TRIPLE_COLS:
        s1, s2, s3 = src(a), src(b), src(c)
        if s1 is None or s2 is None or s3 is None:
            continue
        name = f"{a}__{b}__{c}"
        if name not in out.columns:
            out[name] = s1 + "__" + s2 + "__" + s3
    return out


def add_round_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in ["MonthlyCharges", "TotalCharges", "MonthlyCharges_snap", "TotalCharges_snap"]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(_series(df, c), errors="coerce")
        out[f"ROUND0_{c}"] = s.round(0).astype("float32")
        out[f"ROUND1_{c}"] = s.round(1).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c, prefix in [("MonthlyCharges", "mc"), ("TotalCharges", "tc")]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(_series(df, c), errors="coerce").fillna(0)
        sf = s.map(lambda x: f"{float(x):.2f}")
        parts = sf.str.split(".", n=1, expand=True)
        left = parts[0].str.replace("-", "", regex=False).str.zfill(4)
        right = parts[1].str.pad(2, side="right", fillchar="0")
        vals = {
            f"{prefix}0": left.str[-1],
            f"{prefix}1": left.str[-2],
            f"{prefix}2": left.str[-3],
            f"{prefix}f0": right.str[0],
            f"{prefix}f1": right.str[1],
            f"{prefix}_digit_bucket": left.str[-2:] + "_" + right.str[:1],
        }
        for k, v in vals.items():
            if k not in df.columns:
                out[k] = v.astype("string")
    return pd.DataFrame(out, index=df.index)


def add_cat_counts_and_rare(df: pd.DataFrame, cat_cols) -> pd.DataFrame:
    out = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = _series(df, c).astype("string").fillna("NA")
        vc = s.value_counts(dropna=False)
        out[f"CAT_CNT_{c}"] = s.map(vc).fillna(0).astype("float32")
        out[f"CAT_RARE_{c}"] = (s.map(vc).fillna(0) <= 20).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_count_features_combined(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = _series(df, c).astype("string").fillna("NA")
        vc = s.value_counts(dropna=False)
        out[f"COUNT_{c}"] = s.map(vc).fillna(0).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_groupby_means(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    nums = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]
    groups = [c for c in ["Contract", "InternetService", "PaymentMethod", "tenure_bin"] if c in df.columns]
    for g in groups:
        gs = _series(df, g).astype("string").fillna("NA")
        for n in nums:
            x = pd.to_numeric(_series(df, n), errors="coerce").astype("float32")
            m = x.groupby(gs).transform("mean")
            out[f"MEAN_{n}_BY_{g}"] = m.astype("float32")
            out[f"DIFF_{n}_BY_{g}"] = (x - m).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_orig_priors(df: pd.DataFrame, orig: pd.DataFrame, y_orig, cat_cols) -> pd.DataFrame:
    if y_orig is None or len(orig) == 0:
        return pd.DataFrame(index=df.index)
    tmp = orig.copy()
    tmp["_target_"] = y_orig
    out = {}
    global_mean = float(np.mean(y_orig))
    for c in cat_cols:
        if c not in df.columns or c not in tmp.columns:
            continue
        sref = tmp[c].astype("string").fillna("NA")
        stats = tmp.groupby(sref)["_target_"].agg(["mean", "count"])
        m = _series(df, c).astype("string").fillna("NA").map(stats["mean"]).fillna(global_mean).astype("float32")
        cnt = _series(df, c).astype("string").fillna("NA").map(stats["count"]).fillna(0).astype("float32")
        out[f"ORIG_PRIOR_{c}"] = m
        out[f"ORIG_CNT_{c}"] = cnt
    for n in [c for c in BASE_NUMS if c in df.columns and c in tmp.columns]:
        ref = pd.to_numeric(tmp[n], errors="coerce").astype("float32")
        vals = pd.to_numeric(_series(df, n), errors="coerce").astype("float32")
        m0 = ref[y_orig == 0].mean() if np.any(y_orig == 0) else 0.0
        m1 = ref[y_orig == 1].mean() if np.any(y_orig == 1) else 0.0
        s = ref.std() if ref.std() > 1e-8 else 1.0
        out[f"ORIG_DIST0_{n}"] = ((vals - m0) / s).astype("float32")
        out[f"ORIG_DIST1_{n}"] = ((vals - m1) / s).astype("float32")
        out[f"ORIG_GAP_{n}"] = (np.abs(vals - m1) - np.abs(vals - m0)).astype("float32")
    return pd.DataFrame(out, index=df.index)


def add_drift_ratio(df_all: pd.DataFrame, n_train: int, cat_cols) -> pd.DataFrame:
    tr = df_all.iloc[:n_train].copy()
    te = df_all.iloc[n_train:].copy()
    out = pd.DataFrame(index=df_all.index)
    for c in cat_cols:
        if c not in df_all.columns:
            continue
        trc = _series(tr, c).astype("string").fillna("NA")
        tec = _series(te, c).astype("string").fillna("NA")
        ftr = trc.value_counts(normalize=True, dropna=False)
        fte = tec.value_counts(normalize=True, dropna=False)
        allv = _series(df_all, c).astype("string").fillna("NA")
        out[f"DRIFT_RATIO_{c}"] = (
            (allv.map(ftr).fillna(0.0) + EPS) / (allv.map(fte).fillna(0.0) + EPS)
        ).astype("float32")
    return out


def add_orig_aux_model_features(tr: pd.DataFrame, te: pd.DataFrame, orig_x: pd.DataFrame, y_orig):
    """Distances to the original churner / non-churner centroids of the standardized numerics and a logistic score."""
    if y_orig is None or len(orig_x) == 0:
        return pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in tr.columns and c in orig_x.columns]
    if not num_cols:
        return pd.DataFrame(index=tr.index), pd.DataFrame(index=te.index)

    ref = orig_x.copy()
    for c in num_cols:
        ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0).astype("float32")
    mu0 = (ref.loc[y_orig == 0, num_cols].mean().values.astype("float32") if np.any(y_orig == 0)
           else np.zeros(len(num_cols), dtype="float32"))
    mu1 = (ref.loc[y_orig == 1, num_cols].mean().values.astype("float32") if np.any(y_orig == 1)
           else np.zeros(len(num_cols), dtype="float32"))
    sd = ref[num_cols].std().replace(0, 1).fillna(1).values.astype("float32")

    def build(df):
        X = pd.DataFrame(index=df.index)
        vals = np.column_stack([
            pd.to_numeric(_series(df, c), errors="coerce").fillna(0).astype("float32").values for c in num_cols
        ])
        z0 = ((vals - mu0) / sd)
        z1 = ((vals - mu1) / sd)
        d0 = np.sqrt((z0 ** 2).sum(axis=1))
        d1 = np.sqrt((z1 ** 2).sum(axis=1))
        X["AUX_dist0"] = d0.astype("float32")
        X["AUX_dist1"] = d1.astype("float32")
        X["AUX_gap10"] = (d1 - d0).astype("float32")
        score = 1 / (1 + np.exp(np.clip(d1 - d0, -20, 20)))
        X["AUX_prob_like"] = score.astype("float32")
        return X

    return build(tr), build(te)


def build_feature_frames(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, y_orig_bin: np.ndarray):
    n_train, n_test = len(train), len(test)
    tr_base = train.drop(columns=[TARGET], errors="ignore").reset_index(drop=True).copy()
    te_base = test.reset_index(drop=True).copy()

    combined = pd.concat([tr_base.copy(), te_base.copy()], axis=0, ignore_index=True)
    combined = _ensure_unique_columns(combined)
    for c in BASE_CATS:
        if c in combined.columns:
            combined[c] = combined[c].astype("string").fillna("NA")

    blocks = []

    snap = add_snap_features(combined)
    combined = _safe_concat(combined, snap)
    blocks.append(snap)

    num_for_transforms = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "MonthlyCharges_snap",
                                      "TotalCharges_snap"] if c in combined.columns]
    blocks.append(add_freq_rank_features(combined, num_for_transforms))
    blocks.append(add_num_transforms(combined, num_for_transforms))
    blocks.append(add_charge_relationships(combined))
    blocks.append(add_service_counts(combined))
    blocks.append(add_yes_no_other_flags(combined))

    bins = add_bins_and_cross_cats(combined)
    combined = _safe_concat(combined, bins)
    blocks.append(bins)

    rounds = add_round_features(combined)
    digits = add_digit_features(combined)
    combined = _safe_concat(combined, digits)
    blocks.append(rounds)
    blocks.append(digits)

    ngram_cols = []
    top = [c for c in TOP_CATS_FOR_NGRAMS if c in combined.columns][:8]
    for a, b in combinations(top, 2):
        name = f"NGRAM_{a}__{b}"
        if name not in combined.columns:
            combined[name] = (_series(combined, a).astype("string").fillna("NA") + "__"
                              + _series(combined, b).astype("string").fillna("NA"))
        ngram_cols.append(name)
    top3 = [c for c in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"] if c in combined.columns]
    for a, b, c in combinations(top3, 3):
        name = f"NGRAM_{a}__{b}__{c}"
        if name not in combined.columns:
            combined[name] = (_series(combined, a).astype("string").fillna("NA") + "__"
                              + _series(combined, b).astype("string").fillna("NA") + "__"
                              + _series(combined, c).astype("string").fillna("NA"))
        ngram_cols.append(name)

    cat_like_cols = []
    for c in combined.columns:
        dt = str(combined[c].dtype)
        if (dt in ("object", "string", "category") or c in BASE_CATS or c.endswith("_bin") or "__" in c
                or c.startswith("NGRAM_")):
            cat_like_cols.append(c)
    cat_like_cols = list(dict.fromkeys(cat_like_cols))

    blocks.append(add_cat_counts_and_rare(combined, cat_like_cols))
    blocks.append(add_count_features_combined(combined, cat_like_cols))
    blocks.append(add_groupby_means(combined))
    blocks.append(add_orig_priors(combined, orig.drop(columns=[TARGET], errors="ignore").copy(), y_orig_bin,
                                  [c for c in cat_like_cols if c in orig.columns]))
    blocks.append(add_drift_ratio(combined, n_train, [c for c in cat_like_cols if c in combined.columns][:120]))

    feat = pd.DataFrame(index=combined.index)
    feat = _safe_concat(feat, combined)
    for b in blocks:
        feat = _safe_concat(feat, b)

    feat = _ensure_unique_columns(feat)
    tr = feat.iloc[:n_train].reset_index(drop=True).copy()
    te = feat.iloc[n_train:n_train + n_test].reset_index(drop=True).copy()
    return tr, te


def native_combo_columns(df: pd.DataFrame):
    """Categorical-like columns plus the relation / service / original / drift / centroid / rank / frequency /
    group-mean / count columns."""
    keep = []
    for c in [c for c in df.columns if c not in [TARGET, ID_COL]]:
        dt = str(df[c].dtype)
        if c in BASE_CATS or "__" in c or c.startswith("NGRAM_") or c.endswith("_bin") or dt in ("object", "string",
                                                                                                "category"):
            keep.append(c)
        elif any(k in c for k in NATIVE_COMBO_KEYWORDS):
            keep.append(c)
    return list(dict.fromkeys(keep))
