"""Features of the LightGBM model (lgbm_exact_te36).

Builds:
  1. raw + derived (target-free) features: 9 raw numerics (NaN kept) and 23 derived columns that propagate NaN (no
     imputation, so the exact-value tokens of the derived quantities stay clean);
  2. 18 original-dataset per-value rates (label rate and 'Severe' addiction-level rate of each numeric's exact value);
  3. 36 nested exact-value target encodings: 12 single columns (smoothing 30), 4 derived quantities (slack, comp_sum,
     sleep_daily: 30; utilization: 50), 5 columns at smoothing 10 and 200, 10 pairs (smoothing 100).
Target encodings per outer fold: outer-train rows from 5 inner folds (StratifiedKFold(5, shuffle=True,
random_state=777)), validation and test rows from the map of all outer-train rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import TARGET, folds as fold_ids, read_original, read_test, read_train

NUMERICS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CATEGORICALS = ["gender", "stress_level", "academic_work_impact"]


def load_data():
    train, test = read_train(), read_test()
    y = train[TARGET].values.astype(np.float64)
    return train, test, y, fold_ids(y)


def _tok(v):
    """Canonical exact-value token (avoids float-representation join bugs)."""
    if v.dtype == object or v.dtype == "string":
        return v.fillna("__NA__").astype(str)
    return v.map(lambda x: f"{x:.2f}" if pd.notna(x) else "__NA__")


def _te_map(tokens, labels, prior, smooth):
    g = pd.DataFrame({"t": tokens, "y": labels}).groupby("t")["y"].agg(["sum", "count"])
    return (g["sum"] + smooth * prior) / (g["count"] + smooth)


def _apply_map(tokens, mp, prior):
    s = pd.Series(tokens).map(mp)
    return s.fillna(prior).values.astype(np.float64)


def compute_derived(df):
    """Add derived (target-free) columns to a copy of df. Returns the copy."""
    df = df.copy()
    d = df["daily_screen_time_hours"]
    s = df["social_media_hours"]
    g = df["gaming_hours"]
    w = df["work_study_hours"]
    sl = df["sleep_hours"]
    n = df["notifications_per_day"]
    a = df["app_opens_per_day"]
    wk = df["weekend_screen_time"]

    comp_sum = s + g + w
    df["slack"] = d - comp_sum
    df["comp_sum"] = comp_sum
    df["utilization"] = comp_sum / d
    df["slack_cent"] = (df["slack"] * 100).round() % 100
    df["comp_sum_cent"] = (comp_sum * 100).round() % 100
    df["util_cent"] = (df["utilization"] * 100).round() % 100
    df["is_exact_budget"] = (np.abs(df["slack"]) < 1e-9).astype(np.int8)

    df["sleep_daily"] = sl + d
    df["at_20"] = (np.abs(df["sleep_daily"] - 20.0) < 1e-9).astype(np.int8)

    df["social_ratio"] = s / d
    df["gaming_ratio"] = g / d
    df["work_ratio"] = w / d
    df["notif_per_hour"] = n / d
    df["app_per_hour"] = a / d
    df["weekend_daily_diff"] = wk - d
    df["weekend_daily_ratio"] = wk / d

    df["notif_x_app"] = n * a
    df["social_x_gaming"] = s * g
    df["gaming_x_work"] = g * w

    # common-scale repair: components rescaled to ~85-95% utilization
    df["repair_flag"] = ((df["utilization"] >= 0.85) & (df["utilization"] <= 0.95)).astype(np.int8)

    return df


def build_raw_derived(train, test):
    """Target-free raw + derived features. Returns (Xtr, Xte) DataFrames.
    Derived features keep NaN (no imputation) so their exact-value tokens stay clean."""
    tr_d = compute_derived(train)
    te_d = compute_derived(test)
    for df in (tr_d, te_d):
        df["gender_ord"] = df["gender"].map({"Female": 0, "Male": 1, "Other": 2}).astype(np.float64)
        df["stress_ord"] = df["stress_level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(np.float64)
        df["impact_ord"] = df["academic_work_impact"].map({"No": 0, "Yes": 1}).astype(np.float64)

    derived_cols = [
        "slack", "comp_sum", "utilization", "slack_cent", "comp_sum_cent",
        "util_cent", "is_exact_budget", "sleep_daily", "at_20",
        "social_ratio", "gaming_ratio", "work_ratio", "notif_per_hour",
        "app_per_hour", "weekend_daily_diff", "weekend_daily_ratio",
        "notif_x_app", "social_x_gaming", "gaming_x_work",
        "repair_flag", "gender_ord", "stress_ord", "impact_ord",
    ]
    # raw numerics from the original (unimputed) frames
    Xtr = pd.concat([train[NUMERICS].reset_index(drop=True), tr_d[derived_cols].reset_index(drop=True)], axis=1)
    Xte = pd.concat([test[NUMERICS].reset_index(drop=True), te_d[derived_cols].reset_index(drop=True)], axis=1)
    return Xtr, Xte


def build_original_rates(train, test):
    """Per-value label rates from the original 7,500-row dataset (fixed external evidence)."""
    orig = read_original(keep_level=True)
    out_tr, out_te = {}, {}
    for c in NUMERICS:
        tk = _tok(orig[c])
        g = pd.DataFrame({"t": tk, "y": orig[TARGET].values}).groupby("t")["y"].agg(["mean", "count"])
        mp = g["mean"]
        out_tr[f"orig_rate_{c}"] = _apply_map(_tok(train[c]), mp, orig[TARGET].mean())
        out_te[f"orig_rate_{c}"] = _apply_map(_tok(test[c]), mp, orig[TARGET].mean())
        sev = (orig["addiction_level"] == "Severe").astype(np.float64)
        g2 = pd.DataFrame({"t": tk, "y": sev}).groupby("t")["y"].agg(["mean", "count"])
        mp2 = g2["mean"]
        out_tr[f"orig_sev_{c}"] = _apply_map(_tok(train[c]), mp2, sev.mean())
        out_te[f"orig_sev_{c}"] = _apply_map(_tok(test[c]), mp2, sev.mean())
    return pd.DataFrame(out_tr), pd.DataFrame(out_te)


def _te_specs():
    """Return list of (name, token_fn, smooth). token_fn(df) -> token Series."""
    specs = []

    # single columns
    for c in NUMERICS + CATEGORICALS:
        specs.append((f"te_{c}", (lambda df, c=c: _tok(df[c])), 30.0))

    # derived quantities are also generated tokens
    for c, sm in [("slack", 30.0), ("comp_sum", 30.0), ("sleep_daily", 30.0),
                  ("utilization", 50.0)]:
        specs.append((f"te_{c}", (lambda df, c=c: _tok(df[c])), sm))

    # multi-scale smoothing for the top fields (sharp captures the per-token response)
    for c in ["daily_screen_time_hours", "weekend_screen_time",
              "social_media_hours", "notifications_per_day", "app_opens_per_day"]:
        specs.append((f"te_{c}_s10", (lambda df, c=c: _tok(df[c])), 10.0))
        specs.append((f"te_{c}_s200", (lambda df, c=c: _tok(df[c])), 200.0))

    # pairs
    pairs = [
        ("notifications_per_day", "app_opens_per_day"),
        ("daily_screen_time_hours", "weekend_screen_time"),
        ("sleep_hours", "age"),
        ("social_media_hours", "gaming_hours"),
        ("daily_screen_time_hours", "social_media_hours"),
        ("daily_screen_time_hours", "work_study_hours"),
        ("daily_screen_time_hours", "gaming_hours"),
        ("sleep_hours", "stress_level"),
        ("age", "gender"),
        ("social_media_hours", "work_study_hours"),
    ]
    for a, b in pairs:
        specs.append((f"te_{a}__{b}", (lambda df, a=a, b=b: _tok(df[a]) + "__" + _tok(df[b])), 100.0))

    return specs


def build_te_features(train, test, y, folds, n_inner=5, inner_seed=777):
    """Nested target encodings.

    Returns:
      te_train: dict fold -> ndarray (n_train, n_te)
      te_test:  dict fold -> ndarray (n_test,  n_te)
      te_names: list of feature names
    """
    tr_d = compute_derived(train)
    te_d = compute_derived(test)

    specs = _te_specs()
    n_tr, n_te = len(train), len(test)
    te_names = [nm for nm, _, _ in specs]
    te_train = {f: np.full((n_tr, len(specs)), np.nan) for f in range(5)}
    te_test = {f: np.full((n_te, len(specs)), np.nan) for f in range(5)}

    tok_tr = {nm: fn(tr_d).values for nm, fn, _ in specs}
    tok_te = {nm: fn(te_d).values for nm, fn, _ in specs}

    for f in range(5):
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        y_trn = y[trn]
        prior = y_trn.mean()
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=inner_seed)
        inner_splits = list(inner.split(np.zeros(len(trn)), y_trn))

        for j, (nm, _, smooth) in enumerate(specs):
            t_tr = tok_tr[nm]
            t_te = tok_te[nm]
            for i_fit, i_app in inner_splits:
                mp = _te_map(t_tr[trn[i_fit]], y_trn[i_fit], prior, smooth)
                te_train[f][trn[i_app], j] = _apply_map(t_tr[trn[i_app]], mp, prior)
            mp = _te_map(t_tr[trn], y_trn, prior, smooth)
            te_train[f][val, j] = _apply_map(t_tr[val], mp, prior)
            te_test[f][:, j] = _apply_map(t_te, mp, prior)

    return te_train, te_test, te_names


def build_all(train, test, y, folds, n_inner=5, inner_seed=777):
    """Build everything. Returns (Xraw_tr, Xraw_te, te_train, te_test, te_names)."""
    Xraw_tr, Xraw_te = build_raw_derived(train, test)
    Xorig_tr, Xorig_te = build_original_rates(train, test)
    Xraw_tr = pd.concat([Xraw_tr, Xorig_tr], axis=1)
    Xraw_te = pd.concat([Xraw_te, Xorig_te], axis=1)
    te_train, te_test, te_names = build_te_features(train, test, y, folds, n_inner, inner_seed)
    return Xraw_tr, Xraw_te, te_train, te_test, te_names
