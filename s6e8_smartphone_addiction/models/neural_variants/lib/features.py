"""Features shared by the target-encoding models (NODE, FT-Transformer, wide linear model, gcForest, RuleFit).

build_raw_derived     the 9 raw numerics (NaN kept) + 23 label-free derived columns. Derived columns propagate NaN (no
                      imputation), so their exact-value tokens stay clean: budget slack = daily - (social + gaming +
                      work), component sum, utilization = sum / daily, the cent parts of these three, exact-budget flag,
                      sleep + daily, at-20 flag, social / gaming / work / notifications / app-opens per daily hour,
                      weekend - daily, weekend / daily, notifications x app opens, social x gaming, gaming x work,
                      repair flag (utilization in [0.85, 0.95]) and ordinal codes of the 3 categoricals.
build_original_rates  per-value label rate and 'Severe' addiction-level rate of each numeric in the 7,500-row original
                      dataset (18 columns; unseen values get the original's overall rate).
build_te_features     36 target encodings of exact value tokens (numerics formatted %.2f, NaN -> '__NA__'), smoothed
                      (sum + m * prior) / (count + m): the 12 columns (m 30), slack / comp_sum / sleep_daily (m 30) and
                      utilization (m 50), daily / weekend / social / notifications / app opens again at m 10 and m 200,
                      and 10 column pairs (m 100). Per outer fold the outer-train rows get inner 5-fold cross-fitted
                      values (StratifiedKFold, seed 777); the held-out and test rows get the map of all outer-train rows.
build_all             raw + derived + original rates (50 columns) and the 36 fold target encodings."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import CATEGORICALS as _CATS, NUMERICS as _NUMS, TARGET, folds as fold_ids, labels, read_original, \
    read_test, read_train

NUMERICS = list(_NUMS)
CATEGORICALS = list(_CATS)


def load_data():
    """train.csv, test.csv, the target as float64 and the fold id (0-4) of every train row."""
    train, test = read_train(), read_test()
    y = labels(train[TARGET])
    return train, test, y.astype(np.float64), fold_ids(y)


def _tok(v):
    """Exact-value token: numerics as '%.2f', strings as is, NaN -> '__NA__'."""
    if v.dtype == object or v.dtype == "string":
        return v.fillna("__NA__").astype(str)
    return v.map(lambda x: f"{x:.2f}" if pd.notna(x) else "__NA__")


def _te_map(tokens, y_fit, prior, smooth):
    g = pd.DataFrame({"t": tokens, "y": y_fit}).groupby("t")["y"].agg(["sum", "count"])
    return (g["sum"] + smooth * prior) / (g["count"] + smooth)


def _apply_map(tokens, mp, prior):
    s = pd.Series(tokens).map(mp)
    return s.fillna(prior).values.astype(np.float64)


def compute_derived(df):
    """Add the derived (label-free) columns to a copy of df."""
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

    # rows whose components were rescaled to ~85-95% of the daily total
    df["repair_flag"] = ((df["utilization"] >= 0.85) & (df["utilization"] <= 0.95)).astype(np.int8)
    return df


def build_raw_derived(train, test):
    """Label-free raw + derived features (32 columns). Returns (Xtr, Xte) DataFrames."""
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
    # raw numerics from the unimputed frames
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
    """List of (name, token_fn, smoothing); token_fn(df) -> token Series."""
    specs = []

    # single columns
    for c in NUMERICS + CATEGORICALS:
        specs.append((f"te_{c}", (lambda df, c=c: _tok(df[c])), 30.0))

    # derived quantities are value tokens too
    for c, sm in [("slack", 30.0), ("comp_sum", 30.0), ("sleep_daily", 30.0), ("utilization", 50.0)]:
        specs.append((f"te_{c}", (lambda df, c=c: _tok(df[c])), sm))

    # sharp (m 10) and smooth (m 200) encodings of the most informative columns
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
    """Nested target encodings. Returns te_train (dict fold -> (n_train, 36)), te_test (dict fold -> (n_test, 36)) and
    the 36 names."""
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
    """Returns (Xraw_tr, Xraw_te, te_train, te_test, te_names): raw + derived + original rates (50 columns) and the
    nested target encodings."""
    Xraw_tr, Xraw_te = build_raw_derived(train, test)
    Xorig_tr, Xorig_te = build_original_rates(train, test)
    Xraw_tr = pd.concat([Xraw_tr, Xorig_tr], axis=1)
    Xraw_te = pd.concat([Xraw_te, Xorig_te], axis=1)
    te_train, te_test, te_names = build_te_features(train, test, y, folds, n_inner, inner_seed)
    return Xraw_tr, Xraw_te, te_train, te_test, te_names
