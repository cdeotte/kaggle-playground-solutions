"""TabPFN input features: the 12 raw columns + nested exact-value target encodings of the 12 columns + 4 derived columns.

Target encodings (smoothing 20, tokens '%.2f' / '__NA__'): per outer fold, outer-train rows (the TabPFN context) are
encoded by 5 inner folds (StratifiedKFold(5, shuffle=True, random_state=777)); validation and test rows by the map of all
outer-train rows."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

NUMERICS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CATS = ["gender", "stress_level", "academic_work_impact"]


def _tok(v):
    if v.dtype == object or v.dtype == "string":
        return v.fillna("__NA__").astype(str)
    return v.map(lambda x: f"{x:.2f}" if pd.notna(x) else "__NA__")


def config_a_raw(train):
    """12 raw columns; NaN kept; categoricals ordinal-encoded."""
    X = train[NUMERICS].values.astype(np.float32)
    X = np.hstack([
        X,
        train["gender"].map({"Female": 0, "Male": 1, "Other": 2}).fillna(-1).values.reshape(-1, 1),
        train["stress_level"].map({"Low": 0, "Medium": 1, "High": 2}).fillna(-1).values.reshape(-1, 1),
        train["academic_work_impact"].map({"No": 0, "Yes": 1}).fillna(-1).values.reshape(-1, 1),
    ]).astype(np.float32)
    return X


def _derived(train):
    d = train["daily_screen_time_hours"]
    s = train["social_media_hours"]
    g = train["gaming_hours"]
    w = train["work_study_hours"]
    sl = train["sleep_hours"]
    comp = s + g + w
    slack = (d - comp).values
    sleep_daily = (sl + d).values
    util = (comp / d).values
    missing_count = train[NUMERICS + CATS].isna().sum(axis=1).values.astype(np.float32)
    return np.column_stack([slack, sleep_daily, util, missing_count]).astype(np.float32)


def config_b_features(train, test, y, folds, smooth=20.0, n_inner=5, seed=777):
    """Raw 12 + TEs (smoothing 20) of the 12 columns + slack / sleep + daily / utilization / missing count (28 columns).

    Returns (X_fold, X_test): dicts fold -> array. X_fold[f] has the train rows (outer-train rows carry inner-fold TEs,
    the rows of fold f the full-map TEs); X_test[f] the test rows encoded with fold f's full map.
    """
    raw_tr = config_a_raw(train)
    raw_te = config_a_raw(test)
    der_tr = _derived(train)
    der_te = _derived(test)

    cols = NUMERICS + CATS
    n_tr, n_te = len(train), len(test)
    te_train = {f: np.zeros((n_tr, len(cols)), dtype=np.float32) for f in range(5)}
    te_test = {f: np.zeros((n_te, len(cols)), dtype=np.float32) for f in range(5)}

    for j, c in enumerate(cols):
        tok_tr = _tok(train[c]).values
        tok_te = _tok(test[c]).values
        for f in range(5):
            trn = np.where(folds != f)[0]
            val = np.where(folds == f)[0]
            y_trn = y[trn]
            prior = y_trn.mean()
            g = pd.DataFrame({"t": tok_tr[trn], "y": y_trn}).groupby("t")["y"].agg(["sum", "count"])
            mp = (g["sum"] + smooth * prior) / (g["count"] + smooth)
            te_train[f][val, j] = pd.Series(tok_tr[val]).map(mp).fillna(prior).values
            te_test[f][:, j] = pd.Series(tok_te).map(mp).fillna(prior).values
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
            for i_fit, i_app in inner.split(np.zeros(len(trn)), y_trn):
                g2 = pd.DataFrame({"t": tok_tr[trn[i_fit]], "y": y_trn[i_fit]}).groupby("t")["y"].agg(["sum", "count"])
                mp2 = (g2["sum"] + smooth * prior) / (g2["count"] + smooth)
                te_train[f][trn[i_app], j] = pd.Series(tok_tr[trn[i_app]]).map(mp2).fillna(prior).values

    X_fold = {f: np.hstack([raw_tr, te_train[f], der_tr]).astype(np.float32) for f in range(5)}
    X_test = {f: np.hstack([raw_te, te_test[f], der_te]).astype(np.float32) for f in range(5)}
    return X_fold, X_test
