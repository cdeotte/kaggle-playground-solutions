"""Features of the CatBoost model (catboost_quantile_bins): a smooth representation without exact-value tokens or target
encodings, 25 columns:
  * 9 quantile-binned numerics (16 bins; missing values stay NaN for CatBoost's native missing handling)
  * 3 quantile-binned geometry columns (slack, utilization, weekend / daily)
  * mask / slice block: n_miss_screen, n_miss_total, coin_region (6 < daily <= 8 and social <= 4), daily_missing,
    >= 3 missing screen columns
  * 5 conditional-median imputations (2-level fallback)
  * 3 native CatBoost categoricals (gender, stress_level, academic_work_impact; NA -> 'NA')
Bin edges and conditional medians are fitted per outer fold on the outer-train rows (no labels)."""
import numpy as np
import pandas as pd

NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
SCREEN = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
          "work_study_hours", "weekend_screen_time"]
N_BINS = 16
CAT_COLS = CAT


def _safe_div(a, b):
    return a / b.where(b > 0)


def derived(df):
    d, so, g, w = (df[c] for c in ["daily_screen_time_hours", "social_media_hours",
                                   "gaming_hours", "work_study_hours"])
    we = df["weekend_screen_time"]
    comp = so + g + w
    out = pd.DataFrame(index=df.index)
    out["slack"] = d - comp
    out["utilization"] = _safe_div(comp, d)
    out["wed_over_daily"] = _safe_div(we, d)
    return out


def _bin_edges(vals: np.ndarray) -> np.ndarray:
    """16 quantile edges from the finite values of the fit rows."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return np.array([], dtype="float64")
    q = np.linspace(0, 1, N_BINS + 1)[1:-1]
    return np.quantile(v, q)


def _binize(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = vals.astype("float64").copy()
    if len(edges) == 0:
        out[:] = np.nan
        return out
    finite = np.isfinite(vals)
    out[finite] = np.digitize(vals[finite], edges)   # 0..N_BINS
    out[~finite] = np.nan
    return out


def _cond_median(col: str, key_cols, fit_df: pd.DataFrame) -> dict:
    """Per-row conditional median (2-level fallback) fitted on fit_df; applied by _apply_cond_median."""
    t = fit_df[col].to_numpy(dtype="float64")
    if len(key_cols) > 1:
        keys = pd.MultiIndex.from_arrays([fit_df[c] for c in key_cols])
        med2 = pd.DataFrame({"t": t, "k": keys}).groupby("k")["t"].median()
        k0 = fit_df[key_cols[0]]
        med1 = pd.DataFrame({"t": t, "k": k0}).groupby("k")["t"].median()
    else:
        k0 = fit_df[key_cols[0]]
        med1 = pd.DataFrame({"t": t, "k": k0}).groupby("k")["t"].median()
        med2 = None
    g0 = np.nanmedian(t)
    return {"col": col, "med1": med1, "med2": med2, "g0": g0, "keys": key_cols, "k0col": key_cols[0]}


def _apply_cond_median(spec, df: pd.DataFrame) -> np.ndarray:
    col, k0col = spec["col"], spec["k0col"]
    out = df[col].to_numpy(dtype="float64").copy()
    m = np.isnan(out)
    if not m.any():
        return out
    k0 = df[k0col].to_numpy()[m]
    if spec["med2"] is not None:
        k1 = df[spec["keys"][1]].to_numpy()[m]
        v = spec["med2"].reindex(pd.MultiIndex.from_arrays([k0, k1])).to_numpy(dtype="float64").copy()
        still = np.isnan(v)
        if still.any():
            v[still] = spec["med1"].reindex(pd.Index(k0[still])).to_numpy(dtype="float64").copy()
        still = np.isnan(v)
        v[still] = spec["g0"]
        out[m] = v
    else:
        v = spec["med1"].reindex(pd.Index(k0)).to_numpy(dtype="float64").copy()
        still = np.isnan(v)
        v[still] = spec["g0"]
        out[m] = v
    return out


def fit_and_transform(train: pd.DataFrame, test: pd.DataFrame, fit_idx: np.ndarray):
    """Fit the features on train[fit_idx] (outer-train), return (X_train_df, X_test_df)."""
    fit_df = train.loc[fit_idx]

    # binned numerics (16 quantile bins)
    edges = {c: _bin_edges(fit_df[c].to_numpy(dtype="float64")) for c in NUM}

    # binned geometry
    der_fit = derived(fit_df)
    geo_edges = {c: _bin_edges(der_fit[c].to_numpy(dtype="float64")) for c in ["slack", "utilization", "wed_over_daily"]}

    # conditional-median specs
    cmi_specs = {
        "cmi_daily": _cond_median("daily_screen_time_hours", ["social_media_hours", "gaming_hours"], fit_df),
        "cmi_social": _cond_median("social_media_hours", ["daily_screen_time_hours", "gaming_hours"], fit_df),
        "cmi_weekend": _cond_median("weekend_screen_time", ["daily_screen_time_hours", "social_media_hours"], fit_df),
        "cmi_notif": _cond_median("notifications_per_day", ["app_opens_per_day"], fit_df),
        "cmi_app": _cond_median("app_opens_per_day", ["notifications_per_day"], fit_df),
    }

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in NUM:
            X[f"bin_{c}"] = _binize(df[c].to_numpy(dtype="float64"), edges[c])
        der = derived(df)
        for c in ["slack", "utilization", "wed_over_daily"]:
            X[f"bin_{c}"] = _binize(der[c].to_numpy(dtype="float64"), geo_edges[c])
        # mask / block structure
        miss = df[NUM].isna().to_numpy()
        X["n_miss_screen"] = df[SCREEN].isna().sum(axis=1).to_numpy(dtype="float64")
        X["n_miss_total"] = miss.sum(axis=1).astype("float64")
        # slice indicators
        d = df["daily_screen_time_hours"].to_numpy(dtype="float64")
        s = df["social_media_hours"].to_numpy(dtype="float64")
        coin = np.isfinite(d) & np.isfinite(s) & (d > 6) & (d <= 8) & (s <= 4)
        X["coin_region"] = coin.astype("float64")
        X["daily_missing"] = np.isnan(d).astype("float64")
        X["ge3_miss_screen"] = (df[SCREEN].isna().sum(axis=1).to_numpy() >= 3).astype("float64")
        # conditional-median imputations
        for k, spec in cmi_specs.items():
            X[k] = _apply_cond_median(spec, df)
        # native categoricals (string; NA -> "NA")
        for c in CAT:
            X[c] = df[c].fillna("NA").astype(str)
        return X

    return transform(train), transform(test)
