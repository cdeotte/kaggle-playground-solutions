"""Label-free inputs of the FT-Transformer (one matrix for train + test, no fold machinery).

Imputation: for each of the 9 numerics, LightGBM quantile regressors (alpha 0.1 / 0.5 / 0.9; 200 rounds, 63 leaves,
lr 0.1, min_data_in_leaf 100, feature / bagging fraction 0.8, seed 777) predict the column from the other 8 numerics
(NaN kept) and the 3 ordinal categoricals; fit on the observed train + test rows.
Numeric inputs (47): the 9 q50 fills, 9 uncertainties (q90 - q10 where the cell is missing, else 0), the 9 raw values
(NaN -> q50 fill) and 20 budget-geometry columns recomputed from the fills.
Categorical inputs (12): gender / stress / impact codes (missing = own category) and the exact value token ('%.2f',
NaN -> '__NA__') of each numeric."""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

NUMERICS = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]
CATEGORICALS = ["gender", "stress_level", "academic_work_impact"]


def _tok(v):
    if v.dtype == object or v.dtype == "string":
        return v.fillna("__NA__").astype(str)
    return v.map(lambda x: f"{x:.2f}" if pd.notna(x) else "__NA__")


def _design(all_data):
    X_all = all_data[NUMERICS].copy()
    X_all["gender_ord"] = all_data["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    X_all["stress_ord"] = all_data["stress_level"].map({"Low": 0, "Medium": 1, "High": 2})
    X_all["impact_ord"] = all_data["academic_work_impact"].map({"No": 0, "Yes": 1})
    return X_all.values.astype(np.float32)


def _fit_imputers(train, test):
    """Per-column quantile imputers (label-free, train + test)."""
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    X_all = _design(all_data)

    imputers = {}
    for c in NUMERICS:
        ci = NUMERICS.index(c)
        obs = ~np.isnan(X_all[:, ci])
        Xo = X_all[obs]
        cols = [j for j in range(X_all.shape[1]) if j != ci]  # predictors: all other columns
        Xtr = Xo[:, cols]
        ytr = Xo[:, ci]
        fits = {}
        for alpha in (0.1, 0.5, 0.9):
            params = {
                "objective": "quantile", "alpha": alpha, "metric": "quantile",
                "num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 100,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                "verbosity": -1, "num_threads": 16, "seed": 777,
            }
            fits[alpha] = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=200)
        imputers[c] = (cols, fits)
    return imputers


def _impute_and_uncertainty(all_data, imputers):
    """Returns (imputed values (n, 9), uncertainty (n, 9))."""
    X_all = _design(all_data)
    n = X_all.shape[0]
    imp = np.zeros((n, 9), dtype=np.float32)
    unc = np.zeros((n, 9), dtype=np.float32)
    for c in NUMERICS:
        ci = NUMERICS.index(c)
        cols, fits = imputers[c]
        Xp = X_all[:, cols]
        q10 = fits[0.1].predict(Xp)
        q50 = fits[0.5].predict(Xp)
        q90 = fits[0.9].predict(Xp)
        imp[:, ci] = q50
        # uncertainty: 0 where observed, interval width where missing
        miss = np.isnan(X_all[:, ci])
        unc[:, ci] = np.where(miss, q90 - q10, 0.0)
    return imp, unc


def _geometry_from_imputed(imp):
    """Budget geometry recomputed from the fills (always defined)."""
    d = imp[:, 1]
    s = imp[:, 2]
    g = imp[:, 3]
    w = imp[:, 4]
    sl = imp[:, 5]
    n = imp[:, 6]
    a = imp[:, 7]
    wk = imp[:, 8]
    comp_sum = s + g + w
    out = {}
    out["slack"] = d - comp_sum
    out["comp_sum"] = comp_sum
    out["utilization"] = comp_sum / np.maximum(d, 1e-6)
    out["sleep_daily"] = sl + d
    out["social_ratio"] = s / np.maximum(d, 1e-6)
    out["gaming_ratio"] = g / np.maximum(d, 1e-6)
    out["work_ratio"] = w / np.maximum(d, 1e-6)
    out["notif_per_hour"] = n / np.maximum(d, 1e-6)
    out["app_per_hour"] = a / np.maximum(d, 1e-6)
    out["weekend_daily_diff"] = wk - d
    out["weekend_daily_ratio"] = wk / np.maximum(d, 1e-6)
    out["notif_x_app"] = n * a
    out["social_x_gaming"] = s * g
    out["gaming_x_work"] = g * w
    out["slack_cent"] = (out["slack"] * 100).round() % 100
    out["comp_sum_cent"] = (comp_sum * 100).round() % 100
    out["util_cent"] = (out["utilization"] * 100).round() % 100
    out["repair_flag"] = ((out["utilization"] >= 0.85) & (out["utilization"] <= 0.95)).astype(np.float32)
    out["is_exact_budget"] = (np.abs(out["slack"]) < 1e-9).astype(np.float32)
    out["at_20"] = (np.abs(out["sleep_daily"] - 20.0) < 1e-9).astype(np.float32)
    return out


def build_features(train, test, cache_path=None):
    """FT-Transformer input matrices over train + test rows (cached to cache_path if given).
    Returns X_num (n, 47), X_cat (n, 12) integer codes, the 12 cardinalities, num_names, cat_names."""
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=True)
        return (z["X_num"], z["X_cat"], list(z["cat_card"]),
                list(z["num_names"]), list(z["cat_names"]))

    imputers = _fit_imputers(train, test)
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    imp, unc = _impute_and_uncertainty(all_data, imputers)

    num_parts = [imp, unc]
    num_names = [f"imp_{c}" for c in NUMERICS] + [f"unc_{c}" for c in NUMERICS]

    # raw values (NaN -> imputed, so always defined)
    raw = all_data[NUMERICS].values.astype(np.float32)
    num_parts.append(np.where(np.isnan(raw), imp, raw))
    num_names += [f"raw_{c}" for c in NUMERICS]

    geo = _geometry_from_imputed(imp)
    num_parts.append(np.column_stack([geo[k] for k in geo]).astype(np.float32))
    num_names += list(geo.keys())
    X_num = np.hstack(num_parts)

    # categorical inputs: ordinal categoricals + exact value tokens of the raw numerics
    cat_parts, cat_names, cat_card = [], [], []
    for c in CATEGORICALS:
        codes = all_data[c].map({"Female": 0, "Male": 1, "Other": 2} if c == "gender"
                                else ({"Low": 0, "Medium": 1, "High": 2} if c == "stress_level"
                                      else {"No": 0, "Yes": 1})).fillna(-1).astype(np.int64).values
        codes = codes + 1  # missing -> its own category 0
        cat_parts.append(codes.reshape(-1, 1))
        cat_names.append(c)
        cat_card.append(codes.max() + 1)
    for c in NUMERICS:
        tk = _tok(all_data[c]).astype("category")
        cat_parts.append(tk.cat.codes.values.astype(np.int64).reshape(-1, 1))
        cat_names.append(f"tok_{c}")
        cat_card.append(tk.cat.categories.size)
    X_cat = np.hstack(cat_parts)

    if cache_path is not None:
        np.savez_compressed(cache_path, X_num=X_num, X_cat=X_cat, cat_card=np.array(cat_card),
                            num_names=np.array(num_names), cat_names=np.array(cat_names))
    return X_num, X_cat, cat_card, num_names, cat_names
