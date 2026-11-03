"""Label-free numeric side input of the AFN model (44 columns, cached).

LightGBM quantile imputers (alpha 0.1 / 0.5 / 0.9; 300 rounds, 63 leaves, lr 0.1, min_data_in_leaf 100, feature and
bagging fraction 0.8, seed 777) predict each numeric from the other 8 numerics (NaN kept) and the 3 ordinal
categoricals; they are fit on the rows of train + test where the column is observed.
Side input = 9 raw numerics, 9 q50 fills, 9 fill uncertainties (q90 - q10 where the cell is missing, else 0) and
17 budget-geometry columns recomputed from the fills; every column standardized on train + test (raw NaN -> 0)."""
import pickle
import time

import numpy as np
import pandas as pd

from common.data import NUMERICS as _NUMS, cache_dir

NUMERICS = list(_NUMS)
FOLDER = 'factorization_machines'


def _design_matrix(all_data):
    X = all_data[NUMERICS].copy()
    X["gender_ord"] = all_data["gender"].map({"Female": 0, "Male": 1, "Other": 2})
    X["stress_ord"] = all_data["stress_level"].map({"Low": 0, "Medium": 1, "High": 2})
    X["impact_ord"] = all_data["academic_work_impact"].map({"No": 0, "Yes": 1})
    return X.values.astype(np.float32)


def fit_imputers(train, test, n_trees=300):
    """Per-column LightGBM quantile imputers (label-free, train + test). Cached."""
    cache_path = cache_dir(FOLDER) / 'imputers.pkl'
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    import lightgbm as lgb

    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    X_all = _design_matrix(all_data)
    imputers = {}
    for c in NUMERICS:
        t0 = time.time()
        ci = NUMERICS.index(c)
        obs = ~np.isnan(X_all[:, ci])
        cols = [j for j in range(X_all.shape[1]) if j != ci]
        Xtr, ytr = X_all[obs][:, cols], X_all[obs, ci]
        fits = {}
        for alpha in (0.1, 0.5, 0.9):
            params = {
                "objective": "quantile", "alpha": alpha, "metric": "quantile",
                "num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 100,
                "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 1, "verbosity": -1, "num_threads": 16,
                "seed": 777,
            }
            fits[alpha] = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=n_trees)
        imputers[c] = (cols, fits)
        print(f"  imputer[{c}] fit in {time.time() - t0:.1f}s", flush=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(imputers, fh)
    return imputers


def impute(train, test, imputers):
    """Returns (imp, unc) for train + test stacked: (n, 9) each."""
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    X_all = _design_matrix(all_data)
    n = X_all.shape[0]
    imp = np.zeros((n, 9), dtype=np.float64)
    unc = np.zeros((n, 9), dtype=np.float64)
    for c in NUMERICS:
        ci = NUMERICS.index(c)
        cols, fits = imputers[c]
        Xp = X_all[:, cols]
        q10 = fits[0.1].predict(Xp)
        q50 = fits[0.5].predict(Xp)
        q90 = fits[0.9].predict(Xp)
        imp[:, ci] = q50
        miss = np.isnan(X_all[:, ci])
        unc[:, ci] = np.where(miss, q90 - q10, 0.0)
    return imp, unc


def geometry_from_imputed(imp):
    """Budget geometry recomputed from the fills (always defined), 17 columns."""
    d, s, g, w, sl, n, a, wk = (imp[:, i] for i in range(1, 9))
    comp_sum = s + g + w
    out = {}
    out["fi_slack"] = d - comp_sum
    out["fi_comp_sum"] = comp_sum
    out["fi_utilization"] = comp_sum / np.maximum(d, 1e-6)
    out["fi_sleep_daily"] = sl + d
    out["fi_social_ratio"] = s / np.maximum(d, 1e-6)
    out["fi_gaming_ratio"] = g / np.maximum(d, 1e-6)
    out["fi_work_ratio"] = w / np.maximum(d, 1e-6)
    out["fi_notif_per_hour"] = n / np.maximum(d, 1e-6)
    out["fi_app_per_hour"] = a / np.maximum(d, 1e-6)
    out["fi_weekend_daily_diff"] = wk - d
    out["fi_weekend_daily_ratio"] = wk / np.maximum(d, 1e-6)
    out["fi_notif_x_app"] = n * a
    out["fi_social_x_gaming"] = s * g
    out["fi_gaming_x_work"] = g * w
    out["fi_repair_flag"] = ((out["fi_utilization"] >= 0.85) & (out["fi_utilization"] <= 0.95)).astype(np.float64)
    out["fi_is_exact_budget"] = (np.abs(out["fi_slack"]) < 1e-9).astype(np.float64)
    out["fi_at_20"] = (np.abs(out["fi_sleep_daily"] - 20.0) < 1e-9).astype(np.float64)
    return out


def build_numeric_side_full(train, test):
    """The 44-column side input (raw 9 + imputed 9 + uncertainty 9 + geometry 17), standardized. Cached."""
    cache_path = cache_dir(FOLDER) / 'fm_side_full.npz'
    if cache_path.exists():
        z = np.load(cache_path)
        return z["side_train"], z["side_test"]

    imputers = fit_imputers(train, test)
    imp, unc = impute(train, test, imputers)
    geo = geometry_from_imputed(imp)
    geo_mat = np.column_stack([geo[k] for k in geo]).astype(np.float64)

    al = pd.concat([train, test], ignore_index=True)
    raw = al[NUMERICS].values.astype(np.float64)

    parts = []
    for M, has_nan in [(raw, True), (imp, False), (unc, False), (geo_mat, False)]:
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = (M - mu) / sd
        if has_nan:
            Z = np.where(np.isnan(Z), 0.0, Z)
        parts.append(Z)
    side = np.hstack(parts).astype(np.float32)
    side = np.nan_to_num(side, nan=0.0, posinf=0.0, neginf=0.0)
    side_tr, side_te = side[:len(train)], side[len(train):]
    np.savez_compressed(cache_path, side_train=side_tr, side_test=side_te)
    print(f"built side input: {side.shape[1]} cols (raw 9 + imputed 9 + uncertainty 9 + geometry 17)", flush=True)
    return side_tr, side_te
