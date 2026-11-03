"""Smooth response fields (`sf`): Gaussian-blurred response surfaces as continuous features, cross-fitted per outer fold.

1-D (sf1_<col>, 9 numerics): per-value positive sums and counts on the 0.1 grid (the exact grid for the columns with
<= 260 distinct values: age, notifications, app opens), blurred with gaussian_filter1d(sigma 2), rate shrunk to the
outer-train prior with weight 25, as a logit.
2-D (sf2_daily_social, sf2_daily_weekend at 0.1 x 0.1, sf2_notifications_app at 1 x 1): the same on a 2-D grid,
gaussian_filter(sigma 1.5 / 1.5 / 1.0).
Outer-train rows: 5 inner folds (seed 777), each filled from the field of the other 4; outer-validation and test rows: the
field of all outer-train rows."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.model_selection import StratifiedKFold

from common.data import NUMERICS

SF1_COLS = list(NUMERICS)
SF2_PAIRS = [("daily_screen_time_hours", "social_media_hours", 1.5),
             ("daily_screen_time_hours", "weekend_screen_time", 1.5),
             ("notifications_per_day", "app_opens_per_day", 1.0)]
K_SM = 25.0


def _logit(r):
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def _field1(vals, yy, all_grid, sigma, prior):
    g = pd.DataFrame({"v": vals, "y": yy}).dropna().groupby("v")["y"].agg(["sum", "size"])
    idx = pd.Index(all_grid)
    s = g["sum"].reindex(idx, fill_value=0.0).values
    c = g["size"].reindex(idx, fill_value=0.0).values
    sb = gaussian_filter1d(s, sigma)
    cb = gaussian_filter1d(c, sigma)
    rate = (sb + prior * K_SM) / (cb + K_SM)
    return dict(zip(all_grid, _logit(rate)))


def _field2(va, vb, yy, ga, gb, sigma, prior):
    ia = pd.Categorical(va, categories=ga).codes
    ib = pd.Categorical(vb, categories=gb).codes
    ok = (ia >= 0) & (ib >= 0) & ~np.isnan(yy)
    S = np.zeros((len(ga), len(gb)))
    C = np.zeros_like(S)
    np.add.at(S, (ia[ok], ib[ok]), yy[ok])
    np.add.at(C, (ia[ok], ib[ok]), 1.0)
    Sb = gaussian_filter(S, sigma)
    Cb = gaussian_filter(C, sigma)
    return _logit((Sb + prior * K_SM) / (Cb + K_SM))


def add_smooth_fields(A, B, y, trn_idx, val_idx, train, test, sigma1=2.0):
    n_tr = len(train)
    prior = y[trn_idx].mean()
    inner = StratifiedKFold(5, shuffle=True, random_state=777)

    for c in SF1_COLS:
        r = 1 if train[c].nunique() > 260 else 2
        vt = np.round(train[c].values, r)
        ve = np.round(test[c].values, r)
        all_grid = np.unique(np.concatenate([vt[~np.isnan(vt)], ve[~np.isnan(ve)]]))
        col_tr = np.full(n_tr, np.nan, np.float32)
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
            mp = _field1(vt[trn_idx[i_fit]], y[trn_idx[i_fit]], all_grid, sigma1, prior)
            col_tr[trn_idx[i_app]] = pd.Series(vt[trn_idx[i_app]]).map(mp).values
        mp = _field1(vt[trn_idx], y[trn_idx], all_grid, sigma1, prior)
        col_tr[val_idx] = pd.Series(vt[val_idx]).map(mp).values
        A[f"sf1_{c}"] = col_tr
        B[f"sf1_{c}"] = pd.Series(ve).map(mp).values.astype(np.float32)

    for ca, cb, sg in SF2_PAIRS:
        ra = 1 if train[ca].nunique() > 260 else 0
        rb = 1 if train[cb].nunique() > 260 else 0
        va_t, vb_t = np.round(train[ca].values, ra), np.round(train[cb].values, rb)
        va_e, vb_e = np.round(test[ca].values, ra), np.round(test[cb].values, rb)
        ga = np.unique(np.concatenate([va_t[~np.isnan(va_t)], va_e[~np.isnan(va_e)]]))
        gb = np.unique(np.concatenate([vb_t[~np.isnan(vb_t)], vb_e[~np.isnan(vb_e)]]))

        def apply2(F, va, vb):
            ia = pd.Categorical(va, categories=ga).codes
            ib = pd.Categorical(vb, categories=gb).codes
            out = np.full(len(va), np.nan, np.float32)
            ok = (ia >= 0) & (ib >= 0)
            out[ok] = F[ia[ok], ib[ok]]
            return out

        col_tr = np.full(n_tr, np.nan, np.float32)
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
            F = _field2(va_t[trn_idx[i_fit]], vb_t[trn_idx[i_fit]], y[trn_idx[i_fit]].astype(float), ga, gb, sg, prior)
            col_tr[trn_idx[i_app]] = apply2(F, va_t[trn_idx[i_app]], vb_t[trn_idx[i_app]])
        F = _field2(va_t[trn_idx], vb_t[trn_idx], y[trn_idx].astype(float), ga, gb, sg, prior)
        col_tr[val_idx] = apply2(F, va_t[val_idx], vb_t[val_idx])
        nm = f"sf2_{ca.split('_')[0]}_{cb.split('_')[0]}"
        A[nm] = col_tr
        B[nm] = apply2(F, va_e, vb_e)
    return A, B
