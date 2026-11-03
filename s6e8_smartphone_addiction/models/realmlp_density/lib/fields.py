"""Smooth response fields (sf) and MLM row-embedding principal components (emb).

add_smooth_fields: Gaussian-blurred label-response surfaces on a value grid, nested inside the outer fold (inner
  StratifiedKFold(5, shuffle=True, random_state=777) on outer-train; the map of all outer-train rows for validation /
  test rows). Rate = (blurred positives + 25 prior) / (blurred count + 25), stored as a logit.
  1-D (sf1_<col>): the 9 numerics on the 0.1 grid (0.01 grid for the columns with <= 260 distinct values), sigma 2.
  2-D (sf2_<a>_<b>): (daily, social) and (daily, weekend) on the 0.1 grid, sigma 1.5; (notifications, app_opens) on
  the integer grid, sigma 1.0.
add_embeddings: 48 PCA components of the L2-normalized, centred MLM row embeddings (train + test, no labels), from
  row_emb384.npy of ../density_features.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.model_selection import StratifiedKFold

from .assemble import NUMS, density_file

SF2_PAIRS = [('daily_screen_time_hours', 'social_media_hours', 1.5),
             ('daily_screen_time_hours', 'weekend_screen_time', 1.5),
             ('notifications_per_day', 'app_opens_per_day', 1.0)]
K_SM = 25.0


def _logit(r):
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return np.log(r / (1 - r))


def _field1(vals, yy, all_grid, sigma, prior):
    g = pd.DataFrame({'v': vals, 'y': yy}).dropna().groupby('v')['y'].agg(['sum', 'size'])
    idx = pd.Index(all_grid)
    s = g['sum'].reindex(idx, fill_value=0.0).values
    c = g['size'].reindex(idx, fill_value=0.0).values
    rate = (gaussian_filter1d(s, sigma) + prior * K_SM) / (gaussian_filter1d(c, sigma) + K_SM)
    return dict(zip(all_grid, _logit(rate)))


def _field2(va, vb, yy, ga, gb, sigma, prior):
    ia = pd.Categorical(va, categories=ga).codes
    ib = pd.Categorical(vb, categories=gb).codes
    ok = (ia >= 0) & (ib >= 0) & ~np.isnan(yy)
    S = np.zeros((len(ga), len(gb)))
    C = np.zeros_like(S)
    np.add.at(S, (ia[ok], ib[ok]), yy[ok])
    np.add.at(C, (ia[ok], ib[ok]), 1.0)
    return _logit((gaussian_filter(S, sigma) + prior * K_SM) / (gaussian_filter(C, sigma) + K_SM))


def add_smooth_fields(A, B, y, trn_idx, val_idx, train, test, sigma1=2.0):
    n_tr = len(train)
    prior = y[trn_idx].mean()
    inner = StratifiedKFold(5, shuffle=True, random_state=777)

    for c in NUMS:
        r = 1 if train[c].nunique() > 260 else 2
        vt, ve = np.round(train[c].values, r), np.round(test[c].values, r)
        all_grid = np.unique(np.concatenate([vt[~np.isnan(vt)], ve[~np.isnan(ve)]]))
        col_tr = np.full(n_tr, np.nan, np.float32)
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
            mp = _field1(vt[trn_idx[i_fit]], y[trn_idx[i_fit]], all_grid, sigma1, prior)
            col_tr[trn_idx[i_app]] = pd.Series(vt[trn_idx[i_app]]).map(mp).values
        mp = _field1(vt[trn_idx], y[trn_idx], all_grid, sigma1, prior)
        col_tr[val_idx] = pd.Series(vt[val_idx]).map(mp).values
        A[f'sf1_{c}'] = col_tr
        B[f'sf1_{c}'] = pd.Series(ve).map(mp).values.astype(np.float32)

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
        name = f"sf2_{ca.split('_')[0]}_{cb.split('_')[0]}"
        A[name] = col_tr
        B[name] = apply2(F, va_e, vb_e)
    return A, B


def add_embeddings(A, B, n_comp=48):
    from sklearn.decomposition import PCA
    emb = np.load(density_file('row_emb384.npy')).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    pc = PCA(n_components=n_comp, random_state=0, svd_solver='randomized').fit_transform(emb - emb.mean(0))
    n_tr = len(A)
    for j in range(n_comp):
        A[f'e{j}'] = pc[:n_tr, j].astype(np.float32)
        B[f'e{j}'] = pc[n_tr:, j].astype(np.float32)
    return A, B
