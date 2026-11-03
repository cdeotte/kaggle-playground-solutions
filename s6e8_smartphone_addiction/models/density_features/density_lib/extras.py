"""Two blocks added on top of assemble(): smooth response fields (sf) and PCs of the MLM row embeddings (emb).

add_smooth_fields (sf): cross-fitted, Gaussian-blurred label-rate surfaces, on the logit scale.
  1-D: one per numeric column on its value grid (0.1-rounded for columns with > 260 distinct train values, exact 0.01
       grid otherwise), blur sigma1 = 2 grid steps.
  2-D: (daily, social) and (daily, weekend) with sigma 1.5, (notifications, app opens) with sigma 1.0; grid = 0.1-rounded
       values (> 260 distinct values) or integers.
  Label sums and counts are blurred separately; rate = (blurred sum + 25 * prior) / (blurred count + 25).
  Outer-train rows read fields fit on the other inner groups (StratifiedKFold(5, shuffle, seed 777)), validation and test
  rows the field fit on all outer-train rows. Columns sf1_<col>, sf2_<a>_<b> (tag inserts a suffix: sf1<tag>_...).
add_embeddings (emb): PCA (randomized SVD, seed 0) of the L2-normalized MLM row embeddings of train + test (label-free).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from .base import CACHE, NUMS

SF1_COLS = NUMS
SF2_PAIRS = [('daily_screen_time_hours', 'social_media_hours', 1.5),
             ('daily_screen_time_hours', 'weekend_screen_time', 1.5),
             ('notifications_per_day', 'app_opens_per_day', 1.0)]
K_SM = 25.0


def _grid1(v, r=1):
    return np.round(v, r)


def _field1(vals, yy, all_grid, sigma, prior):
    g = pd.DataFrame({'v': vals, 'y': yy}).dropna().groupby('v')['y'].agg(['sum', 'size'])
    idx = pd.Index(all_grid)
    s = g['sum'].reindex(idx, fill_value=0.0).values
    c = g['size'].reindex(idx, fill_value=0.0).values
    sb = gaussian_filter1d(s, sigma)
    cb = gaussian_filter1d(c, sigma)
    rate = (sb + prior * K_SM) / (cb + K_SM)
    lo = np.log(np.clip(rate, 1e-4, 1 - 1e-4) / (1 - np.clip(rate, 1e-4, 1 - 1e-4)))
    return dict(zip(all_grid, lo))


def _field2(va, vb, yy, ga, gb, sigma, prior):
    ia = pd.Categorical(va, categories=ga).codes
    ib = pd.Categorical(vb, categories=gb).codes
    ok = (ia >= 0) & (ib >= 0) & ~np.isnan(yy)
    S = np.zeros((len(ga), len(gb))); C = np.zeros_like(S)
    np.add.at(S, (ia[ok], ib[ok]), yy[ok])
    np.add.at(C, (ia[ok], ib[ok]), 1.0)
    Sb = gaussian_filter(S, sigma); Cb = gaussian_filter(C, sigma)
    R = (Sb + prior * K_SM) / (Cb + K_SM)
    return np.log(np.clip(R, 1e-4, 1 - 1e-4) / (1 - np.clip(R, 1e-4, 1 - 1e-4)))


def add_smooth_fields(A, B, y, trn_idx, val_idx, train, test, sigma1=2.0, sigma2=1.5, tag=''):
    """sigma2 is unused: each 2-D pair has its own sigma (SF2_PAIRS)."""
    n_tr = len(train)
    prior = y[trn_idx].mean()
    inner = StratifiedKFold(5, shuffle=True, random_state=777)

    # 1-D fields
    for c in SF1_COLS:
        r = 1 if train[c].nunique() > 260 else 2   # exact grid for notifications / app opens / age
        vt = _grid1(train[c].values, r); ve = _grid1(test[c].values, r)
        all_grid = np.unique(np.concatenate([vt[~np.isnan(vt)], ve[~np.isnan(ve)]]))
        col_tr = np.full(n_tr, np.nan, np.float32)
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y[trn_idx]):
            mp = _field1(vt[trn_idx[i_fit]], y[trn_idx[i_fit]], all_grid, sigma1, prior)
            col_tr[trn_idx[i_app]] = pd.Series(vt[trn_idx[i_app]]).map(mp).values
        mp = _field1(vt[trn_idx], y[trn_idx], all_grid, sigma1, prior)
        col_tr[val_idx] = pd.Series(vt[val_idx]).map(mp).values
        A[f'sf1{tag}_{c}'] = col_tr
        B[f'sf1{tag}_{c}'] = pd.Series(ve).map(mp).values.astype(np.float32)

    # 2-D fields
    for ca, cb, sg in SF2_PAIRS:
        ra = 1 if train[ca].nunique() > 260 else 0
        rb = 1 if train[cb].nunique() > 260 else 0
        va_t = _grid1(train[ca].values, ra); vb_t = _grid1(train[cb].values, rb)
        va_e = _grid1(test[ca].values, ra); vb_e = _grid1(test[cb].values, rb)
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
            F = _field2(va_t[trn_idx[i_fit]], vb_t[trn_idx[i_fit]],
                        y[trn_idx[i_fit]].astype(float), ga, gb, sg, prior)
            col_tr[trn_idx[i_app]] = apply2(F, va_t[trn_idx[i_app]], vb_t[trn_idx[i_app]])
        F = _field2(va_t[trn_idx], vb_t[trn_idx], y[trn_idx].astype(float), ga, gb, sg, prior)
        col_tr[val_idx] = apply2(F, va_t[val_idx], vb_t[val_idx])
        nm = f"sf2{tag}_{ca.split('_')[0]}_{cb.split('_')[0]}"
        A[nm] = col_tr
        B[nm] = apply2(F, va_e, vb_e)
    return A, B


def add_embeddings(A, B, n_comp=48, path=None):
    emb = np.load(CACHE / 'row_emb384.npy' if path is None else path).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    mu = emb.mean(0); X = emb - mu
    pc = PCA(n_components=n_comp, random_state=0, svd_solver='randomized').fit_transform(X)
    n_tr = len(A)
    for j in range(n_comp):
        A[f'e{j}'] = pc[:n_tr, j].astype(np.float32)
        B[f'e{j}'] = pc[n_tr:, j].astype(np.float32)
    return A, B
