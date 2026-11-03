"""Key coding, the pair-lift estimator and the fold loop shared by the lift banks.

Pair lift of the keys (A, B) estimated on a set of labeled fit rows with label rate p0:
  pA, pB = label rate of the row's A key / B key, smoothed toward p0 with strength 20 (an unseen key gives p0)
  base   = sigmoid(logit pA + logit pB - logit p0)               log-odds independence of A and B
  pair   = (positives in the (A, B) cell + s * base) / (rows in the (A, B) cell + s)
  lift   = pair - base                                           (0 for a cell the fit rows never saw)
Fold discipline: in outer fold f the outer-train rows get lifts from maps fit on the other 4 inner groups
(StratifiedKFold(5, shuffle, seed 777) of the outer-train rows); validation and test rows from maps fit on all outer-train
rows. Keys are coded on train + test together; a missing or undefined value is its own key."""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common.data import TARGET, cache_dir, folds, read_test, read_train

CACHE = cache_dir('lift_banks')
SM = 20.0


def load():
    """train, test, y, fold ids."""
    train, test = read_train(), read_test()
    y = train[TARGET].values
    return train, test, y, folds(y)


def col_all(train, test, c):
    return pd.concat([train[c], test[c]], ignore_index=True)


def codes_from_series(series, kind):
    """Integer key codes: exact ('%.2f'), r01 (0.1-rounded) or q025 (floor(v / 0.25)); missing -> '__NA__'."""
    if kind == 'exact':
        t = series.map(lambda v: f'{v:.2f}' if pd.notna(v) else '__NA__')
    elif kind == 'r01':
        t = series.map(lambda v: f'{round(v, 1):.1f}' if pd.notna(v) else '__NA__')
    elif kind == 'q025':
        v = series.values
        b = np.where(np.isfinite(v), np.floor(np.round(v, 2) / 0.25), np.nan)
        t = pd.Series(b).map(lambda x: f'{x:.0f}' if pd.notna(x) else '__NA__')
    c, u = pd.factorize(t, sort=True)
    return c.astype(np.int64), len(u)


def codes_q5(series):
    """floor(v / 0.5) key codes; missing -> '__NA__'."""
    v = series.values if hasattr(series, 'values') else series
    b = np.where(np.isfinite(v), np.floor(np.round(v, 2) / 0.5), np.nan)
    t = pd.Series(b).map(lambda x: f'{x:.0f}' if pd.notna(x) else '__NA__')
    c, u = pd.factorize(t, sort=True)
    return c.astype(np.int64), len(u)


def lgt(p, eps=1e-9):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def pair_lift(C, y, rows_fit, q_idx, a_key, b_key, sj):
    """Lift of key pair (a_key, b_key) for the global row ids q_idx (test rows = n_train + i), fit on train rows_fit.
    C: {key name: (codes over train + test, number of codes)}."""
    ca, na = C[a_key]; cb, nb = C[b_key]
    yy = y[rows_fit]; pri = yy.mean(); lpri = np.log(pri / (1 - pri))
    pA = (np.bincount(ca[rows_fit], weights=yy, minlength=na) + SM * pri) / (np.bincount(ca[rows_fit], minlength=na) + SM)
    pB = (np.bincount(cb[rows_fit], weights=yy, minlength=nb) + SM * pri) / (np.bincount(cb[rows_fit], minlength=nb) + SM)
    key = ca * nb + cb
    posP = np.bincount(key[rows_fit], weights=yy, minlength=na * nb)
    cntP = np.bincount(key[rows_fit], minlength=na * nb)
    kq = key[q_idx]
    base = 1.0 / (1.0 + np.exp(-(lgt(pA[ca[q_idx]]) + lgt(pB[cb[q_idx]]) - lpri)))
    return (posP[kq] + sj * base) / (cntP[kq] + sj) - base


def build_bank(stem, names, mean_name, lift, y, folds_, n_te, fold_ids, t0=None):
    """lift(channel index, fit rows, query rows) -> lifts. Writes <stem>_f<fold>_{tr,te}.parquet: one column per channel
    + their row mean."""
    t0 = time.time() if t0 is None else t0
    n_tr = len(y)
    te_idx = np.arange(n_tr, n_tr + n_te)
    for f in fold_ids:
        ti = np.where(folds_ != f)[0]; vi = np.where(folds_ == f)[0]
        OUT_tr = np.zeros((n_tr, len(names)), np.float32)
        OUT_te = np.zeros((n_te, len(names)), np.float32)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=777).split(np.zeros(len(ti)), y[ti]))
        for pj in range(len(names)):
            col = np.full(n_tr, np.nan)
            for i_fit, i_app in splits:
                col[ti[i_app]] = lift(pj, ti[i_fit], ti[i_app])
            col[vi] = lift(pj, ti, vi)
            OUT_tr[:, pj] = col
            OUT_te[:, pj] = lift(pj, ti, te_idx)
        Dtr = pd.DataFrame(OUT_tr, columns=names); Dtr[mean_name] = OUT_tr.mean(1)
        Dte = pd.DataFrame(OUT_te, columns=names); Dte[mean_name] = OUT_te.mean(1)
        Dtr.astype(np.float32).to_parquet(CACHE / f'{stem}_f{f}_tr.parquet')
        Dte.astype(np.float32).to_parquet(CACHE / f'{stem}_f{f}_te.parquet')
        print(f'{stem} fold {f} done ({time.time() - t0:.0f}s)', flush=True)
