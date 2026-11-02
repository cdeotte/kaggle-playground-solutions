#!/usr/bin/env python
"""Builds the fold-specific nested target-encoding library (203 channels, the "Q8" set) for every outer fold k:
features/X_f{k}_tr.npy (all train rows), features/X_f{k}_te.npy (test rows), features/cols.json.

For outer fold k (prior = mean label of the outer-train rows):
  outer-train rows: nested encodings: StratifiedKFold(5, shuffle=True, random_state=seed) over the outer-train rows, each
                    inner part encoded by the map fitted on the other four parts;
  validation rows (fold k) and test rows: the map fitted on all outer-train rows.
Single channels use seed 777; averaged channels (teav, teqav, teqbv: 3 seeds; w4*, w6*: 5 seeds; w7*: 8 seeds, counting
up from 777) average the logits over the inner splits.  ares = te1 - te20.  The column order is fixed (column sampling
in XGBoost depends on it).  Needs about 6 GB of memory and writes about 4 GB.
Usage: python build_library.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import te_library as L  # noqa: E402

HC = ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']
LOWCARD = L.SCAT + ['Number_of_Cars_Owned', 'Environmental_Concern_Level']
COARSE = ['IncQ500', 'IncQ1000', 'IncQ2000', 'IncQ5000', 'IncQ10000', 'ComQ0.5', 'ComQ1.0', 'ComQ2.0', 'ComQ5.0']
T4 = ['IncQ100', 'IncQ250', 'IncQ750', 'IncQ1500', 'IncQ3000', 'IncQ20000', 'ComQ0.2', 'ComQ0.25', 'ComQ3.0', 'ComQ10.0']
T6 = ([f'Inc{w}' for w in (50, 150, 300, 400, 600, 1200, 1750, 2500, 4000, 7500, 15000, 30000)]
      + [f'Com{w}' for w in (0.1, 0.3, 0.4, 0.7, 1.5, 4.0, 7.0)] + [f'Age{w}' for w in (2, 3, 5, 10)])
T7 = ([f'Inc{w}' for w in (75, 100, 200, 250, 500, 800, 1000, 1500, 2000, 3000, 5000, 6000, 10000, 20000)]
      + [f'Com{w}' for w in (0.2, 0.25, 0.5, 0.6, 0.8, 1.0, 2.0, 2.5, 3.0, 5.0, 10.0)] + [f'Age{w}' for w in (1, 4, 7, 15)])
S1, S3, S5, S8 = (777,), (777, 778, 779), tuple(range(777, 782)), tuple(range(777, 785))


def channels():
    """(name, kind, token, s, seeds) in library column order"""
    ch = [(f'base_{c}', 'static') for c in L.P] + [('atom_inc30k', 'static'), ('atom_com5', 'static')]
    for c in L.P:
        ch += [(f'te1_{c}', 'te', c, 1.0, S1), (f'te20_{c}', 'te', c, 20.0, S1)]
    for t in COARSE:
        ch += [(f'teq1_{t}', 'te', t, 1.0, S1), (f'teq20_{t}', 'te', t, 20.0, S1)]
    ch += [(f'ares_{c}', 'ares', c) for c in L.P]
    ch += [(f'teav_{c}', 'te', c, 1.0, S3) for c in HC]
    ch += [(f'teqav_{t}_s1.0', 'te', t, 1.0, S3) for t in COARSE]
    ch += [(f'teqbv_{t}_s20.0', 'te', t, 20.0, S3) for t in COARSE]
    ch += [(f'w4a_{t}_s1.0', 'te', t, 1.0, S5) for t in T4]
    ch += [(f'w4b_{t}_s20.0', 'te', t, 20.0, S5) for t in T4]
    ch += [(f'w6a_{t}', 'te', t, 1.0, S5) for t in T6]
    ch += [(f'w6b_{t}', 'te', t, 20.0, S5) for t in T6]
    ch += [(f'w7a_{t}', 'te', t, 1.0, S8) for t in T7]
    ch += [(f'w7b_{t}', 'te', t, 20.0, S8) for t in T7]
    for nb in (60, 120, 240, 480):
        ch += [(f'w7c_logInc{nb}_s1', 'te', f'logInc{nb}', 1.0, S8), (f'w7c_logInc{nb}_s20', 'te', f'logInc{nb}', 20.0, S8)]
    drop = {f'{f}_{c}' for f in ('te1', 'te20', 'ares') for c in LOWCARD}
    return [c for c in ch if c[0] not in drop]


_SPLITS = {}


def inner_splits(seed):
    """per outer fold: the 5 (fit rows, encoded rows) pairs of the inner split of the outer-train rows"""
    if seed not in _SPLITS:
        sp = []
        for k in range(5):
            itr = np.where(L.FOLDS != k)[0]
            sp.append([(itr[a], itr[b]) for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(itr, L.Y[itr])])
        _SPLITS[seed] = sp
    return _SPLITS[seed]


def nested_te(token, s, seed):
    """per outer fold: probabilities of the train rows (nested for outer-train, full map for fold k) and of the test rows"""
    codes, K = L.tok(token)
    kt, ke = codes[:L.NTR], codes[L.NTR:]
    out = []
    for k, splits in enumerate(inner_splits(seed)):
        itr, iva = np.where(L.FOLDS != k)[0], np.where(L.FOLDS == k)[0]
        pri = L.Y[itr].mean()
        A = np.full(L.NTR, pri)
        for f, ap in splits:
            A[ap] = L.rate(kt[f], L.Y[f], K, s, pri)[kt[ap]]
        m = L.rate(kt[itr], L.Y[itr], K, s, pri)
        A[iva] = m[kt[iva]]
        out.append((A, m[ke]))
    return out


def main():
    t0 = time.time()
    ch = channels()
    names = [c[0] for c in ch]
    assert len(names) == 203, len(names)
    col = {n: j for j, n in enumerate(names)}
    Xtr = [np.zeros((L.NTR, len(ch)), np.float32) for _ in range(5)]
    Xte = [np.zeros((L.NTE, len(ch)), np.float32) for _ in range(5)]
    for j, (n, kind, *arg) in enumerate(ch):
        if kind == 'static':
            for k in range(5):
                Xtr[k][:, j], Xte[k][:, j] = L.STATIC[n][:L.NTR], L.STATIC[n][L.NTR:]
        elif kind == 'te':
            token, s, seeds = arg
            A, B = [np.zeros(L.NTR) for _ in range(5)], [np.zeros(L.NTE) for _ in range(5)]
            for seed in seeds:
                for k, (a, b) in enumerate(nested_te(token, s, seed)):
                    A[k] += L.lg(a) / len(seeds)
                    B[k] += L.lg(b) / len(seeds)
            for k in range(5):
                Xtr[k][:, j], Xte[k][:, j] = A[k], B[k]
        else:
            i1, i2 = col[f'te1_{arg[0]}'], col[f'te20_{arg[0]}']
            for k in range(5):
                Xtr[k][:, j], Xte[k][:, j] = Xtr[k][:, i1] - Xtr[k][:, i2], Xte[k][:, i1] - Xte[k][:, i2]
        if j % 20 == 0:
            print(f'  {j + 1}/{len(ch)} {n} [{time.time() - t0:.0f}s]', flush=True)
    L.FEATURES.mkdir(parents=True, exist_ok=True)
    json.dump(names, open(L.FEATURES / 'cols.json', 'w'))
    for k in range(5):
        np.save(L.FEATURES / f'X_f{k}_tr.npy', np.nan_to_num(Xtr[k], nan=0., posinf=0., neginf=0.))
        np.save(L.FEATURES / f'X_f{k}_te.npy', np.nan_to_num(Xte[k], nan=0., posinf=0., neginf=0.))
    print(f'5 fold-specific matrices of {len(names)} channels saved to {L.FEATURES} [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
