#!/usr/bin/env python
"""Label-free static bases of the 5 numeric columns (identical for every fold), computed over train + test:
  rbf_   Gaussian bumps at 16 quantile centres (income without the 30,000 atom, commute without the 5.0 atom),
         bandwidth = median centre spacing
  per_   sin / cos(2 pi f z) of the standardised value z, f = 0.5, 1, 2   (the only family the pairs use)
  per2_  the same at f = 0.25, 4
  rk_    rank-gauss of the value
  cnt_   log(1 + count of the exact value)
-> features/static_tr.npy, features/static_te.npy, features/static_cols.json
Usage: python static_bases.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import read_test, read_train  # noqa: E402

NUM = ['Annual_Income_USD', 'Daily_Commute_km', 'Age', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work']


def main():
    tr, te = read_train(), read_test()
    cols, A = [], []

    def add(n, v):
        cols.append(n)
        A.append(np.asarray(v, np.float32))
    for c in NUM:
        v = np.r_[tr[c].values, te[c].values].astype(float)
        base = v[v != 30000] if c == 'Annual_Income_USD' else (v[v != 5.0] if c == 'Daily_Commute_km' else v)
        centres = np.unique(np.quantile(base, np.linspace(0.03, 0.97, 16)))
        bw = np.median(np.diff(centres)) if len(centres) > 1 else 1.0
        for j, m in enumerate(centres):
            add(f'rbf_{c}_{j}', np.exp(-0.5 * ((v - m) / bw) ** 2))
        z = (v - v.mean()) / v.std()
        for f in (0.5, 1.0, 2.0):
            add(f'per_{c}_s{f}', np.sin(2 * np.pi * f * z))
            add(f'per_{c}_c{f}', np.cos(2 * np.pi * f * z))
        for f in (0.25, 4.0):
            add(f'per2_{c}_s{f}', np.sin(2 * np.pi * f * z))
            add(f'per2_{c}_c{f}', np.cos(2 * np.pi * f * z))
        add(f'rk_{c}', ndtri(pd.Series(v).rank(method='average').values / (len(v) + 1.0)))
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        add(f'cnt_{c}', np.log1p(cnt[inv]))
    X = np.column_stack(A)
    out = HERE / 'features'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'static_tr.npy', X[:len(tr)])
    np.save(out / 'static_te.npy', X[len(tr):])
    json.dump(cols, open(out / 'static_cols.json', 'w'))
    print(f'{X.shape[1]} static columns -> {out}')


if __name__ == '__main__':
    main()
