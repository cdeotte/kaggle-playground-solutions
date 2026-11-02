#!/usr/bin/env python
"""Stage-8 digit features of income and commute (label-free, train + test rows): income mod 10 / 100 / 1000 / 10000,
the 5 lowest income digits, distance to the nearest 1000, commute tenths digit and integer part.
Writes cache/stage8/digits_all.npy + .json (names).
Usage: python extra_features.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from layout import STAGE1, STAGE8  # noqa: E402


def main():
    df = pd.read_parquet(STAGE1/'predictors.parquet', columns=['Annual_Income_USD', 'Daily_Commute_km'])
    income = df.Annual_Income_USD.to_numpy('float64')
    commute = df.Daily_Commute_km.to_numpy('float64')
    assert np.isfinite(income).all() and np.isfinite(commute).all()
    cols = {}
    inc = np.rint(income).astype('int64')
    for power in range(1, 5):
        cols[f'inc_mod_{10**power}'] = inc % 10**power
    for power in range(5):
        cols[f'inc_digit_{power}'] = (inc//10**power) % 10
    cols['inc_nearest1000_delta'] = inc-np.rint(inc/1000)*1000
    cols['commute_fraction'] = np.rint(commute*10).astype('int64') % 10
    cols['commute_integer'] = np.floor(commute)
    a = np.column_stack(list(cols.values())).astype('float32')
    STAGE8.mkdir(parents=True, exist_ok=True)
    np.save(STAGE8/'digits_all.npy', a)
    (STAGE8/'digits_all.json').write_text(json.dumps(dict(names=list(cols), shape=list(a.shape)), indent=2))
    print('digits', a.shape, flush=True)


if __name__ == '__main__':
    main()
