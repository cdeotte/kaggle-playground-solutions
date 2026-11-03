#!/usr/bin/env python
"""Per-fold 282-column matrix of the nested-margin random forest (cuml_rf_nested_margins):
  99   cuML-built columns (cache/cf_f<k>_*.parquet from build_cuml_features.py)
  135  masked-language-model channels denv4, denv5, mfill4, qfill, rrd, cmp4 (../density_features)
  48   lift-bank block plift, cbank, wbank, wbank12, sbank, ebank (../lift_banks)
NaN -> outer-train median, then inf / NaN -> 0. Writes cache/M_f<k>.npy (train rows then test rows, float32) and
cache/M_f<k>.cols.json (column names).
Usage: python build_rf_matrix.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import cache_dir, folds as fold_ids, target  # noqa: E402
from common.pairs import fold_list  # noqa: E402
from lib.channels import append_channels, read_channel  # noqa: E402

MLM = ['denv4', 'denv5', 'mfill4', 'qfill', 'rrd', 'cmp4']
BANKS = ['plift', 'cbank', 'wbank', 'wbank12', 'sbank', 'ebank']
CACHE = cache_dir('cuml_density_models')


def banks_block(fold, split):
    """The 6 lift banks of one fold as one block (columns <bank>_<column>, duplicates dropped)."""
    D = pd.concat([read_channel(nm, fold, split) for nm in BANKS], axis=1)
    return D.loc[:, ~D.columns.duplicated()]


def build(fold, folds):
    t0 = time.time()
    trn_idx = np.where(folds != fold)[0]
    A = pd.read_parquet(CACHE / f'cf_f{fold}_tr.parquet')
    B = pd.read_parquet(CACHE / f'cf_f{fold}_te.parquet')
    A, B = append_channels(A, B, MLM, fold)
    Ptr, Pte = banks_block(fold, 'tr'), banks_block(fold, 'te')
    dup = [c for c in Ptr.columns if c in A.columns]
    if dup:
        Ptr, Pte = Ptr.drop(columns=dup), Pte.drop(columns=dup)
    A, B = pd.concat([A, Ptr], axis=1), pd.concat([B, Pte], axis=1)
    cols = list(A.columns)
    med = A.iloc[trn_idx].median()
    Xa = A.fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
    Xb = B.fillna(med).replace([np.inf, -np.inf], 0.0).fillna(0.0).values.astype(np.float32)
    X = np.vstack([Xa, Xb])
    np.save(CACHE / f'M_f{fold}.npy', X)
    json.dump(cols, open(CACHE / f'M_f{fold}.cols.json', 'w'))
    print(f'[{time.time() - t0:.0f}s] fold {fold}: matrix {X.shape} -> {CACHE}/M_f{fold}.npy', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    folds = fold_ids(target())
    for k in fold_list(a.folds):
        build(k, folds)


if __name__ == '__main__':
    main()
