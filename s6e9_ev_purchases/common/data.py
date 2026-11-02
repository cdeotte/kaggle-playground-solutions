"""Data loading and folds shared by every model.

Put the competition files (train.csv, test.csv, sample_submission.csv) and the original dataset
(EV_Adoption_and_Range_Anxiety_Dataset.csv) in ../data, or point the S6E9_DATA environment variable to the folder that
holds them."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('S6E9_DATA', ROOT / 'data'))
ORIGINAL_FILE = 'EV_Adoption_and_Range_Anxiety_Dataset.csv'
TARGET = 'Will_Buy_EV'
N_FOLDS, FOLD_SEED = 5, 42
CATEGORICALS = ('Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level')


def read_train():
    return pd.read_csv(DATA_DIR / 'train.csv')


def read_test():
    return pd.read_csv(DATA_DIR / 'test.csv')


def read_original():
    """The 10,000-row original dataset the competition data was generated from (Buyer_ID dropped)."""
    return pd.read_csv(DATA_DIR / ORIGINAL_FILE).drop(columns=['Buyer_ID'], errors='ignore')


def labels(series):
    """Yes/No (or bool, 0/1) labels -> int8 0/1."""
    if series.dtype == bool or series.dtype.kind in 'iu':
        return series.astype(np.int8).to_numpy()
    m = series.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0})
    assert m.notna().all(), 'unknown label values'
    return m.astype(np.int8).to_numpy()


def target():
    return labels(read_train()[TARGET])


def folds(y=None):
    """Fold id (0-4) of every train row: StratifiedKFold(5, shuffle=True, random_state=42) on the target.
    Every model and the stacker use these folds."""
    y = target() if y is None else y
    f = np.full(len(y), -1, np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(y)), y)):
        f[va] = k
    return f


def raw_arrays(include_original=False):
    """The 13 raw features as float32 arrays in train.csv column order. Categoricals are coded 0..k-1 by their sorted
    levels over train+test; NaN is kept. Returns X_train, X_test, [X_original,] meta (features, cat_idx, cat_codes)."""
    tr, te = read_train(), read_test()
    feats = [c for c in tr.columns if c not in ('id', TARGET)]
    cats = [c for c in feats if c in CATEGORICALS]
    codes = {c: sorted(set(tr[c].dropna()) | set(te[c].dropna())) for c in cats}

    def encode(df):
        X = df[feats].copy()
        for c, lev in codes.items():
            X[c] = X[c].map({v: i for i, v in enumerate(lev)})
        return X.to_numpy(np.float32)
    out = [encode(tr), encode(te)]
    if include_original:
        out.append(encode(read_original()))
    meta = dict(features=feats, cat_idx=[feats.index(c) for c in cats], cat_codes=codes)
    return (*out, meta)
