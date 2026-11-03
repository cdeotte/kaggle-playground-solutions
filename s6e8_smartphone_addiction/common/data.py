"""Data loading, folds and shared paths for every model.

Put the competition files (train.csv, test.csv, sample_submission.csv) and the original dataset
(Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv) in ../data, or point the S6E8_DATA environment variable to the
folder that holds them."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('S6E8_DATA', ROOT / 'data'))
MODELS_DIR = ROOT / 'models'
ORIGINAL_FILE = 'Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'
TARGET = 'addicted_label'
N_FOLDS, FOLD_SEED = 5, 42
NUMERICS = ('age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 'work_study_hours', 'sleep_hours',
            'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time')
CATEGORICALS = ('gender', 'stress_level', 'academic_work_impact')
FEATURES = NUMERICS + CATEGORICALS


def read_train():
    return pd.read_csv(DATA_DIR / 'train.csv')


def read_test():
    return pd.read_csv(DATA_DIR / 'test.csv')


def read_original(keep_level=False):
    """The 7,500-row original dataset the competition data was generated from. transaction_id and user_id are dropped;
    addiction_level (a field the competition data does not have) is kept only with keep_level=True."""
    df = pd.read_csv(DATA_DIR / ORIGINAL_FILE).drop(columns=['transaction_id', 'user_id'], errors='ignore')
    return df if keep_level else df.drop(columns=['addiction_level'], errors='ignore')


def labels(series):
    """0/1 (or bool, Yes/No) labels -> int8 0/1."""
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


def model_dir(name):
    """models/<name>/"""
    return MODELS_DIR / name


def cache_dir(name):
    """models/<name>/cache/: intermediate files (feature matrices, embeddings, density-model outputs) written by the
    scripts in models/<name>/ and read by other folders. Never committed."""
    d = MODELS_DIR / name / 'cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_arrays(include_original=False):
    """The 12 raw features as float32 arrays in train.csv column order. Categoricals are coded 0..k-1 by their sorted
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
