"""Data loading, folds and shared paths for every model.

Put the competition files (train.csv, test.csv, sample_submission.csv) and the original dataset (f1_strategy_dataset_v4.csv)
in ../data, or point the S6E5_DATA environment variable to the folder that holds them."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('S6E5_DATA', ROOT / 'data'))
MODELS_DIR = ROOT / 'models'
ORIGINAL_FILE = 'f1_strategy_dataset_v4.csv'
ID, TARGET = 'id', 'PitNextLap'
N_FOLDS, FOLD_SEED = 5, 42
FEATURES = ('Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife', 'Position', 'LapTime (s)',
            'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change')
CATEGORICALS = ('Driver', 'Compound', 'Race')


def read_train():
    return pd.read_csv(DATA_DIR / 'train.csv')


def read_test():
    return pd.read_csv(DATA_DIR / 'test.csv')


def read_original():
    """The 101,371-row original dataset, in train.csv column order (no id). Normalized_TyreLife, a column the competition
    data does not have, is dropped."""
    df = pd.read_csv(DATA_DIR / ORIGINAL_FILE).drop(columns=['Normalized_TyreLife'], errors='ignore')
    return df[list(FEATURES) + [TARGET]]


def labels(series):
    """0/1 (or bool) labels -> int8 0/1."""
    return pd.Series(series).astype(np.int8).to_numpy()


def target():
    return labels(read_train()[TARGET])


def _fold_ids(y):
    f = np.full(len(y), -1, np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(y)), y)):
        f[va] = k
    return f


def folds(y=None):
    """Fold id (0-4) of every train row: StratifiedKFold(5, shuffle=True, random_state=42) on the target.
    Every model and the stacker use these folds."""
    return _fold_ids(target() if y is None else y)


def orig_folds(y_orig=None):
    """Fold id (0-4) of every original row, from the same StratifiedKFold on the original target. Models that append
    original rows to the training side of outer fold k use the original rows whose id is not k (the fold-matched 4/5
    subset); others append all original rows."""
    return _fold_ids(read_original()[TARGET].to_numpy() if y_orig is None else y_orig)


def model_dir(name):
    """models/<name>/"""
    return MODELS_DIR / name


def cache_dir(name):
    """models/<name>/cache/: intermediate files written by the scripts in models/<name>/. Never committed."""
    d = MODELS_DIR / name / 'cache'
    d.mkdir(parents=True, exist_ok=True)
    return d
