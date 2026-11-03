"""Data loading, folds and shared paths for every model.

Put the competition files (train.csv, test.csv, sample_submission.csv) and the original dataset
(WA_Fn-UseC_-Telco-Customer-Churn.csv) in ../data, or point the S6E3_DATA environment variable to the folder that holds
them."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('S6E3_DATA', ROOT / 'data'))
MODELS_DIR = ROOT / 'models'
ORIGINAL_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
ID, TARGET = 'id', 'Churn'
N_FOLDS, FOLD_SEED = 5, 42
NUMERICS = ('tenure', 'MonthlyCharges', 'TotalCharges')
CATEGORICALS = ('gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod')


def read_train():
    return pd.read_csv(DATA_DIR / 'train.csv')


def read_test():
    return pd.read_csv(DATA_DIR / 'test.csv')


def read_original():
    """The original IBM Telco dataset (7,043 rows, customerID dropped). TotalCharges is blank for 11 new customers: it is
    coerced to numeric (NaN); each model handles those rows as its source notebook did."""
    df = pd.read_csv(DATA_DIR / ORIGINAL_FILE).drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def labels(series):
    """Yes / No (or 0 / 1) -> int8 0 / 1."""
    s = pd.Series(series)
    if s.dtype.kind in 'iub':
        return s.astype(np.int8).to_numpy()
    m = s.astype(str).str.strip().map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
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
    """models/<name>/cache/: intermediate files written by the scripts in models/<name>/. Never committed."""
    d = MODELS_DIR / name / 'cache'
    d.mkdir(parents=True, exist_ok=True)
    return d
