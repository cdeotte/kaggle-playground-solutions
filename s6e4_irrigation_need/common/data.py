"""Data loading, folds and shared paths for every model.

Put the competition files (train.csv, test.csv, sample_submission.csv) and the original dataset (irrigation_prediction.csv)
in ../data, or point the S6E4_DATA environment variable to the folder that holds them."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('S6E4_DATA', ROOT / 'data'))
MODELS_DIR = ROOT / 'models'
ORIGINAL_FILE = 'irrigation_prediction.csv'
ID, TARGET = 'id', 'Irrigation_Need'
CLASSES = ('Low', 'Medium', 'High')
TARGET_MAP = {c: i for i, c in enumerate(CLASSES)}
N_FOLDS, FOLD_SEED = 5, 42
CATEGORICALS = ('Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Water_Source',
                'Mulching_Used', 'Region')
NUMERICS = ('Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 'Temperature_C', 'Humidity',
            'Rainfall_mm', 'Sunlight_Hours', 'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm')


def read_train():
    return pd.read_csv(DATA_DIR / 'train.csv')


def read_test():
    return pd.read_csv(DATA_DIR / 'test.csv')


def read_original():
    """The 10,000-row original dataset (same columns as train.csv, no id)."""
    return pd.read_csv(DATA_DIR / ORIGINAL_FILE)


def labels(series):
    """Low / Medium / High -> 0 / 1 / 2 (int64)."""
    return pd.Series(series).map(TARGET_MAP).astype(np.int64).to_numpy()


def target():
    return labels(read_train()[TARGET])


def folds(y=None):
    """Fold id (0-4) of every train row: StratifiedKFold(5, shuffle=True, random_state=42) on the target.
    Every level-1 model uses these folds."""
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
