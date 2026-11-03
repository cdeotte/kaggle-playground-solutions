"""Where this folder reads its upstream fold caches and writes its own.

Caches made by other repo folders:
  models/lgbm_margin      cache/features/fold_<k>/  (its fold feature builder, build_fold_features.py)
      {train,valid,test}_numeric.npy             strong forensic core (strong__*), fixed-original channels
                                                 (fixed_source__*), categorical-as-numeric channels (cat_numeric__*)
      {train,valid,test}_categories.npy          native__* / semantic__* category codes (outer-training vocabulary)
      {train,valid,test}_category_frequency.npy  log1p outer-training count of every category column
      {train,valid,test}_category_summary.npy    6 row summaries of those counts
      manifest.json                              keys base_numeric_names, category_names, category_frequency_names,
                                                 category_summary_names
  models/realmlp_repair   cache/reconstruction-production-v1/fold_<k>/{train,valid,test}_features.npy
                              cross-fitted XGBoost / masked-autoencoder reconstruction, all 189 columns in the order of
                              upstream.reconstruction_names() (compact 71, detailed 52, full 66; the autoencoder fill
                              view of the XGBoost learner is 25 of these columns)
                          cache/density_completion_response_production_features/fold_<k>/{train,valid,test}_features.npy
                              the 30 density-completion response columns (upstream.density_completion_names())
train = outer-training rows of fold k (train.csv order), valid = rows of fold k, test = test.csv order.

Written here: cache/pairlift/fold_<k>/{train,valid,test}_features.npy (build_pairlift.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data import cache_dir, folds, model_dir  # noqa: E402

NAME = 'pairlift_models'
MODEL_DIR = model_dir(NAME)


def fold_cache_dir(fold):
    return cache_dir('lgbm_margin') / 'features' / f'fold_{fold}'


def reconstruction_dir(fold):
    return cache_dir('realmlp_repair') / 'reconstruction-production-v1' / f'fold_{fold}'


def density_completion_dir(fold):
    return cache_dir('realmlp_repair') / 'density_completion_response_production_features' / f'fold_{fold}'


def pairlift_dir(fold):
    d = cache_dir(NAME) / 'pairlift' / f'fold_{fold}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def fold_indices(fold, fold_ids=None):
    """(outer-training row indices, outer-validation row indices) in train.csv order."""
    f = folds() if fold_ids is None else fold_ids
    return np.flatnonzero(f != fold).astype(np.int64), np.flatnonzero(f == fold).astype(np.int64)
