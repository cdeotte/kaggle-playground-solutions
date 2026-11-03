#!/usr/bin/env python
"""pairlift_tabm: the imputation-first TabM (pytabkit TabM_D, k = 32, ten inner early-stopping members) + the 10
pair-lift columns.

Features per outer fold, in this order (354, 58 categorical):
  row-local blocks of the raw row (lib/tabm_features.py): raw, exact_grid_onehots, smooth_latent_geometry,
      missing_gated_views, fixed_bins                                                        (69)
  nncomplete__*: raw, smooth_latent_geometry and fixed_bins recomputed on the neurally completed row (missing
      numerics <- hybrid XGBoost / autoencoder completion, missing categoricals <- argmax of the mean category
      probabilities)                                                                         (46)
  nnmissing__*: 12 missing flags of the raw row                                              (12)
  strong__* forensic core of the shared fold cache (incl. the selected base margin / probability)   (104)
  reconstruction__<compact 71> cross-fitted reconstruction view                              (71)
  density_response__<30> density-completion responses                                       (30)
  10 pair-lift columns                                                                       (10)
  12 sparse pair contrasts (fold state)                                                      (12)
Numeric NaNs -> outer-training medians. TabM: d_block 512, 2 blocks, PWL numeric embeddings (48 bins, d 16),
quantile_tabr transform, dropout 0.1, lr 0.002, batch 1024, up to 40 epochs with patience 8 on 1 - AUC, AMP;
n_cv = 10 members, each early-stopping on its own 2.5% stratified holdout of the outer-training rows
(lib/inner_splits.py, seed 20260824 + fold); random_state 142 + fold. Test = mean of the 5 fold models.
Usage: python train_tabm.py [--folds 0,1,2,3,4]"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, read_test, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import pairlift  # noqa: E402
from lib.inner_splits import INNER_SPLIT_BASE_SEED, repeated_stratified_holdouts  # noqa: E402
from lib.paths import fold_indices  # noqa: E402
from lib.reconstruction_views import completed_predictors, missing_flags  # noqa: E402
from lib.tabm_features import build_rowwise_features, prepare_fold_frames  # noqa: E402
from lib.upstream import (FoldCache, density_completion_names, load_density_completion,  # noqa: E402
                          load_reconstruction, reconstruction_names)

PAIR = 'pairlift_tabm'
BLOCKS = ['raw', 'exact_grid_onehots', 'smooth_latent_geometry', 'missing_gated_views', 'fixed_bins',
          'sparse_pair_contrast_12']
COMPLETION_BLOCKS = ['raw', 'smooth_latent_geometry', 'fixed_bins']
FILL_SOURCE = 'hybrid'
PARAMS = {
    'allow_amp': True, 'arch_type': 'tabm', 'batch_size': 1024, 'compile_model': False, 'd_block': 512,
    'd_embedding': 16, 'dropout': 0.1, 'gradient_clipping_norm': 1.0, 'lr': 0.002, 'n_blocks': 2, 'n_cv': 10,
    'n_epochs': 40, 'n_refit': 0, 'n_threads': 4, 'num_emb_n_bins': 48, 'num_emb_type': 'pwl', 'patience': 8,
    'share_training_batches': False, 'tabm_k': 32, 'tfms': ['quantile_tabr'], 'val_fraction': 0.025,
    'val_metric_name': '1-auc_ovr', 'verbosity': 0, 'weight_decay': 0.0,
}


def rowwise(frame, reconstruction, recon_names):
    """Raw-row blocks + the completed-row blocks (prefix nncomplete__) + missing flags."""
    output, cats = build_rowwise_features(frame, BLOCKS)
    completed = completed_predictors(frame.reset_index(drop=True), reconstruction, recon_names, FILL_SOURCE)
    completed_output, completed_cats = build_rowwise_features(completed, COMPLETION_BLOCKS)
    output = pd.concat([output, completed_output.add_prefix('nncomplete__'), missing_flags(frame)], axis=1)
    return output, cats + [f'nncomplete__{c}' for c in completed_cats]


def appended(frame, cache, fold, role, reconstruction, density, recon_names, density_names):
    strong_idx = cache.numeric_indices(['strong'])
    parts = [frame.reset_index(drop=True),
             pd.DataFrame(np.asarray(cache.array(role, 'numeric')[:, strong_idx], np.float32),
                          columns=[cache.numeric_names[i] for i in strong_idx]),
             pd.DataFrame(np.asarray(reconstruction[:, :71], np.float32),
                          columns=[f'reconstruction__{n}' for n in recon_names[:71]]),
             pd.DataFrame(np.asarray(density, np.float32), columns=[f'density_response__{n}' for n in density_names])]
    return pairlift.append(pd.concat(parts, axis=1), fold, role)


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_fold(fold, y, fold_ids, train_raw, test_raw):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from pytabkit import TabM_D_Classifier

    train_idx, valid_idx = fold_indices(fold, fold_ids)
    cache = FoldCache(fold)
    recon_names, density_names = reconstruction_names(), density_completion_names()
    recon_full, recon_test = load_reconstruction(fold, train_idx, valid_idx, len(y))
    density_full, density_test = load_density_completion(fold, train_idx, valid_idx, len(y))

    base_full, cats = rowwise(train_raw, recon_full, recon_names)
    train_frame = appended(base_full.iloc[train_idx], cache, fold, 'train', recon_full[train_idx],
                           density_full[train_idx], recon_names, density_names)
    valid_frame = appended(base_full.iloc[valid_idx], cache, fold, 'valid', recon_full[valid_idx],
                           density_full[valid_idx], recon_names, density_names)
    base_test, _ = rowwise(test_raw, recon_test, recon_names)
    test_frame = appended(base_test, cache, fold, 'test', recon_test, density_test, recon_names, density_names)
    del base_full, base_test
    train_frame, valid_frame, test_frame, fold_cats = prepare_fold_frames(train_frame, valid_frame, test_frame, cats,
                                                                          BLOCKS)
    val_idxs = repeated_stratified_holdouts(y[train_idx], PARAMS['n_cv'], PARAMS['val_fraction'],
                                            INNER_SPLIT_BASE_SEED + fold)
    seed = 142 + fold
    seed_everything(seed, torch)
    model = TabM_D_Classifier(**PARAMS, device='cuda:0', random_state=seed)
    model.fit(train_frame, y[train_idx], cat_col_names=fold_cats, val_idxs=val_idxs)
    valid_pred = model.predict_proba(valid_frame)[:, 1].astype(np.float64)
    test_pred = model.predict_proba(test_frame)[:, 1].astype(np.float64)
    print(f'fold {fold}: {train_frame.shape[1]} features ({len(fold_cats)} categorical), '
          f'AUC {roc_auc_score(y[valid_idx], valid_pred):.6f}', flush=True)
    return valid_pred, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    y = target()
    fold_ids = folds(y)
    train_raw, test_raw = read_train(), read_test()
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        valid_pred, test_pred = run_fold(fold, y, fold_ids, train_raw, test_raw)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
    assemble(HERE, PAIR, fold_ids, y)


if __name__ == '__main__':
    main()
