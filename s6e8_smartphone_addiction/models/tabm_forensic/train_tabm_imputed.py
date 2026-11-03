#!/usr/bin/env python
"""TabM on a dual raw / neurally-completed view + forensic columns + reconstruction and density-completion channels.

Features (344, 58 categorical):
  the row-local blocks of the raw rows (raw, exact_grid_onehots, smooth_latent_geometry, missing_gated_views,
  fixed_bins); the raw, smooth_latent_geometry and fixed_bins blocks of the completed rows ('nncomplete__'; missing
  numerics from the hybrid XGBoost / autoencoder reconstruction, missing categoricals from the mean of the two class
  probability vectors); 12 missing flags; the 104 'strong__' columns of the ../lgbm_margin fold cache; the first 71
  reconstruction diagnostics ('compact71') and all 30 density-completion response channels ('full30') of
  ../realmlp_repair; 12 fold-fitted pair contrasts.
TabM-D (lib/runner.py) with n_cv = 10 internal members, each early-stopped on its own stratified 2.5% holdout of the
outer-training rows. Test = mean of the 5 fold models.
Usage: python train_tabm_imputed.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, read_train, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.tabm_forensic.lib.completion import dual_rowwise_features, load_density, load_reconstruction  # noqa: E402
from models.tabm_forensic.lib.forensic_cache import load_fold  # noqa: E402
from models.tabm_forensic.lib.runner import MODEL_DEFAULTS, configure_environment, run_fold  # noqa: E402

PAIR = "tabm_forensic_imputed"
MODEL_PARAMS = dict(MODEL_DEFAULTS, n_cv=10, val_fraction=0.025)
COMPLETION_BLOCKS = ["raw", "smooth_latent_geometry", "fixed_bins"]
RECONSTRUCTION_VIEW = 71  # 'compact71': the first 71 reconstruction columns


def fold_inputs(fold):
    n_train = len(read_train())
    recon_names, recon_full, recon_test = load_reconstruction(fold, n_train)
    density_names, density_full, density_test = load_density(fold, n_train)
    _, train_idx, valid_idx = load_fold(fold)
    recon_rows = {"train": recon_full[train_idx], "valid": recon_full[valid_idx], "test": recon_test}
    density_rows = {"train": density_full[train_idx], "valid": density_full[valid_idx], "test": density_test}

    def rowwise_builder(frame, blocks, split):
        reconstruction = recon_full if split == "train" else recon_test
        return dual_rowwise_features(frame, reconstruction, recon_names, blocks, COMPLETION_BLOCKS)

    def append_extra(frame, role):
        recon_cols = [f"reconstruction__{name}" for name in recon_names[:RECONSTRUCTION_VIEW]]
        recon = pd.DataFrame(np.asarray(recon_rows[role][:, :RECONSTRUCTION_VIEW], np.float32), columns=recon_cols)
        density = pd.DataFrame(np.asarray(density_rows[role], np.float32),
                               columns=[f"density_response__{name}" for name in density_names])
        return pd.concat([frame.reset_index(drop=True), recon, density], axis=1)

    return rowwise_builder, append_extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    configure_environment()
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        rowwise_builder, append_extra = fold_inputs(fold)
        valid_pred, test_pred, y_valid = run_fold(fold, MODEL_PARAMS, rowwise_builder=rowwise_builder,
                                                  append_extra=append_extra)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
        from sklearn.metrics import roc_auc_score
        print(f"{PAIR} fold {fold}: AUC {roc_auc_score(y_valid, valid_pred):.6f}", flush=True)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
