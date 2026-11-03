#!/usr/bin/env python
"""TabM on row-local TabM views + the 104 forensic 'strong__' columns of the ../lgbm_margin fold cache.

Features (185, 35 categorical): raw, exact_grid_onehots, smooth_latent_geometry, missing_gated_views, fixed_bins
(lib/row_features.py), the strong columns (the 102-column forensic matrix + base margin + base probability), 12
fold-fitted pair contrasts. TabM-D (lib/runner.py) with n_cv = 5 internal members, each early-stopped on its own
stratified 5% holdout of the outer-training rows. Test = mean of the 5 fold models.
Usage: python train_tabm_strong.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.tabm_forensic.lib.runner import MODEL_DEFAULTS, configure_environment, run_fold  # noqa: E402

PAIR = "tabm_forensic_strong"
MODEL_PARAMS = dict(MODEL_DEFAULTS, n_cv=5, val_fraction=0.05, verbosity=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    configure_environment()
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        valid_pred, test_pred, y_valid = run_fold(fold, MODEL_PARAMS)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
        from sklearn.metrics import roc_auc_score
        print(f"{PAIR} fold {fold}: AUC {roc_auc_score(y_valid, valid_pred):.6f}", flush=True)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
