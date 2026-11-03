#!/usr/bin/env python
"""TabM recipes on lib/tabm_harness.py: fold-local features (base, artifact digits, source flag, group stats, target encodings
of Race x Compound and Race x Year), competition rows + all original rows, early stopping on the validation fold.
One model per fold (pytabkit n_cv 2: two TabM models on the same split, averaged); test = mean of the 5 fold models.

Recipes (pair name):
  tabm_group_stats_digits_d448         tabm, k 32, 4 blocks x 448, pwl embeddings d 24 / 64 bins, lr 3.5e-4; seed offset 6700
  tabm_group_stats_digits_d512         tabm, k 32, 4 blocks x 512, pwl embeddings d 32 / 96 bins, lr 3.2e-4; seed offset 7500
  tabm_group_stats_digits_d512_reseed  the d512 model with seed offset 8900
All: dropout 0.1, weight decay 3e-5, batch 512, up to 55 epochs, patience 9, AMP, gradient clipping 1.
Usage: python train.py <pair> [--folds 0,1,2,3,4] [--force]
Folds can run in separate processes; the pair is written once all 5 folds are done."""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.tabm.lib.tabm_harness import TabmConfig, load_data, run_fold  # noqa: E402

D512 = {"arch_type": "tabm", "tabm_k": 32, "n_cv": 2, "n_blocks": 4, "d_block": 512, "d_embedding": 32, "num_emb_n_bins": 96,
        "dropout": 0.10, "lr": 3.2e-4, "weight_decay": 3e-5, "batch_size": 512, "n_epochs": 55, "patience": 9}
RECIPES = {
    "tabm_group_stats_digits_d448": TabmConfig(
        model_seed_offset=6700,
        params_override={"arch_type": "tabm", "tabm_k": 32, "n_cv": 2, "n_blocks": 4, "d_block": 448, "d_embedding": 24,
                         "num_emb_n_bins": 64, "dropout": 0.10, "lr": 3.5e-4, "weight_decay": 3e-5, "batch_size": 512,
                         "n_epochs": 55, "patience": 9}),
    "tabm_group_stats_digits_d512": TabmConfig(model_seed_offset=7500, params_override=dict(D512)),
    "tabm_group_stats_digits_d512_reseed": TabmConfig(model_seed_offset=8900, params_override=dict(D512)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(RECIPES))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--force", action="store_true", help="rerun folds that are already done")
    args = ap.parse_args()
    cfg = RECIPES[args.pair]
    y = target()
    fold_ids = folds(y)
    todo = [k for k in fold_list(args.folds) if args.force or not have_part(HERE, args.pair, k)]
    if todo:
        data = load_data()
        for k in todo:
            va_pred, te_pred, n_features = run_fold(cfg, data, fold_ids, k)
            save_part(HERE, args.pair, k, np.clip(va_pred, 1e-7, 1 - 1e-7), np.clip(te_pred, 1e-7, 1 - 1e-7))
            print(f"{args.pair} fold {k}: {n_features} features, AUC {roc_auc_score(y[fold_ids == k], va_pred):.6f}", flush=True)
    assemble(HERE, args.pair, fold_ids, y)


if __name__ == "__main__":
    main()
