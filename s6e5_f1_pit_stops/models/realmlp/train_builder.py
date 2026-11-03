#!/usr/bin/env python
"""RealMLP on the per-fold feature builder of lib/fold_builder.py (frequency, anchor-statistic and original-prior blocks,
nested target encodings, fold-matched original rows appended).

Recipes (pair name):
  realmlp_sequence_anchor_priors  + race-stint sequence features; 14 target-encoded keys x smoothing 10 / 50 (28 columns);
                                  460 columns in all; mish with parametric activations, [768, 384, 192], lr 0.024,
                                  embedding 8, PLR sigma 1.0, up to 72 epochs
  realmlp_pair_grid_te            + pair grid of 9 categorical keys; 24 target-encoded keys x smoothing 5 / 20 / 100
                                  (72 columns); 547 columns in all; silu, [1024, 512, 256], lr 0.021, embedding 10,
                                  PLR sigma 1.6, up to 80 epochs
Both: n_ens 16, batch 512, early stopping on the validation 1-AUC (multiplicative patience 1, additive 12 / 14), no
l2_normalize transform, seed 42 + fold.
Usage: python train_builder.py <pair>"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import save_pair  # noqa: E402
from lib.fold_builder import run  # noqa: E402

COMMON_PARAMS = {
    "batch_size": 512,
    "early_stopping_multiplicative_patience": 1,
    "lr_sched": "lin_cos_log_15",
    "ls_eps_sched": "sqrt_cos",
    "n_ens": 16,
    "p_drop_sched": "expm4t",
    "plr_act_name": "gelu",
    "plr_hidden_1": 32,
    "plr_hidden_2": 16,
    "random_state": 42,
    "sq_mom": 0.98,
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
    "use_early_stopping": True,
    "val_metric_name": "1-auc_ovr",
    "verbosity": 2,
}

SPECS = {
    "realmlp_sequence_anchor_priors": {
        "use_sequence": True,
        "use_pair_grid": False,
        "te_limit": 14,
        "te_smooths": [10, 50],
        "model_params": {**COMMON_PARAMS,
                         "act": "mish",
                         "act_lr_factor": 0.1,
                         "early_stopping_additive_patience": 12,
                         "embedding_size": 8,
                         "first_layer_lr_factor": 0.125,
                         "hidden_sizes": [768, 384, 192],
                         "lr": 0.024,
                         "ls_eps": 0.006,
                         "max_one_hot_cat_size": 18,
                         "n_epochs": 72,
                         "p_drop": 0.045,
                         "plr_lr_factor": 0.16,
                         "plr_sigma": 1.0,
                         "use_parametric_act": True,
                         "wd": 0.014},
    },
    "realmlp_pair_grid_te": {
        "use_sequence": False,
        "use_pair_grid": True,
        "te_limit": 24,
        "te_smooths": [5, 20, 100],
        "model_params": {**COMMON_PARAMS,
                         "act": "silu",
                         "early_stopping_additive_patience": 14,
                         "embedding_size": 10,
                         "first_layer_lr_factor": 0.15,
                         "hidden_sizes": [1024, 512, 256],
                         "lr": 0.021,
                         "ls_eps": 0.01,
                         "max_one_hot_cat_size": 16,
                         "n_epochs": 80,
                         "p_drop": 0.07,
                         "plr_lr_factor": 0.13,
                         "plr_sigma": 1.6,
                         "wd": 0.018},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", choices=list(SPECS))
    args = ap.parse_args()
    oof, test = run(SPECS[args.pair])
    save_pair(HERE, args.pair, oof, test)


if __name__ == "__main__":
    main()
