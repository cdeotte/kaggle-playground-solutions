#!/usr/bin/env python
"""LightGBM recipes on lib/lgbm_harness.py: competition rows + all original rows (weighted), label-free features built on
train + original + test rows, early stopping on the validation fold. One model per fold; test = mean of the 5 fold models.

Recipes (pair name):
  lgbm_domain_counts_te_original_w035           domain + interactions + counts + per-fold target encodings (smooth 30);
                                                original weight 0.35; 96 leaves, feature_fraction 0.84, lambda_l2 0.9
  lgbm_domain_counts_group_stats_original_w085  domain + interactions + counts + group stats; original weight 0.85;
                                                104 leaves, feature_fraction 0.84, lambda_l2 1.0
  lgbm_lean_original_w105                       domain + interactions + counts (55 features); original weight 1.05;
                                                104 leaves, feature_fraction 0.10, bagging 0.92, lambda_l2 1.8
  lgbm_lean_original_w125                       the same with original weight 1.25
  lgbm_lean_original_w125_slow                  the same with learning rate 0.012 and up to 9,500 rounds
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
from models.lightgbm.lib.lgbm_harness import LgbmConfig, build_data, run_fold  # noqa: E402

FEATURES = dict(add_domain_features=True, add_interaction_cats=True, add_count_features=True)
LEAN_PARAMS = {"learning_rate": 0.018, "num_leaves": 104, "max_depth": -1, "feature_fraction": 0.10, "bagging_fraction": 0.92,
               "bagging_freq": 5, "lambda_l1": 0.10, "lambda_l2": 1.8, "n_estimators": 6500, "min_child_samples": 80}
RECIPES = {
    "lgbm_domain_counts_te_original_w035": LgbmConfig(
        original_weight=0.35, **FEATURES, add_target_encoding=True, te_smooth=30.0,
        lgbm_params_override={"learning_rate": 0.018, "num_leaves": 96, "max_depth": -1, "feature_fraction": 0.84,
                              "bagging_fraction": 0.82, "bagging_freq": 5, "lambda_l1": 0.05, "lambda_l2": 0.9,
                              "n_estimators": 5500, "min_child_samples": 60}),
    "lgbm_domain_counts_group_stats_original_w085": LgbmConfig(
        original_weight=0.85, **FEATURES, add_group_stats=True,
        lgbm_params_override={"learning_rate": 0.018, "num_leaves": 104, "max_depth": -1, "feature_fraction": 0.84,
                              "bagging_fraction": 0.82, "bagging_freq": 5, "lambda_l1": 0.04, "lambda_l2": 1.0,
                              "n_estimators": 6000, "min_child_samples": 60}),
    "lgbm_lean_original_w105": LgbmConfig(original_weight=1.05, **FEATURES, lgbm_params_override=dict(LEAN_PARAMS)),
    "lgbm_lean_original_w125": LgbmConfig(original_weight=1.25, **FEATURES, lgbm_params_override=dict(LEAN_PARAMS)),
    "lgbm_lean_original_w125_slow": LgbmConfig(
        original_weight=1.25, **FEATURES,
        lgbm_params_override=dict(LEAN_PARAMS, learning_rate=0.012, n_estimators=9500)),
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
        data = build_data(cfg)
        for k in todo:
            va_pred, te_pred, best_iter, n_features = run_fold(cfg, data, fold_ids, k)
            save_part(HERE, args.pair, k, np.clip(va_pred, 1e-7, 1 - 1e-7), np.clip(te_pred, 1e-7, 1 - 1e-7))
            print(f"{args.pair} fold {k}: {n_features} features, best_iteration {best_iter}, "
                  f"AUC {roc_auc_score(y[fold_ids == k], va_pred):.6f}", flush=True)
    assemble(HERE, args.pair, fold_ids, y)


if __name__ == "__main__":
    main()
