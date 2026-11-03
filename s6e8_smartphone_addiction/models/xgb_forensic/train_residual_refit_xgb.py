#!/usr/bin/env python
"""XGBoost on the 290-column frontier matrix (lib/frontier_features.py) with a completion base margin and a
selector / full-refit training rule (lib/refit_xgb.py).

Features: the 102 forensic columns (lib/strong.py) plus, after the screen-response block, 24 fixed-original residual
source channels, the 134 lift-bank / generator-temperature / global-density / repair-posterior columns and the 10
completion aggregates of ../realmlp_repair, 14 source-tempered residual contrasts and 6 daily x social residual
surfaces.
Base margin: logit(prior) + 0.55 raw_te_woe_sum + 0.40 (completion_logit_mean - logit(prior)).
XGBoost (GPU hist): lr 0.02, depth 6, max_bin 1024, min_child_weight 20, subsample 0.7, colsample_bytree 0.7,
colsample_bynode 0.6, lambda 4, 6 parallel trees per round, seed 42. Rounds: selector on 9/10 of the outer-training
rows with early stopping on the other tenth (StratifiedKFold(10, seed 91027), split 0, cap 5,000), then a refit on all
outer-training rows for ceil(best x 10 / 9) rounds. Test = mean of the 5 fold models.
Usage: python train_residual_refit_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.xgb_forensic.lib.frontier_features import FRONTIER_PARAMS, completion_margin, frontier_parts  # noqa: E402
from models.xgb_forensic.lib.refit_xgb import fit_selector_refit  # noqa: E402
from models.xgb_forensic.lib.strong import build_strong_fold  # noqa: E402

PAIR = "xgb_forensic_original_residual_refit"


def run_fold(fold):
    s = build_strong_fold(fold, extra_parts=frontier_parts)
    margins = [completion_margin(frame, s.prior) for frame in (s.train, s.valid, s.test)]
    model = fit_selector_refit(s.train, s.y_train, margins[0], FRONTIER_PARAMS)
    valid_pred = np.asarray(model.predict_proba(s.valid, base_margin=margins[1])[:, 1], np.float64)
    test_pred = np.asarray(model.predict_proba(s.test, base_margin=margins[2])[:, 1], np.float64)
    save_part(HERE, PAIR, fold, valid_pred, test_pred)
    from sklearn.metrics import roc_auc_score
    print(f"{PAIR} fold {fold}: AUC {roc_auc_score(s.y_valid, valid_pred):.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        if not have_part(HERE, PAIR, fold):
            run_fold(fold)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == "__main__":
    main()
