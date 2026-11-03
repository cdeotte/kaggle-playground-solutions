#!/usr/bin/env python
"""XGBoost on the 102-column forensic matrix (lib/strong.py) with a base margin.

Base margin: logit(prior) + 0.40 raw_te_woe_sum + 0.10 unsupported-pair weighted mean (computed the same way for the
training, validation and test rows). XGBoost (GPU hist): 12,000 rounds, lr 0.006875, depth 6, max_bin 2048,
min_child_weight 1, subsample 0.8, colsample_bytree 0.8, alpha 0.05, lambda 10, seed 42, native categoricals;
no early stopping. One model per outer fold; test = mean of the 5 fold models.
Usage: python train_forensic_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.xgb_forensic.lib.strong import XGB_PARAMS, base_margin, build_strong_fold  # noqa: E402

PAIR = "xgb_forensic_woe_margin"


def run_fold(fold):
    import xgboost as xgb
    s = build_strong_fold(fold)
    margins = [base_margin(frame, s.prior) for frame in (s.train, s.valid, s.test)]
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(s.train, s.y_train, base_margin=margins[0], verbose=False)
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
