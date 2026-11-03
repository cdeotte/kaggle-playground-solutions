#!/usr/bin/env python
"""pairlift_xgb: the forensic GPU XGBoost (12,000 fixed rounds on a base margin) after masked-autoencoder imputation,
+ the 10 pair-lift columns.

Features per outer fold, in this order (137):
  102 forensic columns of ../xgb_forensic (lib/strong.build_strong_fold: raw columns with native categoricals,
      screen-budget features, frequencies, nested target encodings + WOE sums, partial-budget residuals, screen imputer,
      screen responses, observed-pair PMIs, original copula ratios, unsupported-pair response)
  25 masked-autoencoder fill columns (completed numerics, categorical probabilities if missing, completed screen
      geometry; row-excluded cross-fit on the outer-training rows)
  10 pair-lift columns
Base margin = logit(prior) + 0.40 * raw_te_woe_sum + 0.10 * unsupported-pair response, passed to fit and to every
prediction. Same XGBoost parameters as the forensic XGBoost (lr 0.006875, depth 6, max_bin 2048, subsample / colsample
0.8, alpha 0.05, lambda 10, native categoricals, random_state 42); no early stopping; test = mean of the 5 fold models.
Usage: python train_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from models.xgb_forensic.lib.strong import XGB_PARAMS, base_margin, build_strong_fold  # noqa: E402
from lib import pairlift  # noqa: E402
from lib.upstream import IMPUTATION_FILL_NAMES, load_imputation_fill  # noqa: E402

PAIR = 'pairlift_xgb'


def model_frame(frame, fold, role):
    """forensic columns + autoencoder fill view + pair lift."""
    fill = pd.DataFrame(load_imputation_fill(fold, role), columns=IMPUTATION_FILL_NAMES)
    return pairlift.append(pd.concat([frame.reset_index(drop=True), fill], axis=1), fold, role)


def run_fold(fold):
    import xgboost as xgb

    s = build_strong_fold(fold)
    margins = {role: base_margin(frame, s.prior) for role, frame in (('train', s.train), ('valid', s.valid),
                                                                     ('test', s.test))}
    frames = {role: model_frame(frame, fold, role) for role, frame in (('train', s.train), ('valid', s.valid),
                                                                       ('test', s.test))}
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(frames['train'], s.y_train, base_margin=margins['train'], verbose=False)
    valid_pred = np.asarray(model.predict_proba(frames['valid'], base_margin=margins['valid'])[:, 1], np.float64)
    test_pred = np.asarray(model.predict_proba(frames['test'], base_margin=margins['test'])[:, 1], np.float64)
    print(f'fold {fold}: {frames["train"].shape[1]} features, AUC {roc_auc_score(s.y_valid, valid_pred):.6f}',
          flush=True)
    return valid_pred, test_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    args = ap.parse_args()
    for fold in fold_list(args.folds):
        if have_part(HERE, PAIR, fold):
            continue
        valid_pred, test_pred = run_fold(fold)
        save_part(HERE, PAIR, fold, valid_pred, test_pred)
    y = target()
    assemble(HERE, PAIR, folds(y), y)


if __name__ == '__main__':
    main()
