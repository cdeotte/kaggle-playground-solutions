#!/usr/bin/env python
"""XGBoost on 49 base columns + 24 exact-value target encodings at two smoothings (pair xgb_dual_smooth_te).

Features: lib/xgb_features.py (73 columns).
Per outer fold f: the round count is the rounded mean best iteration of 5 inner models (StratifiedKFold(5, shuffle=True,
random_state=2042 + f) on the outer-train rows; early stopping on the inner validation part, patience 100, at most 8000
rounds); the model is then refit on all outer-train rows for that many rounds. One seed (42).
XGBoost: binary:logistic, hist on CUDA, lr 0.02, max_depth 8, min_child_weight 60, subsample 0.8, colsample_bytree 0.8,
lambda 2, alpha 0, max_bin 1023, num_parallel_tree 5. Test = mean of the 5 fold models.
Usage: python train_xgb.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import folds as fold_ids, target  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import xgb_features as XF  # noqa: E402

PAIR = 'xgb_dual_smooth_te'
SEED_INNER = 2042
PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": "cuda", "nthread": 8,
          "seed": 42, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.0, "max_depth": 8, "min_child_weight": 60,
          "reg_lambda": 2.0, "max_bin": 1023, "num_parallel_tree": 5, "learning_rate": 0.02}
ES_ROUNDS = 100
ROUND_CAP = 8000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    todo = [f for f in fold_list(a.folds) if not have_part(HERE, PAIR, f)]
    y = target()
    folds = fold_ids(y)
    if todo:
        Xb_tr, Xb_te, te_oof, te_te, y, folds, names = XF.build()
        print(f'{len(names)} features', flush=True)

    for f in todo:
        t0 = time.time()
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx].astype(np.float32)
        X_trn = np.hstack([Xb_tr[trn_idx], te_oof[f][trn_idx]]).astype(np.float32)
        X_val = np.hstack([Xb_tr[val_idx], te_oof[f][val_idx]]).astype(np.float32)
        D_test = xgb.DMatrix(np.hstack([Xb_te, te_te[f]]).astype(np.float32))

        # inner 5-fold CV on the outer-train rows selects the round count
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED_INNER + f)
        best_iters, inner_aucs = [], []
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y_trn):
            m = xgb.train(PARAMS, xgb.DMatrix(X_trn[i_fit], label=y_trn[i_fit]), ROUND_CAP,
                          evals=[(xgb.DMatrix(X_trn[i_app], label=y_trn[i_app]), "v")],
                          early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
            best_iters.append(m.best_iteration)
            p = m.predict(xgb.DMatrix(X_trn[i_app]), iteration_range=(0, m.best_iteration + 1))
            inner_aucs.append(roc_auc_score(y_trn[i_app], p))
        refit_rounds = int(round(float(np.mean(best_iters))))

        # refit on all outer-train rows at that round count, no evaluation set
        final = xgb.train(PARAMS, xgb.DMatrix(X_trn, label=y_trn), refit_rounds, verbose_eval=False)
        pv = final.predict(xgb.DMatrix(X_val)).astype(np.float64)
        pt = final.predict(D_test).astype(np.float64)
        save_part(HERE, PAIR, f, pv, pt)
        print(f'fold {f}: inner AUC {np.mean(inner_aucs):.6f}  best iters {best_iters} -> refit {refit_rounds} rounds  '
              f'val AUC {roc_auc_score(y[val_idx], pv):.6f}  ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, PAIR, folds, y)


if __name__ == '__main__':
    main()
