#!/usr/bin/env python
"""CatBoost (GPU) on quantile-binned numerics, conditional-median imputations and native categoricals
(pair catboost_quantile_bins).

Features: lib/cat_features.py (25 columns, fitted per outer fold on the outer-train rows; no target encodings).
Per outer fold f: the round count is the rounded mean best iteration of 5 inner models (StratifiedKFold(5, shuffle=True,
random_state=3042 + f) on the outer-train rows; early stopping on the inner validation part, patience 200, at most 9000
rounds); the model is then refit on all outer-train rows for that many rounds. One seed (42).
CatBoost: Logloss, depth 8, lr 0.03, l2_leaf_reg 3, Bernoulli bootstrap with subsample 0.8, one GPU. Test = mean of the 5
fold models' probabilities.
Usage: python train_catboost.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoost, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.data import TARGET, folds as fold_ids, read_test, read_train  # noqa: E402
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib.cat_features import CAT_COLS, fit_and_transform  # noqa: E402

PAIR = 'catboost_quantile_bins'
SEED_INNER = 3042
PARAMS = {"loss_function": "Logloss", "task_type": "GPU", "devices": "0", "bootstrap_type": "Bernoulli", "subsample": 0.8,
          "l2_leaf_reg": 3, "random_seed": 42, "thread_count": 8, "depth": 8, "learning_rate": 0.03, "verbose": 0}
ES_ROUNDS = 200
ROUND_CAP = 9000


def prob_predict(model, pool):
    return model.predict(pool, prediction_type="Probability")[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    train, test = read_train(), read_test()
    y = train[TARGET].values.astype(np.int8)
    folds = fold_ids(y)

    for f in fold_list(a.folds):
        if have_part(HERE, PAIR, f):
            continue
        t0 = time.time()
        trn_idx = np.where(folds != f)[0]
        val_idx = np.where(folds == f)[0]
        y_trn = y[trn_idx].astype(np.float64)
        Xtr, Xte = fit_and_transform(train, test, trn_idx)
        cat_idx = [Xtr.columns.get_loc(c) for c in CAT_COLS]

        # inner 5-fold CV on the outer-train rows selects the round count
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED_INNER + f)
        best_iters, inner_aucs = [], []
        for i_fit, i_app in inner.split(np.zeros(len(trn_idx)), y_trn):
            m = CatBoost(params={**PARAMS, "iterations": ROUND_CAP})
            m.fit(Pool(Xtr.iloc[trn_idx[i_fit]], label=y_trn[i_fit], cat_features=cat_idx),
                  eval_set=Pool(Xtr.iloc[trn_idx[i_app]], label=y_trn[i_app], cat_features=cat_idx),
                  early_stopping_rounds=ES_ROUNDS)
            best_iters.append(m.get_best_iteration())
            p = prob_predict(m, Pool(Xtr.iloc[trn_idx[i_app]], cat_features=cat_idx))
            inner_aucs.append(roc_auc_score(y_trn[i_app], p))
        refit_rounds = int(round(float(np.mean(best_iters))))

        # refit on all outer-train rows at that round count, no evaluation set
        final = CatBoost(params={**PARAMS, "iterations": refit_rounds})
        final.fit(Pool(Xtr.iloc[trn_idx], label=y_trn, cat_features=cat_idx))
        pv = prob_predict(final, Pool(Xtr.iloc[val_idx], cat_features=cat_idx))
        pt = prob_predict(final, Pool(Xte, cat_features=cat_idx))
        save_part(HERE, PAIR, f, pv, pt)
        print(f'fold {f}: inner AUC {np.mean(inner_aucs):.6f}  refit {refit_rounds} rounds  '
              f'val AUC {roc_auc_score(y[val_idx], pv):.6f}  ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, PAIR, folds, y)


if __name__ == '__main__':
    main()
