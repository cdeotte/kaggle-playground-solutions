#!/usr/bin/env python
"""LightGBM (CUDA) on raw + derived + original-data rates + 36 nested exact-value target encodings (pair lgbm_exact_te36).

Features (lib/features.py): 9 raw numerics (NaN kept), 23 derived columns (slack, utilization, ratios, cents, repair flag,
ordinal categoricals, ...), 18 original-dataset per-value rates, 36 target encodings = 86 columns.
Per outer fold: the round count is chosen by early stopping (patience 100, at most 5000) on one third of the outer-train
rows (StratifiedKFold(3, shuffle=True, random_state=2026 + fold), first split; trained on the other two thirds), then the
model is refit on all outer-train rows for that many rounds. One seed (2026). Test = mean of the 5 fold models.
Usage: python train_lgbm.py [--folds 0,1,2,3,4]   (finished folds are kept in parts/ and skipped)"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as F  # noqa: E402

PAIR = 'lgbm_exact_te36'
SEED = 2026
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 255,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "device_type": "cuda",
    "verbosity": -1,
    "seed": SEED,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    todo = [f for f in fold_list(a.folds) if not have_part(HERE, PAIR, f)]

    t0 = time.time()
    train, test, y, folds = F.load_data()
    if todo:
        Xraw_tr, Xraw_te, te_train, te_test, te_names = F.build_all(train, test, y, folds)
        print(f'built features in {time.time() - t0:.1f}s ({Xraw_tr.shape[1]} raw+derived, {len(te_names)} TE)', flush=True)

    for f in todo:
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        Xtr = np.hstack([Xraw_tr.values, te_train[f]])
        Xte = np.hstack([Xraw_te.values, te_test[f]])

        es_skf = StratifiedKFold(3, shuffle=True, random_state=SEED + f)
        es_trn, es_val = next(iter(es_skf.split(np.zeros(len(trn)), y[trn])))
        itrn, ival = trn[es_trn], trn[es_val]

        dtrain = lgb.Dataset(Xtr[itrn], y[itrn])
        dvalid = lgb.Dataset(Xtr[ival], y[ival], reference=dtrain)
        model = lgb.train(PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dvalid], valid_names=["inner"],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        best_iter = model.best_iteration or NUM_BOOST_ROUND

        model_full = lgb.train(PARAMS, lgb.Dataset(Xtr[trn], y[trn]), num_boost_round=best_iter)
        pv = model_full.predict(Xtr[val])
        pt = model_full.predict(Xte)
        save_part(HERE, PAIR, f, pv, pt)
        print(f'fold {f}: best_iter={best_iter}  val AUC={roc_auc_score(y[val], pv):.6f} ({time.time() - t0:.0f}s)',
              flush=True)
    assemble(HERE, PAIR, folds, y)


if __name__ == '__main__':
    main()
