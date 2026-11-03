#!/usr/bin/env python
"""gcForest cascade (RandomForest + ExtraTrees per level) on the raw + derived + original-rate matrix and the nested
target encodings.

Inputs per fold: the 50 label-free columns of lib/features.build_all (9 raw numerics, 23 derived, 18 original-data
rates; median-imputed on train + test) and the fold's 36 nested target encodings.
Model (lib/gcforest.py): every level = RandomForest (seed 777 + fold) + ExtraTrees (seed 778 + fold), 200 trees each,
min_samples_leaf 5, max_features 0.5, full depth; at most 5 levels, patience 1. The cascade is grown on 90% of the
outer-train rows (first split of StratifiedKFold(10, seed 777 + fold)); the other 10% choose its depth. No refit on the
full outer-train. Test = mean of the 5 fold models.
Usage: python train.py [--folds 0,1,2,3,4] [--n_estimators 200] [--max_levels 5] [--min_samples_leaf 5]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as FEAT  # noqa: E402
from lib.gcforest import GCForest  # noqa: E402

NAME = 'gcforest_rf_et_te'
SEED = 777


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--n_estimators', type=int, default=200)
    ap.add_argument('--max_levels', type=int, default=5)
    ap.add_argument('--min_samples_leaf', type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    train, test, y, folds = FEAT.load_data()
    Xraw_tr, Xraw_te, te_train, te_test, te_names = FEAT.build_all(train, test, y, folds)
    print(f'built features in {time.time() - t0:.0f}s: {Xraw_tr.shape[1]} raw+derived+orig, {len(te_names)} TE',
          flush=True)

    # median imputation (label-free, fit on train + test): the forests do not take NaN
    imp = SimpleImputer(strategy='median')
    imp.fit(np.vstack([Xraw_tr.values, Xraw_te.values]))
    Xraw_i = imp.transform(Xraw_tr.values).astype(np.float32)
    Xraw_te_i = imp.transform(Xraw_te.values).astype(np.float32)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        X_fold = np.hstack([Xraw_i, te_train[f]]).astype(np.float32)
        X_test = np.hstack([Xraw_te_i, te_test[f]]).astype(np.float32)

        # holdout (10% of outer-train) that chooses the cascade depth
        es = StratifiedKFold(10, shuffle=True, random_state=SEED + f)
        grow, validate = next(iter(es.split(np.zeros(len(trn)), y[trn])))
        grow, validate = trn[grow], trn[validate]

        gc = GCForest(n_estimators=args.n_estimators, max_levels=args.max_levels,
                      min_samples_leaf=args.min_samples_leaf, seed=SEED + f)
        gc.fit(X_fold[grow], y[grow], X_fold[validate], y[validate])

        oof_val = gc.predict_proba(X_fold[val])
        save_part(HERE, NAME, f, oof_val, gc.predict_proba(X_test))
        print(f'fold {f}: best_level={gc.best_level} inner_auc={gc.best_auc:.6f} '
              f'val AUC={roc_auc_score(y[val], oof_val):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
