#!/usr/bin/env python
"""RuleFit on the raw + derived columns, the original-data rates and the nested target encodings.

Inputs per fold: the 32 raw + derived columns of lib/features.build_raw_derived (median-imputed on train + test), the
18 original-data rates and the fold's 36 nested target encodings (86 columns).
Model (lib/rulefit.py): rules from a RandomForest (seed 777 + fold) + ExtraTrees (seed 778 + fold), 20 trees each,
max_depth 3, min_samples_leaf 10, max_features 0.5 (about 320 distinct rules per fold); then SGD elastic-net logistic
regression (alpha 1e-4, l1_ratio 0.9, max_iter 500, tol 1e-4) on [86 columns, rule firings], standardized on the
outer-train rows. Fit on all outer-train rows. Test = mean of the 5 fold models.
Usage: python train_rulefit.py [--folds 0,1,2,3,4] [--n_estimators 20] [--max_depth 3]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as FEAT  # noqa: E402
from lib.rulefit import RuleFit  # noqa: E402

NAME = 'rulefit_rf_et_te'
SEED = 777


def build_base(train, test):
    """Label-free base columns: raw + derived (median-imputed, fit on train + test) and the original-data rates."""
    Xraw_tr, Xraw_te = FEAT.build_raw_derived(train, test)
    Xorig_tr, Xorig_te = FEAT.build_original_rates(train, test)
    imp = SimpleImputer(strategy='median')
    imp.fit(np.vstack([Xraw_tr.values, Xraw_te.values]))
    d_tr = imp.transform(Xraw_tr.values).astype(np.float32)
    d_te = imp.transform(Xraw_te.values).astype(np.float32)
    return d_tr, d_te, Xorig_tr.values.astype(np.float32), Xorig_te.values.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--n_estimators', type=int, default=20)
    ap.add_argument('--max_depth', type=int, default=3)
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--l1_ratio', type=float, default=0.9)
    args = ap.parse_args()

    t0 = time.time()
    train, test, y, folds = FEAT.load_data()
    te_train, te_test, te_names = FEAT.build_te_features(train, test, y, folds)
    d_tr, d_te, o_tr, o_te = build_base(train, test)
    print(f'built features in {time.time() - t0:.0f}s: {len(te_names)} TE + {d_tr.shape[1]} raw+derived + '
          f'{o_tr.shape[1]} orig', flush=True)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        X_fold = np.hstack([d_tr, o_tr, te_train[f]]).astype(np.float32)
        X_test = np.hstack([d_te, o_te, te_test[f]]).astype(np.float32)

        rf = RuleFit(n_estimators=args.n_estimators, max_depth=args.max_depth, alpha=args.alpha,
                     l1_ratio=args.l1_ratio, seed=SEED + f)
        rf.fit(X_fold[trn], y[trn])
        oof_val = rf.predict_proba(X_fold[val])
        save_part(HERE, NAME, f, oof_val, rf.predict_proba(X_test))
        print(f'fold {f}: val AUC={roc_auc_score(y[val], oof_val):.6f} ({len(rf.rules)} rules, '
              f'{time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
