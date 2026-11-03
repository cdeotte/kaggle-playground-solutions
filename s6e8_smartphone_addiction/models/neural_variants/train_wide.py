#!/usr/bin/env python
"""Wide linear model: elastic-net logistic regression (SGD) on target encodings, their pairwise products, the
raw + derived columns and the original-data rates.

Inputs per fold: the 32 raw + derived columns of lib/features.build_raw_derived (median-imputed on train + test), the
18 original-data rates, the fold's 36 nested target encodings and the 36 pairwise products of the 9 single-numeric
target encodings (122 columns), standardized on the outer-train rows.
Model: SGDClassifier(loss='log_loss', penalty='elasticnet', alpha 1e-4, l1_ratio 0.9, max_iter 2000, tol 1e-4,
random_state 777), fit on all outer-train rows. Test = mean of the 5 fold models.
Usage: python train_wide.py [--folds 0,1,2,3,4] [--alpha 1e-4] [--l1_ratio 0.9]"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from common.pairs import assemble, fold_list, have_part, save_part  # noqa: E402
from lib import features as FEAT  # noqa: E402

NAME = 'wide_elasticnet_te'
SEED = 777


def build_base(train, test):
    """Label-free base columns: raw + derived (median-imputed, fit on train + test) and the original-data rates."""
    Xraw_tr, Xraw_te = FEAT.build_raw_derived(train, test)
    Xorig_tr, Xorig_te = FEAT.build_original_rates(train, test)
    imp = SimpleImputer(strategy='median')
    imp.fit(np.vstack([Xraw_tr.values, Xraw_te.values]))
    d_tr = imp.transform(Xraw_tr.values).astype(np.float32)
    d_te = imp.transform(Xraw_te.values).astype(np.float32)
    return d_tr, d_te, Xorig_tr.values, Xorig_te.values


def fold_matrix(f, d_tr, d_te, o_tr, o_te, te_train, te_test, single_idx):
    base_tr = np.hstack([d_tr, o_tr, te_train[f]])
    base_te = np.hstack([d_te, o_te, te_test[f]])
    inter_tr = np.column_stack([te_train[f][:, single_idx[i]] * te_train[f][:, single_idx[j]]
                                for i in range(9) for j in range(i + 1, 9)])
    inter_te = np.column_stack([te_test[f][:, single_idx[i]] * te_test[f][:, single_idx[j]]
                                for i in range(9) for j in range(i + 1, 9)])
    return (np.hstack([base_tr, inter_tr]).astype(np.float32),
            np.hstack([base_te, inter_te]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--alpha', type=float, default=1e-4)
    ap.add_argument('--l1_ratio', type=float, default=0.9)
    args = ap.parse_args()

    t0 = time.time()
    train, test, y, folds = FEAT.load_data()
    te_train, te_test, te_names = FEAT.build_te_features(train, test, y, folds)
    d_tr, d_te, o_tr, o_te = build_base(train, test)
    single_idx = [te_names.index(f'te_{c}') for c in FEAT.NUMERICS]
    print(f'built features in {time.time() - t0:.0f}s: {len(te_names)} TE + {d_tr.shape[1]} raw+derived + '
          f'{o_tr.shape[1]} orig + 36 interactions', flush=True)

    for f in fold_list(args.folds):
        if have_part(HERE, NAME, f):
            continue
        trn = np.where(folds != f)[0]
        val = np.where(folds == f)[0]
        X_fold, X_test = fold_matrix(f, d_tr, d_te, o_tr, o_te, te_train, te_test, single_idx)
        scaler = StandardScaler().fit(X_fold[trn])
        Xs = scaler.transform(X_fold)
        Xts = scaler.transform(X_test)

        m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=args.alpha, l1_ratio=args.l1_ratio,
                          max_iter=2000, tol=1e-4, random_state=SEED)
        m.fit(Xs[trn], y[trn])
        oof_val = m.predict_proba(Xs[val])[:, 1]
        save_part(HERE, NAME, f, oof_val, m.predict_proba(Xts)[:, 1])
        print(f'fold {f}: val AUC={roc_auc_score(y[val], oof_val):.6f} ({time.time() - t0:.0f}s)', flush=True)
    assemble(HERE, NAME, folds, y)


if __name__ == '__main__':
    main()
