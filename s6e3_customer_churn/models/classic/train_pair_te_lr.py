#!/usr/bin/env python
"""Pair target encodings -> logit polynomial expansion -> cuML logistic regression (two presets).

All 171 pairs of the 19 columns (every column label-encoded over train + test from its string value, so the numerics by
their exact value; missing values get their own code). Per outer fold and pair: cuML TargetEncoder of the pair's value
combination (mean target, no smoothing, random inner split seeded 42 + fold number): the outer-training rows get inner
out-of-fold means, the held-out and test rows the means over all outer-training rows -> 171 columns, clipped to
[1e-5, 1 - 1e-5], logit z, expanded to [z, z^2, z^3] (513 columns) or [z, z^2] (342 columns), standardized on the
outer-training rows. Model: cuML LogisticRegression (L2, C 0.5, max_iter 4000, tol 1e-4). The original rows are not
used.

Presets:
  lr_pair_te_logit3   [z, z^2, z^3]
  lr_pair_te_logit2   [z, z^2]
Usage: python train_pair_te_lr.py <preset>"""
import argparse
import sys
from pathlib import Path

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import FOLD_SEED, ID, N_FOLDS, TARGET, labels, model_dir, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

FOLDER = 'classic'
RANDOM_STATE = 42
TE_N_FOLDS = 5
EPS = 1e-5

PRESETS = {
    'lr_pair_te_logit3': dict(degree=3, test_folds=15, test_te_folds=10),
    'lr_pair_te_logit2': dict(degree=2, test_folds=15, test_te_folds=10),
}

LR_PARAMS = dict(penalty='l2', C=0.5, max_iter=4000, tol=1e-4, fit_intercept=True, verbose=0)


def label_encode_train_test(train_df, test_df, cols):
    """Per column: codes in order of first appearance over train + test (values as strings, missing = '__MISSING__')."""
    train_out = train_df.copy()
    test_out = test_df.copy()
    for c in cols:
        tr_s = train_out[c].astype('string').fillna('__MISSING__')
        te_s = test_out[c].astype('string').fillna('__MISSING__')
        uniq = pd.Index(pd.concat([tr_s, te_s], axis=0).unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        train_out[c] = tr_s.map(mapping).astype(np.int32)
        test_out[c] = te_s.map(mapping).astype(np.int32)
    return train_out, test_out


def logit_poly(m, degree, eps=EPS):
    """[z, z^2, ..., z^degree] with z = logit of m clipped to [eps, 1 - eps]."""
    m = np.clip(m, eps, 1.0 - eps)
    z = np.log(m / (1.0 - m)).astype(np.float32)
    return np.hstack([z ** p for p in range(1, degree + 1)]).astype(np.float32)


def cross_validate(X_all_g, X_test_g, y_all, pair_cols, degree, n_splits, te_n_folds):
    """Stratified n_splits-fold CV (shuffle, seed 42): out-of-fold predictions and mean test prediction."""
    n, n_test, n_pair = len(y_all), len(X_test_g), len(pair_cols)
    oof = np.zeros(n, dtype=np.float32)
    pred = np.zeros(n_test, dtype=np.float32)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=FOLD_SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.zeros(n), y_all), 1):
        X_tr_g = X_all_g.take(cudf.Series(tr_idx))
        X_va_g = X_all_g.take(cudf.Series(va_idx))
        y_tr = y_all[tr_idx].astype(np.int32)
        y_va = y_all[va_idx].astype(np.int32)
        y_tr_g = cudf.Series(y_tr)

        tr_pair = np.zeros((len(tr_idx), n_pair), dtype=np.float32)
        va_pair = np.zeros((len(va_idx), n_pair), dtype=np.float32)
        te_pair = np.zeros((n_test, n_pair), dtype=np.float32)
        te = TargetEncoder(n_folds=te_n_folds, smooth=0, seed=RANDOM_STATE + fold, split_method='random', stat='mean',
                           output_type='cupy')
        for t, (f1, f2) in enumerate(pair_cols):
            X_tr_ij_g = X_tr_g[[f1, f2]]
            tr_pair[:, t] = cp.asnumpy(te.fit_transform(X_tr_ij_g, y_tr_g)).ravel().astype(np.float32)
            te.fit(X_tr_ij_g, y_tr_g)
            va_pair[:, t] = cp.asnumpy(te.transform(X_va_g[[f1, f2]])).ravel().astype(np.float32)
            te_pair[:, t] = cp.asnumpy(te.transform(X_test_g[[f1, f2]])).ravel().astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(logit_poly(tr_pair, degree)).astype(np.float32)
        X_va = scaler.transform(logit_poly(va_pair, degree)).astype(np.float32)
        X_te = scaler.transform(logit_poly(te_pair, degree)).astype(np.float32)

        meta = cuLogReg(**LR_PARAMS)
        meta.fit(cp.asarray(X_tr), cp.asarray(y_tr))
        oof[va_idx] = cp.asnumpy(meta.predict_proba(cp.asarray(X_va))[:, 1]).astype(np.float32)
        pred += cp.asnumpy(meta.predict_proba(cp.asarray(X_te))[:, 1]).astype(np.float32) / n_splits
        print(f'  fold {fold}/{n_splits} ({X_tr.shape[1]} columns): AUC {roc_auc_score(y_va, oof[va_idx]):.6f}',
              flush=True)
    return oof, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preset', choices=sorted(PRESETS))
    args = ap.parse_args()
    pair, cfg = args.preset, PRESETS[args.preset]

    train, test = read_train(), read_test()
    y_all = labels(train[TARGET]).astype(np.int32)
    features = [c for c in train.columns if c not in (ID, TARGET)]
    train_enc, test_enc = label_encode_train_test(train[features], test[features], features)
    X_all_g = cudf.DataFrame({f: cudf.Series(train_enc[f].values).astype('int32') for f in features})
    X_test_g = cudf.DataFrame({f: cudf.Series(test_enc[f].values).astype('int32') for f in features})
    pair_cols = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    print(f'{pair}: {len(features)} columns, {len(pair_cols)} pairs', flush=True)

    oof, _ = cross_validate(X_all_g, X_test_g, y_all, pair_cols, cfg['degree'], N_FOLDS, TE_N_FOLDS)
    print(f'{pair}: OOF AUC {roc_auc_score(y_all, oof):.6f}', flush=True)
    _, test_pred = cross_validate(X_all_g, X_test_g, y_all, pair_cols, cfg['degree'], cfg['test_folds'],
                                  cfg['test_te_folds'])
    save_pair(model_dir(FOLDER), pair, oof, test_pred)


if __name__ == '__main__':
    main()
