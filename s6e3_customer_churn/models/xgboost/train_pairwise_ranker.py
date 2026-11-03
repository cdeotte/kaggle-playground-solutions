#!/usr/bin/env python
"""GPU XGBoost pairwise ranker, Platt-scaled to probabilities (pair xgb_pairwise_ranker_platt).

Features: the 19 raw columns (id dropped); the 15 string columns as integer codes factorized over train + test; all
float32. Model: XGBRanker with objective rank:pairwise; the query groups are consecutive blocks of 256 rows of the
fold's training rows (validation rows likewise, for the AUC monitor); depth 6, lr 0.03, min_child_weight 1,
subsample / colsample_bytree 0.85, lambda 1, up to 4,000 rounds, early stopping 200 on the validation fold,
random_state 42. The raw ranking scores become probabilities through a logistic regression (C 50) fitted on the scores
of the fold's training rows.
Usage: python train_pairwise_ranker.py"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import ID, TARGET, labels, read_test, read_train  # noqa: E402
from common.pairs import save_pair  # noqa: E402

PAIR = 'xgb_pairwise_ranker_platt'
N_FOLDS = 5
TEST_FOLDS = 15
SEED = 42

GROUP_SIZE = 256
GROUP_SEED_OFFSET = 10_000
CALIB_C = 50.0

XGB_PARAMS = dict(
    objective='rank:pairwise',
    tree_method='hist',
    device='cuda',
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.0,
    n_estimators=4000,
    learning_rate=0.03,
    gamma=0.0,
    random_state=SEED,
    n_jobs=-1,
    eval_metric='auc',
    early_stopping_rounds=200,
)


def factorize_all_categoricals(train, test, drop_cols):
    """Integer codes of the object columns, factorized over train + test (no target used)."""
    Xtr = train.drop(columns=drop_cols, errors='ignore').copy()
    Xte = test.drop(columns=drop_cols, errors='ignore').copy()
    cat_cols = [c for c in Xtr.columns
                if Xtr[c].dtype == object or str(Xtr[c].dtype).startswith('string')
                or str(Xtr[c].dtype) == 'category' or Xtr[c].dtype == bool]
    for c in cat_cols:
        both = pd.concat([Xtr[c], Xte[c]], axis=0, ignore_index=True).astype('string')
        codes, _ = pd.factorize(both, sort=False)
        Xtr[c] = codes[:len(Xtr)].astype('int32')
        Xte[c] = codes[len(Xtr):].astype('int32')
    return Xtr.astype('float32', copy=False), Xte.astype('float32', copy=False), cat_cols


def make_random_groups(n, group_size, seed):
    """Group sizes (group_size each, a last group of 1 merged into the previous one) that sum to n."""
    assert group_size >= 2
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    sizes = []
    start = 0
    while start < n:
        end = min(start + group_size, n)
        sz = end - start
        if sz == 1 and len(sizes) > 0:
            sizes[-1] += 1
        else:
            sizes.append(sz)
        start = end
    return np.array(sizes, dtype=np.uint32)


def platt_fit(train_scores, y):
    clf = LogisticRegression(C=CALIB_C, solver='lbfgs', max_iter=2000, random_state=SEED)
    clf.fit(train_scores.reshape(-1, 1), y.astype(int))
    return clf


def platt_predict(clf, scores):
    return clf.predict_proba(scores.reshape(-1, 1))[:, 1].astype('float32')


def run_cv(X, y, X_test, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(X), dtype='float32')
    test_prob = np.zeros(len(X_test), dtype='float32')

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        g_tr = make_random_groups(len(tr_idx), GROUP_SIZE, seed=SEED + GROUP_SEED_OFFSET + fold)
        g_va = make_random_groups(len(va_idx), GROUP_SIZE, seed=SEED + GROUP_SEED_OFFSET + 100 + fold)

        model = XGBRanker(**XGB_PARAMS)
        model.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va], verbose=200)

        va_s = model.predict(X_va).astype('float32')
        tr_s = model.predict(X_tr).astype('float32')
        te_s = model.predict(X_test).astype('float32')

        calib = platt_fit(tr_s, y_tr)
        va_p = platt_predict(calib, va_s)
        te_p = platt_predict(calib, te_s)
        oof_prob[va_idx] = va_p
        test_prob += te_p / n_folds

        print(f'{n_folds} folds, fold {fold}: AUC raw {roc_auc_score(y_va, va_s):.6f}, calibrated '
              f'{roc_auc_score(y_va, va_p):.6f}, best_iteration {model.best_iteration}', flush=True)
        del model, calib
        gc.collect()

    print(f'{n_folds} folds: OOF AUC {roc_auc_score(y, oof_prob):.6f}', flush=True)
    return oof_prob, test_prob


def main():
    train = read_train()
    test = read_test()
    y = labels(train[TARGET])
    X, X_test, cat_cols = factorize_all_categoricals(train, test, drop_cols=[TARGET, ID])
    print(f'train {X.shape}, test {X_test.shape}, {len(cat_cols)} categorical columns', flush=True)

    oof, _ = run_cv(X, y, X_test, N_FOLDS)
    _, test_pred = run_cv(X, y, X_test, TEST_FOLDS)
    save_pair(HERE, PAIR, oof, test_pred)


if __name__ == '__main__':
    main()
