#!/usr/bin/env python
"""Pair gpt2_field_pseudolikelihood: nested XGBoost (GPU) on the 1,330 features of build_field_features.py.

Per outer fold: 4 candidates (feature family x depth / regularization, CANDIDATES) are trained on each inner split of the other 4
folds with early stopping (75 rounds, at most 1,500) on the inner held-out fold; the candidate with the best inner OOF AUC on the
outer-train rows is refit on all of them for ceil(4/3 x its median inner best round count) rounds (between 25 and 1,500) and
predicts the held-out fold and the test rows. OOF = held-out probabilities; test = mean of the 5 fold models; both clipped to
[1e-7, 1 - 1e-7].
Usage: python fit_field_xgb.py [--folds 0,1,2,3,4]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.data import folds, target  # noqa: E402
from common.gpt_rows import assemble_pair  # noqa: E402
from common.pairs import fold_list, have_part, save_part  # noqa: E402

NAME = 'gpt2_field_pseudolikelihood'
MAX_ROUNDS, EARLY_STOPPING, OUTER_INNER_ROUND_SCALE = 1500, 75, 4.0 / 3.0
CANDIDATES = [
    dict(name='compact_contrast_d3', family='compact_contrast', max_depth=3, eta=0.04, min_child_weight=64, subsample=0.90,
         colsample_bytree=0.90, reg_lambda=12.0, reg_alpha=0.10),
    dict(name='contrast_d3', family='contrast', max_depth=3, eta=0.04, min_child_weight=48, subsample=0.90, colsample_bytree=0.85,
         reg_lambda=12.0, reg_alpha=0.10),
    dict(name='contrast_d4', family='contrast', max_depth=4, eta=0.03, min_child_weight=64, subsample=0.90, colsample_bytree=0.80,
         reg_lambda=16.0, reg_alpha=0.15),
    dict(name='full_d3', family='full', max_depth=3, eta=0.035, min_child_weight=80, subsample=0.90, colsample_bytree=0.75,
         reg_lambda=20.0, reg_alpha=0.20),
]


def parameters(c, seed):
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'device': 'cuda', 'max_bin': 256,
            'max_depth': c['max_depth'], 'eta': c['eta'], 'min_child_weight': c['min_child_weight'], 'subsample': c['subsample'],
            'colsample_bytree': c['colsample_bytree'], 'lambda': c['reg_lambda'], 'alpha': c['reg_alpha'], 'seed': seed, 'verbosity': 0}


def materialize(matrix, rows, columns):
    return np.ascontiguousarray(matrix[rows][:, np.asarray(columns, np.int32)], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0,1,2,3,4')
    a = ap.parse_args()
    y = target()
    f = folds(y)
    n_train = len(y)
    matrix = np.load(HERE / 'features' / 'field_features.npy', mmap_mode='r')
    families = json.loads((HERE / 'features' / 'field_feature_families.json').read_text())
    test_rows = np.arange(n_train, len(matrix), dtype=np.int64)
    family_order = list(dict.fromkeys(c['family'] for c in CANDIDATES))
    for outer in fold_list(a.folds):
        if have_part(HERE, NAME, outer):
            continue
        train_rows, valid_rows = np.flatnonzero(f != outer), np.flatnonzero(f == outer)
        local = np.full(n_train, -1, np.int64)
        local[train_rows] = np.arange(len(train_rows))
        inner_pred = {c['name']: np.full(len(train_rows), np.nan) for c in CANDIDATES}
        inner_rounds = {c['name']: [] for c in CANDIDATES}
        for inner in [v for v in range(5) if v != outer]:
            fit_rows, inner_valid = np.flatnonzero((f != outer) & (f != inner)), np.flatnonzero(f == inner)
            for family in family_order:
                dfit = xgb.DMatrix(materialize(matrix, fit_rows, families[family]), label=y[fit_rows])
                dvalid = xgb.DMatrix(materialize(matrix, inner_valid, families[family]), label=y[inner_valid])
                for i, c in enumerate(CANDIDATES):
                    if c['family'] != family:
                        continue
                    model = xgb.train(parameters(c, 42_000 + outer * 1_000 + inner * 100 + i), dfit, num_boost_round=MAX_ROUNDS,
                                      evals=[(dvalid, 'valid')], early_stopping_rounds=EARLY_STOPPING, verbose_eval=False)
                    best = int(model.best_iteration) + 1
                    inner_pred[c['name']][local[inner_valid]] = model.predict(dvalid, iteration_range=(0, best)).astype(np.float64)
                    inner_rounds[c['name']].append(best)
                    del model
                del dfit, dvalid
        auc = {c['name']: roc_auc_score(y[train_rows], inner_pred[c['name']]) for c in CANDIDATES}
        i = max(range(len(CANDIDATES)), key=lambda j: auc[CANDIDATES[j]['name']])
        winner = CANDIDATES[i]
        rounds = int(np.clip(np.ceil(float(np.median(inner_rounds[winner['name']])) * OUTER_INNER_ROUND_SCALE), 25, MAX_ROUNDS))
        print(f'fold {outer}: selected {winner["name"]} (inner OOF AUC {auc[winner["name"]]:.6f}), {rounds} rounds', flush=True)
        cols = families[winner['family']]
        model = xgb.train(parameters(winner, 84_000 + outer * 100 + i),
                          xgb.DMatrix(materialize(matrix, train_rows, cols), label=y[train_rows]), num_boost_round=rounds, verbose_eval=False)
        valid_pred = model.predict(xgb.DMatrix(materialize(matrix, valid_rows, cols))).astype(np.float64)
        test_pred = model.predict(xgb.DMatrix(materialize(matrix, test_rows, cols))).astype(np.float64)
        save_part(HERE, NAME, outer, valid_pred, test_pred)
        print(f'fold {outer}: AUC {roc_auc_score(y[valid_rows], valid_pred):.6f}', flush=True)
    assemble_pair(HERE, NAME, f, y, clip=1e-7)


if __name__ == '__main__':
    main()
